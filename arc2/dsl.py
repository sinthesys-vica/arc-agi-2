"""Small, deterministic transformation DSL for ARC grids.

Owner: Atlas

Every primitive is a pure ``grid -> grid`` function. Functions carry a stable
program signature so search can deduplicate candidates without descriptions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .scene import Scene


Transform = Callable[[np.ndarray], np.ndarray]

_IDENTITY_SIGNATURE = ("identity",)
_IDENTITY_EQUIVALENTS = frozenset({
    ("tile", 1, 1),
    ("scale", 1, 1),
    ("rotate", 0),
})


def _grid(value: np.ndarray | list[list[int]]) -> np.ndarray:
    arr = np.asarray(value, dtype=int)
    if arr.ndim != 2 or not arr.size:
        raise ValueError("transform input must be a non-empty 2D grid")
    return arr


def _decorate(transform: Transform, name: str, *params: Any) -> Transform:
    transform.__name__ = name
    transform.__qualname__ = name
    transform._arc2_signature = (name, *params)  # type: ignore[attr-defined]
    return transform


def _canonical_signature(stored: tuple) -> tuple:
    """Return the semantic normal form of a stored program signature."""

    if stored == _IDENTITY_SIGNATURE or stored in _IDENTITY_EQUIVALENTS:
        return _IDENTITY_SIGNATURE
    if stored and stored[0] == "compose":
        components = tuple(
            _canonical_signature(tuple(component)) for component in stored[1:]
        )
        components = tuple(
            component for component in components if component != _IDENTITY_SIGNATURE
        )
        if not components:
            return _IDENTITY_SIGNATURE
        if len(components) == 1:
            return components[0]
        return ("compose", *components)
    return stored


def signature(transform: Transform) -> tuple:
    """Return a stable, semantically canonical program signature."""

    stored = getattr(transform, "_arc2_signature", None)
    if stored is not None:
        return _canonical_signature(tuple(stored))
    return (getattr(transform, "__qualname__", repr(transform)),)


def identity(grid: np.ndarray) -> np.ndarray:
    """Return an independent copy of the input."""

    return _grid(grid).copy()


_decorate(identity, "identity")


def compose(*transforms: Transform) -> Transform:
    """Compose transforms left-to-right: ``compose(f, g)(x) = g(f(x))``."""

    transforms = tuple(
        transform
        for transform in transforms
        if signature(transform) != _IDENTITY_SIGNATURE
    )
    if not transforms:
        return identity
    if len(transforms) == 1:
        return transforms[0]

    def composed(grid: np.ndarray) -> np.ndarray:
        result = _grid(grid)
        for transform in transforms:
            result = _grid(transform(result))
        return result.copy()

    composed._arc2_components = transforms  # type: ignore[attr-defined]
    return _decorate(composed, "compose", *(signature(t) for t in transforms))


def rotate(turns: int) -> Transform:
    """Rotate counter-clockwise by ``turns * 90`` degrees."""

    normalized = turns % 4

    def transform(grid: np.ndarray) -> np.ndarray:
        return np.rot90(_grid(grid), normalized).copy()

    return _decorate(transform, "rotate", normalized)


def flip_lr(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(_grid(grid)).copy()


def flip_ud(grid: np.ndarray) -> np.ndarray:
    return np.flipud(_grid(grid)).copy()


def transpose(grid: np.ndarray) -> np.ndarray:
    return _grid(grid).T.copy()


def anti_transpose(grid: np.ndarray) -> np.ndarray:
    """Reflect across the anti-diagonal."""

    return np.rot90(_grid(grid), 2).T.copy()


for _function, _name in (
    (flip_lr, "flip_lr"),
    (flip_ud, "flip_ud"),
    (transpose, "transpose"),
    (anti_transpose, "anti_transpose"),
):
    _decorate(_function, _name)


def d4_transforms() -> dict[str, Transform]:
    """The eight symmetries of a square, also valid on rectangles."""

    return {
        "identity": identity,
        "rotate_90": rotate(1),
        "rotate_180": rotate(2),
        "rotate_270": rotate(3),
        "flip_lr": flip_lr,
        "flip_ud": flip_ud,
        "transpose": transpose,
        "anti_transpose": anti_transpose,
    }


def recolor(mapping: Mapping[int, int]) -> Transform:
    """Apply a simultaneous color mapping, including safe color swaps."""

    normalized = tuple(sorted((int(source), int(target)) for source, target in mapping.items()))
    if any(source not in range(10) or target not in range(10) for source, target in normalized):
        raise ValueError("ARC colors must be integers in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        result = source.copy()
        for old, new in normalized:
            result[source == old] = new
        return result

    return _decorate(transform, "recolor", normalized)


def tile(rows: int, cols: int) -> Transform:
    if rows < 1 or cols < 1:
        raise ValueError("tile factors must be positive")

    def transform(grid: np.ndarray) -> np.ndarray:
        return np.tile(_grid(grid), (rows, cols))

    return _decorate(transform, "tile", rows, cols)


def scale(rows: int, cols: int | None = None) -> Transform:
    """Replace each cell with a solid ``rows x cols`` block."""

    cols = rows if cols is None else cols
    if rows < 1 or cols < 1:
        raise ValueError("scale factors must be positive")

    def transform(grid: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(_grid(grid), rows, axis=0), cols, axis=1)

    return _decorate(transform, "scale", rows, cols)


def crop_bbox(r0: int, c0: int, r1: int, c1: int) -> Transform:
    """Crop to an inclusive bounding box."""

    if min(r0, c0) < 0 or r1 < r0 or c1 < c0:
        raise ValueError("invalid bounding box")

    def transform(grid: np.ndarray) -> np.ndarray:
        arr = _grid(grid)
        if r1 >= arr.shape[0] or c1 >= arr.shape[1]:
            raise ValueError("bounding box exceeds grid")
        return arr[r0:r1 + 1, c0:c1 + 1].copy()

    return _decorate(transform, "crop_bbox", r0, c0, r1, c1)


def crop_non_background(grid: np.ndarray) -> np.ndarray:
    """Crop to the inferred non-background bounding box."""

    scene = Scene(_grid(grid))
    bbox = scene.non_background_bbox()
    return scene.grid.copy() if bbox is None else scene.crop(bbox)


_decorate(crop_non_background, "crop_non_background")


def keep_colors(colors: set[int] | tuple[int, ...], background: int = 0) -> Transform:
    """Replace every color outside ``colors`` with ``background``."""

    kept = tuple(sorted(int(color) for color in colors))
    if any(color not in range(10) for color in (*kept, background)):
        raise ValueError("ARC colors must be integers in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        arr = _grid(grid)
        return np.where(np.isin(arr, kept), arr, background).astype(int, copy=False)

    return _decorate(transform, "keep_colors", kept, background)


def largest_component_histogram(
    order: str = "left_to_right",
    layout: str = "rows",
) -> Transform:
    """Render the colors of all largest components as a solid histogram.

    The selected component size becomes the repeated dimension. This captures
    a reusable object-selection family without baking in a task identifier.
    """

    orders = {
        "left_to_right": lambda obj: (obj.bbox.c0, obj.bbox.r0, obj.color),
        "right_to_left": lambda obj: (-obj.bbox.c1, obj.bbox.r0, obj.color),
        "top_to_bottom": lambda obj: (obj.bbox.r0, obj.bbox.c0, obj.color),
        "bottom_to_top": lambda obj: (-obj.bbox.r1, obj.bbox.c0, obj.color),
    }
    if order not in orders:
        raise ValueError(f"unsupported component order: {order}")
    if layout not in {"rows", "columns"}:
        raise ValueError("layout must be 'rows' or 'columns'")

    def transform(grid: np.ndarray) -> np.ndarray:
        objects = Scene(_grid(grid)).objects()
        if not objects:
            raise ValueError("largest-component histogram requires foreground objects")
        largest = max(obj.size for obj in objects)
        selected = sorted(
            (obj for obj in objects if obj.size == largest),
            key=orders[order],
        )
        palette = np.array([obj.color for obj in selected], dtype=int)
        if layout == "rows":
            return np.tile(palette[np.newaxis, :], (largest, 1))
        return np.tile(palette[:, np.newaxis], (1, largest))

    return _decorate(transform, "largest_component_histogram", order, layout)


def routed_ribbon(
    anchor_color: int,
    row_direction: int,
    column_direction: int,
    width: int = 2,
    background: int | None = None,
) -> Transform:
    """Grow a diagonal ribbon through background cells around obstacles.

    A singleton anchor starts the ribbon. Each next row is scanned in the
    horizontal direction from the previous leading edge. Obstacles are left
    untouched, while the previous row is widened to join a detour.
    """

    if anchor_color not in range(10):
        raise ValueError("anchor color must be in [0, 9]")
    if row_direction not in {-1, 1} or column_direction not in {-1, 1}:
        raise ValueError("ribbon directions must be -1 or 1")
    if width < 1:
        raise ValueError("ribbon width must be positive")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        anchors = np.argwhere(source == anchor_color)
        if len(anchors) != 1:
            raise ValueError("routed ribbon requires exactly one anchor cell")

        output = source.copy()
        rows, columns = source.shape
        previous_row, scan_start = (int(value) for value in anchors[0])
        row = previous_row + row_direction

        while 0 <= row < rows:
            column_range = (
                range(scan_start, columns)
                if column_direction == 1
                else range(scan_start, -1, -1)
            )
            painted: list[int] = []
            for column in column_range:
                if int(source[row, column]) == bg:
                    output[row, column] = anchor_color
                    painted.append(column)
                    if len(painted) == width:
                        break
            if not painted:
                break

            previous_cells = np.flatnonzero(output[previous_row] == anchor_color)
            if not len(previous_cells):
                raise ValueError("ribbon lost its previous segment")
            previous_lead = (
                int(previous_cells.max())
                if column_direction == 1
                else int(previous_cells.min())
            )
            for column in range(
                previous_lead + column_direction,
                painted[0] + column_direction,
                column_direction,
            ):
                if int(source[previous_row, column]) == bg:
                    output[previous_row, column] = anchor_color

            previous_row = row
            scan_start = painted[-1]
            if len(painted) < width:
                break
            row += row_direction

        return output

    return _decorate(
        transform,
        "routed_ribbon",
        anchor_color,
        row_direction,
        column_direction,
        width,
        background,
    )


def nested_rectangle_miniature() -> Transform:
    """Compress nested rectangular regions to one-cell-thick rings.

    Repeated colors are deliberately preserved when they occur at distinct
    nesting depths.  The output for ``n`` levels is ``(2n - 1)`` square.
    """

    def transform(grid: np.ndarray) -> np.ndarray:
        palette = Scene(_grid(grid)).nested_rectangular_palette()
        size = 2 * len(palette) - 1
        result = np.full((size, size), palette[-1], dtype=int)
        for depth, color in enumerate(palette):
            result[depth:size - depth, depth:size - depth] = color
        return result

    return _decorate(transform, "nested_rectangle_miniature")


def d4_orbit_consensus(hole_color: int) -> Transform:
    """Fill masked cells from the unique modal color in their D4 orbit."""

    if hole_color not in range(10):
        raise ValueError("hole color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape[0] != source.shape[1]:
            raise ValueError("D4 orbit consensus requires a square grid")
        size = source.shape[0]
        result = source.copy()
        holes = np.argwhere(source == hole_color)
        for row_value, column_value in holes:
            row, column = int(row_value), int(column_value)
            orbit = {
                (row, column),
                (row, size - 1 - column),
                (size - 1 - row, column),
                (size - 1 - row, size - 1 - column),
                (column, row),
                (column, size - 1 - row),
                (size - 1 - column, row),
                (size - 1 - column, size - 1 - row),
            }
            counts = Counter(
                int(source[rr, cc])
                for rr, cc in orbit
                if int(source[rr, cc]) != hole_color
            )
            if not counts:
                raise ValueError("D4 orbit contains no observed color")
            best_count = max(counts.values())
            winners = [color for color, count in counts.items() if count == best_count]
            if len(winners) != 1:
                raise ValueError("D4 orbit consensus is ambiguous")
            result[row, column] = winners[0]
        return result

    return _decorate(transform, "d4_orbit_consensus", hole_color)


def self_similar_stamp(marker_color: int, background: int | None = None) -> Transform:
    """Reconstruct a grid of repeated motifs from the motif's own mask.

    A square motif is repeated on a square lattice.  Full-background lines on
    one axis expose the lattice stride; opaque distractor blocks on the other
    axis may hide repetitions.  The motif's non-background mask determines
    which lattice cells exist, and each restored copy receives ``marker_color``
    at the local coordinate matching its lattice coordinate.
    """

    if marker_color not in range(10):
        raise ValueError("marker color must be in [0, 9]")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape[0] != source.shape[1]:
            raise ValueError("self-similar stamping requires a square grid")
        bg = Scene(source).background_color() if background is None else background
        separator_rows = [
            index for index, row in enumerate(source) if np.all(row == bg)
        ]
        separator_cols = [
            index for index, column in enumerate(source.T) if np.all(column == bg)
        ]
        separators = separator_rows or separator_cols
        if not separators or separators[0] < 1:
            raise ValueError("no full-background lattice separator found")

        block_size = separators[0]
        stride = block_size + 1
        size = source.shape[0]
        if (size + 1) % stride:
            raise ValueError("grid does not fit a regular motif lattice")
        lattice_size = (size + 1) // stride
        expected_separators = [stride * index - 1 for index in range(1, lattice_size)]
        if separators != expected_separators or lattice_size != block_size:
            raise ValueError("motif and lattice dimensions do not agree")

        blocks: list[np.ndarray] = []
        for grid_row in range(lattice_size):
            for grid_column in range(lattice_size):
                row0, column0 = grid_row * stride, grid_column * stride
                block = source[
                    row0:row0 + block_size,
                    column0:column0 + block_size,
                ]
                values = set(int(value) for value in np.unique(block))
                if any(value != bg for value in values) and len(values) > 1:
                    blocks.append(block)
        if not blocks:
            raise ValueError("no non-uniform motif block found")

        frequencies = Counter(tuple(int(value) for value in block.flat) for block in blocks)
        motif_key, frequency = frequencies.most_common(1)[0]
        if list(frequencies.values()).count(frequency) != 1:
            raise ValueError("motif selection is ambiguous")
        motif = np.asarray(motif_key, dtype=int).reshape(block_size, block_size)

        result = np.full(source.shape, bg, dtype=int)
        for grid_row in range(lattice_size):
            for grid_column in range(lattice_size):
                if int(motif[grid_row, grid_column]) == bg:
                    continue
                row0, column0 = grid_row * stride, grid_column * stride
                result[
                    row0:row0 + block_size,
                    column0:column0 + block_size,
                ] = motif
                result[row0 + grid_row, column0 + grid_column] = marker_color
        return result

    return _decorate(transform, "self_similar_stamp", marker_color, background)


def block_analogy(background: int | None = None) -> Transform:
    """Apply each demonstrated A-to-B block rule to its neighboring C block.

    Full-background columns delimit three equal block columns (A, B, C), and
    full-background rows delimit independent examples.  A-to-B is inferred as
    a D4 transform when possible, otherwise as a position-wise color map.
    """

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    priority = (
        "flip_lr",
        "rotate_270",
        "rotate_90",
        "flip_ud",
        "rotate_180",
        "identity",
        "transpose",
        "anti_transpose",
    )

    def runs(indices: list[int]) -> list[tuple[int, int]]:
        if not indices:
            return []
        result: list[tuple[int, int]] = []
        start = previous = indices[0]
        for index in indices[1:]:
            if index != previous + 1:
                result.append((start, previous + 1))
                start = index
            previous = index
        result.append((start, previous + 1))
        return result

    def infer_and_apply(source: np.ndarray, target: np.ndarray, query: np.ndarray) -> np.ndarray:
        d4 = d4_transforms()
        matching = [
            name for name in priority
            if d4[name](source).shape == target.shape
            and np.array_equal(d4[name](source), target)
        ]
        if matching:
            return d4[matching[0]](query)

        mapping: dict[int, int] = {}
        if source.shape != target.shape:
            raise ValueError("analogy color map requires equal demonstration shapes")
        for old, new in zip(source.flat, target.flat):
            old_i, new_i = int(old), int(new)
            previous = mapping.setdefault(old_i, new_i)
            if previous != new_i:
                raise ValueError("analogy color map is inconsistent")
        if any(int(value) not in mapping for value in query.flat):
            raise ValueError("analogy query contains an unmapped color")
        return recolor(mapping)(query)

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        active_columns = [
            index for index, column in enumerate(source.T) if not np.all(column == bg)
        ]
        column_runs = runs(active_columns)
        if len(column_runs) != 3:
            raise ValueError("block analogy requires exactly three block columns")
        widths = {end - start for start, end in column_runs}
        if len(widths) != 1:
            raise ValueError("analogy block columns must have equal width")
        output_width = widths.pop()

        result_rows: list[np.ndarray] = []
        row = 0
        while row < source.shape[0]:
            if np.all(source[row] == bg):
                result_rows.append(np.full(output_width, bg, dtype=int))
                row += 1
                continue
            end_row = row
            while end_row < source.shape[0] and not np.all(source[end_row] == bg):
                end_row += 1
            blocks = [source[row:end_row, start:end] for start, end in column_runs]
            answer = infer_and_apply(blocks[0], blocks[1], blocks[2])
            if answer.shape != (end_row - row, output_width):
                raise ValueError("analogy answer shape differs from its demonstration block")
            result_rows.extend(answer)
            row = end_row
        return np.vstack(result_rows)

    return _decorate(transform, "block_analogy", background)
