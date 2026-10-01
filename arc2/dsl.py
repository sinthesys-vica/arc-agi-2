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


def _active_runs(size: int, separators: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return half-open non-separator spans along one axis."""

    separator_set = set(separators)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(size):
        if index in separator_set:
            if start is not None:
                runs.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        runs.append((start, size))
    return tuple(runs)


def _regular_partition(
    grid: np.ndarray,
) -> tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]:
    """Detect a strict rectangular partition made of non-background lines.

    The detector deliberately fails closed.  All separator rows and columns
    must have one shared color, that color may not occur in content cells, and
    the resulting spans must be regular on each split axis.  Background-only
    gaps therefore do not masquerade as separators.
    """

    source = _grid(grid)
    uniform_rows = [
        (index, int(row[0]))
        for index, row in enumerate(source)
        if np.all(row == row[0])
    ]
    uniform_cols = [
        (index, int(column[0]))
        for index, column in enumerate(source.T)
        if np.all(column == column[0])
    ]
    colors = {color for _, color in (*uniform_rows, *uniform_cols)}
    candidates = []
    for separator_color in sorted(colors):
        rows = tuple(index for index, color in uniform_rows if color == separator_color)
        columns = tuple(index for index, color in uniform_cols if color == separator_color)
        if any(index in {0, source.shape[0] - 1} for index in rows):
            continue
        if any(index in {0, source.shape[1] - 1} for index in columns):
            continue

        row_spans = _active_runs(source.shape[0], rows)
        column_spans = _active_runs(source.shape[1], columns)
        if len(row_spans) == 1 and len(column_spans) == 1:
            continue
        if any(
            len(spans) > 1 and len({end - start for start, end in spans}) != 1
            for spans in (row_spans, column_spans)
        ):
            continue

        separator_mask = np.zeros(source.shape, dtype=bool)
        if rows:
            separator_mask[list(rows), :] = True
        if columns:
            separator_mask[:, list(columns)] = True
        if np.any((source == separator_color) & ~separator_mask):
            continue

        content_values = source[~separator_mask]
        content_colors = [int(value) for value in np.unique(content_values)]
        border_mask = np.zeros(source.shape, dtype=bool)
        border_mask[0, :] = border_mask[-1, :] = True
        border_mask[:, 0] = border_mask[:, -1] = True
        background = min(
            content_colors,
            key=lambda color: (
                -int(np.count_nonzero(content_values == color)),
                -int(np.count_nonzero((source == color) & border_mask & ~separator_mask)),
                color != 0,
                color,
            ),
        )
        if background == separator_color:
            continue
        candidates.append((
            background,
            separator_color,
            row_spans,
            column_spans,
            rows,
            columns,
        ))
    if len(candidates) != 1:
        raise ValueError("partition requires one unambiguous separator color")
    return candidates[0]


def _partition_canvas(
    source: np.ndarray,
    background: int,
    rows: tuple[int, ...],
    columns: tuple[int, ...],
) -> np.ndarray:
    """Create a cleared grid while preserving separator lines exactly."""

    result = np.full(source.shape, background, dtype=int)
    if rows:
        result[list(rows), :] = source[list(rows), :]
    if columns:
        result[:, list(columns)] = source[:, list(columns)]
    return result


def partition_blueprint_expand() -> Transform:
    """Expand the uniquely incomplete block as a colored lattice blueprint."""

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg, _, row_spans, col_spans, rows, columns = _regular_partition(source)
        blocks = [
            source[r0:r1, c0:c1]
            for r0, r1 in row_spans
            for c0, c1 in col_spans
        ]
        shapes = {block.shape for block in blocks}
        if len(shapes) != 1:
            raise ValueError("blueprint blocks must have one shape")
        block_height, block_width = shapes.pop()
        if (len(row_spans), len(col_spans)) != (block_height, block_width):
            raise ValueError("blueprint cells must address the block lattice")

        counts = [int(np.count_nonzero(block != bg)) for block in blocks]
        minimum = min(counts)
        key_indices = [index for index, count in enumerate(counts) if count == minimum]
        if minimum == 0 or len(key_indices) != 1:
            raise ValueError("blueprint block is not uniquely identifiable")
        key_index = key_indices[0]
        other_counts = [count for index, count in enumerate(counts) if index != key_index]
        if not other_counts or len(set(other_counts)) != 1 or minimum >= other_counts[0]:
            raise ValueError("blueprint must be uniquely less complete")

        other_palettes = [
            tuple(sorted(int(value) for value in block.flat if int(value) != bg))
            for index, block in enumerate(blocks)
            if index != key_index
        ]
        if len(set(other_palettes)) != 1:
            raise ValueError("reference blocks do not share one color multiset")

        key = blocks[key_index]
        result = _partition_canvas(source, bg, rows, columns)
        for local_row in range(block_height):
            for local_column in range(block_width):
                color = int(key[local_row, local_column])
                if color == bg:
                    continue
                r0, r1 = row_spans[local_row]
                c0, c1 = col_spans[local_column]
                result[r0:r1, c0:c1] = color
        return result

    return _decorate(transform, "partition_blueprint_expand")


def partition_max_count_fill(minimum_count: int = 2) -> Transform:
    """Fill every block tied for the largest foreground-cell count."""

    if minimum_count < 1:
        raise ValueError("minimum count must be positive")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg, separator, row_spans, col_spans, rows, columns = _regular_partition(source)
        content_colors = set(int(value) for value in np.unique(source)) - {bg, separator}
        if len(content_colors) != 1:
            raise ValueError("max-count fill requires one content color")
        color = content_colors.pop()
        counts = np.array([
            [int(np.count_nonzero(source[r0:r1, c0:c1] == color))
             for c0, c1 in col_spans]
            for r0, r1 in row_spans
        ])
        maximum = int(counts.max())
        if maximum < minimum_count:
            raise ValueError("no sufficiently populated block")
        result = _partition_canvas(source, bg, rows, columns)
        for block_row, (r0, r1) in enumerate(row_spans):
            for block_column, (c0, c1) in enumerate(col_spans):
                if int(counts[block_row, block_column]) == maximum:
                    result[r0:r1, c0:c1] = color
        return result

    return _decorate(transform, "partition_max_count_fill", minimum_count)


def partition_marker_route(marker_color: int) -> Transform:
    """Copy a marked template block to the lattice cell named by the marker."""

    if marker_color not in range(10):
        raise ValueError("marker color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg, _, row_spans, col_spans, rows, columns = _regular_partition(source)
        marker_cells = np.argwhere(source == marker_color)
        if len(marker_cells) != 1:
            raise ValueError("marker routing requires exactly one marker")
        marker_row, marker_column = (int(value) for value in marker_cells[0])

        template: np.ndarray | None = None
        local_marker: tuple[int, int] | None = None
        for r0, r1 in row_spans:
            for c0, c1 in col_spans:
                if r0 <= marker_row < r1 and c0 <= marker_column < c1:
                    template = source[r0:r1, c0:c1].copy()
                    local_marker = (marker_row - r0, marker_column - c0)
        if template is None or local_marker is None:
            raise ValueError("marker is not inside a content block")
        if local_marker[0] >= len(row_spans) or local_marker[1] >= len(col_spans):
            raise ValueError("marker coordinate falls outside the block lattice")

        result = _partition_canvas(source, bg, rows, columns)
        destination_rows = row_spans[local_marker[0]]
        destination_columns = col_spans[local_marker[1]]
        if template.shape != (
            destination_rows[1] - destination_rows[0],
            destination_columns[1] - destination_columns[0],
        ):
            raise ValueError("template and destination block shapes differ")
        result[
            destination_rows[0]:destination_rows[1],
            destination_columns[0]:destination_columns[1],
        ] = template
        return result

    return _decorate(transform, "partition_marker_route", marker_color)


def partition_anchor_relocate(anchor_color: int, anchor_corner: str) -> Transform:
    """Move a panel object so one bbox corner lands on a singleton anchor.

    The source panel also contains a larger component of the anchor color.
    Both anchor components are copied to the opposite, initially empty panel;
    the moved object then overwrites the singleton if its shape occupies the
    selected corner.
    """

    corners = {
        "top_left": lambda r0, c0, r1, c1: (r0, c0),
        "top_right": lambda r0, c0, r1, c1: (r0, c1),
        "bottom_left": lambda r0, c0, r1, c1: (r1, c0),
        "bottom_right": lambda r0, c0, r1, c1: (r1, c1),
    }
    if anchor_color not in range(10):
        raise ValueError("anchor color must be in [0, 9]")
    if anchor_corner not in corners:
        raise ValueError("unsupported anchor corner")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg, _, row_spans, col_spans, rows, columns = _regular_partition(source)
        if (len(row_spans), len(col_spans)) == (2, 1):
            panels = [source[r0:r1, col_spans[0][0]:col_spans[0][1]] for r0, r1 in row_spans]
            orientation = "rows"
        elif (len(row_spans), len(col_spans)) == (1, 2):
            panels = [source[row_spans[0][0]:row_spans[0][1], c0:c1] for c0, c1 in col_spans]
            orientation = "columns"
        else:
            raise ValueError("anchor relocation requires exactly two panels")

        active = [index for index, panel in enumerate(panels) if np.any(panel != bg)]
        empty = [index for index, panel in enumerate(panels) if np.all(panel == bg)]
        if len(active) != 1 or len(empty) != 1:
            raise ValueError("anchor relocation needs one active and one empty panel")
        source_panel = panels[active[0]]
        anchor_objects = Scene(source_panel).components(color=anchor_color)
        singleton = [obj for obj in anchor_objects if obj.size == 1]
        larger = [obj for obj in anchor_objects if obj.size > 1]
        if len(singleton) < 1 or len(larger) != 1:
            raise ValueError("anchor color needs singleton and larger components")
        larger_rows = {row for row, _ in larger[0].cells}
        larger_columns = {column for _, column in larger[0].cells}
        aligned_singletons = [
            obj for obj in singleton
            if obj.cells[0][0] in larger_rows or obj.cells[0][1] in larger_columns
        ]
        if len(aligned_singletons) != 1:
            raise ValueError("singleton relocation anchor is ambiguous")

        object_colors = set(int(value) for value in np.unique(source_panel)) - {bg, anchor_color}
        if len(object_colors) != 1:
            raise ValueError("relocation requires one movable object color")
        object_color = object_colors.pop()
        object_cells = np.argwhere(source_panel == object_color)
        r0, c0 = (int(value) for value in object_cells.min(axis=0))
        r1, c1 = (int(value) for value in object_cells.max(axis=0))
        corner_row, corner_column = corners[anchor_corner](r0, c0, r1, c1)
        target_row, target_column = aligned_singletons[0].cells[0]
        row_shift = target_row - corner_row
        column_shift = target_column - corner_column

        destination = np.full(source_panel.shape, bg, dtype=int)
        destination[source_panel == anchor_color] = anchor_color
        for row_value, column_value in object_cells:
            row = int(row_value) + row_shift
            column = int(column_value) + column_shift
            if not (0 <= row < destination.shape[0] and 0 <= column < destination.shape[1]):
                raise ValueError("relocated object leaves the destination panel")
            destination[row, column] = object_color

        result = _partition_canvas(source, bg, rows, columns)
        target_index = empty[0]
        if orientation == "rows":
            rr0, rr1 = row_spans[target_index]
            cc0, cc1 = col_spans[0]
        else:
            rr0, rr1 = row_spans[0]
            cc0, cc1 = col_spans[target_index]
        result[rr0:rr1, cc0:cc1] = destination
        return result

    return _decorate(transform, "partition_anchor_relocate", anchor_color, anchor_corner)


def frame_and_fill_objects(
    object_color: int,
    frame_color: int,
    hole_color: int,
    background: int | None = None,
) -> Transform:
    """Frame each object bbox and fill holes enclosed by that object."""

    if any(color not in range(10) for color in (object_color, frame_color, hole_color)):
        raise ValueError("ARC colors must be in [0, 9]")
    if len({object_color, frame_color, hole_color}) != 3:
        raise ValueError("object, frame, and hole colors must differ")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        if bg in {object_color, frame_color, hole_color}:
            raise ValueError("background must differ from drawing colors")
        objects = Scene(source).components(color=object_color)
        if not objects:
            raise ValueError("no objects of the requested color")
        result = source.copy()
        height, width = source.shape

        for obj in objects:
            box = obj.bbox
            window = source[box.r0:box.r1 + 1, box.c0:box.c1 + 1]
            open_background = window == bg
            reachable = np.zeros(window.shape, dtype=bool)
            stack: list[tuple[int, int]] = []
            for row in range(window.shape[0]):
                for column in range(window.shape[1]):
                    if row not in {0, window.shape[0] - 1} and column not in {0, window.shape[1] - 1}:
                        continue
                    if open_background[row, column] and not reachable[row, column]:
                        reachable[row, column] = True
                        stack.append((row, column))
            while stack:
                row, column = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = row + dr, column + dc
                    if not (0 <= nr < window.shape[0] and 0 <= nc < window.shape[1]):
                        continue
                    if open_background[nr, nc] and not reachable[nr, nc]:
                        reachable[nr, nc] = True
                        stack.append((nr, nc))
            holes = open_background & ~reachable
            result[box.r0:box.r1 + 1, box.c0:box.c1 + 1][holes] = hole_color

            top, bottom = box.r0 - 1, box.r1 + 1
            left, right = box.c0 - 1, box.c1 + 1
            if top >= 0:
                for column in range(max(0, left), min(width - 1, right) + 1):
                    if int(result[top, column]) == bg:
                        result[top, column] = frame_color
            if bottom < height:
                for column in range(max(0, left), min(width - 1, right) + 1):
                    if int(result[bottom, column]) == bg:
                        result[bottom, column] = frame_color
            if left >= 0:
                for row in range(box.r0, box.r1 + 1):
                    if int(result[row, left]) == bg:
                        result[row, left] = frame_color
            if right < width:
                for row in range(box.r0, box.r1 + 1):
                    if int(result[row, right]) == bg:
                        result[row, right] = frame_color
        return result

    return _decorate(
        transform,
        "frame_and_fill_objects",
        object_color,
        frame_color,
        hole_color,
        background,
    )


def axial_cross_lines(collision_color: int, background: int | None = None) -> Transform:
    """Draw a full row and column through every singleton marker.

    Cells claimed by different marker colors receive ``collision_color``.
    Intersections of lines carrying the same color keep that color.
    """

    if collision_color not in range(10):
        raise ValueError("collision color must be in [0, 9]")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        markers = Scene(source).objects()
        if len(markers) < 2 or any(marker.size != 1 for marker in markers):
            raise ValueError("axial crosses require at least two singleton markers")
        if any(marker.color == collision_color for marker in markers):
            raise ValueError("collision color may not be a marker color")

        claims = [[set() for _ in range(source.shape[1])] for _ in range(source.shape[0])]
        for marker in markers:
            row, column = marker.cells[0]
            for cc in range(source.shape[1]):
                claims[row][cc].add(marker.color)
            for rr in range(source.shape[0]):
                claims[rr][column].add(marker.color)

        result = np.full(source.shape, bg, dtype=int)
        for row in range(source.shape[0]):
            for column in range(source.shape[1]):
                colors = claims[row][column]
                if len(colors) == 1:
                    result[row, column] = next(iter(colors))
                elif len(colors) > 1:
                    result[row, column] = collision_color
        return result

    return _decorate(transform, "axial_cross_lines", collision_color, background)


def connect_closest_horizontal_pairs(background: int | None = None) -> Transform:
    """Connect every horizontal pair tied for the smallest interior gap."""

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        pairs: list[tuple[int, int, int, int]] = []
        for row in range(source.shape[0]):
            columns = np.flatnonzero(source[row] != bg)
            if not len(columns):
                continue
            if len(columns) != 2:
                raise ValueError("each active row must contain exactly two endpoints")
            left, right = (int(value) for value in columns)
            color = int(source[row, left])
            if int(source[row, right]) != color:
                raise ValueError("row endpoints must share one color")
            pairs.append((row, left, right, color))
        if len(pairs) < 2:
            raise ValueError("closest-pair connection requires multiple endpoint rows")

        minimum_gap = min(right - left - 1 for _, left, right, _ in pairs)
        result = source.copy()
        for row, left, right, color in pairs:
            if right - left - 1 == minimum_gap:
                result[row, left:right + 1] = color
        return result

    return _decorate(transform, "connect_closest_horizontal_pairs", background)


def frame_around_markers(
    marker_color: int,
    frame_color: int,
    radius: int = 1,
    background: int | None = None,
) -> Transform:
    """Draw a square ring around each isolated marker, clipping at borders."""

    if marker_color not in range(10) or frame_color not in range(10):
        raise ValueError("marker and frame colors must be in [0, 9]")
    if marker_color == frame_color:
        raise ValueError("marker and frame colors must differ")
    if radius < 1:
        raise ValueError("frame radius must be positive")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        if set(int(value) for value in np.unique(source)) - {bg} != {marker_color}:
            raise ValueError("marker framing requires one foreground color")
        markers = Scene(source).components(color=marker_color)
        if not markers or any(marker.size != 1 for marker in markers):
            raise ValueError("marker framing requires isolated singleton markers")
        result = source.copy()
        for marker in markers:
            row, column = marker.cells[0]
            ring = [
                (rr, cc)
                for rr in range(max(0, row - radius), min(source.shape[0], row + radius + 1))
                for cc in range(max(0, column - radius), min(source.shape[1], column + radius + 1))
                if max(abs(rr - row), abs(cc - column)) == radius
            ]
            if any(int(source[rr, cc]) != bg for rr, cc in ring):
                raise ValueError("marker frames overlap foreground content")
            for rr, cc in ring:
                if int(result[rr, cc]) not in {bg, frame_color}:
                    raise ValueError("marker frames overlap ambiguously")
                result[rr, cc] = frame_color
        return result

    return _decorate(
        transform,
        "frame_around_markers",
        marker_color,
        frame_color,
        radius,
        background,
    )


def periodic_boundary_stripes(background: int | None = None) -> Transform:
    """Extend two boundary markers as alternating, equally spaced stripes."""

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        cells = np.argwhere(source != bg)
        if len(cells) != 2:
            raise ValueError("periodic stripes require exactly two markers")
        markers = [
            (int(row), int(column), int(source[row, column]))
            for row, column in cells
        ]
        if markers[0][2] == markers[1][2]:
            raise ValueError("periodic stripe markers must have different colors")

        column_boundary = all(
            column in {0, source.shape[1] - 1}
            for _, column, _ in markers
        )
        row_boundary = all(
            row in {0, source.shape[0] - 1}
            for row, _, _ in markers
        )
        if column_boundary == row_boundary:
            raise ValueError("stripe orientation is ambiguous")

        result = np.full(source.shape, bg, dtype=int)
        if column_boundary:
            ordered = sorted((row, color) for row, _, color in markers)
            limit = source.shape[0]
            horizontal = True
        else:
            ordered = sorted((column, color) for _, column, color in markers)
            limit = source.shape[1]
            horizontal = False
        step = ordered[1][0] - ordered[0][0]
        if step <= 0:
            raise ValueError("stripe markers must differ along the stripe axis")
        for stripe_index, coordinate in enumerate(range(ordered[0][0], limit, step)):
            color = ordered[stripe_index % 2][1]
            if horizontal:
                result[coordinate, :] = color
            else:
                result[:, coordinate] = color
        return result

    return _decorate(transform, "periodic_boundary_stripes", background)


def oriented_marker_lines(
    vertical_colors: set[int] | tuple[int, ...],
    horizontal_colors: set[int] | tuple[int, ...],
    horizontal_overwrites: bool = True,
    background: int | None = None,
) -> Transform:
    """Extend marker colors in configured directions with explicit precedence."""

    vertical = tuple(sorted(int(color) for color in vertical_colors))
    horizontal = tuple(sorted(int(color) for color in horizontal_colors))
    if any(color not in range(10) for color in (*vertical, *horizontal)):
        raise ValueError("line colors must be in [0, 9]")
    if set(vertical) & set(horizontal):
        raise ValueError("vertical and horizontal colors must be disjoint")
    if not vertical and not horizontal:
        raise ValueError("at least one line color is required")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        configured = set(vertical) | set(horizontal)
        present = set(int(value) for value in np.unique(source)) - {bg}
        if not present or not present <= configured:
            raise ValueError("grid contains unconfigured marker colors")
        if any(obj.size != 1 for obj in Scene(source).objects()):
            raise ValueError("oriented lines require singleton marker components")

        vertical_claims: dict[int, int] = {}
        horizontal_claims: dict[int, int] = {}
        for row_value, column_value in np.argwhere(source != bg):
            row, column = int(row_value), int(column_value)
            color = int(source[row, column])
            claims = vertical_claims if color in vertical else horizontal_claims
            coordinate = column if color in vertical else row
            previous = claims.setdefault(coordinate, color)
            if previous != color:
                raise ValueError("different colors claim the same line")

        result = np.full(source.shape, bg, dtype=int)

        def draw_vertical() -> None:
            for column, color in vertical_claims.items():
                result[:, column] = color

        def draw_horizontal() -> None:
            for row, color in horizontal_claims.items():
                result[row, :] = color

        if horizontal_overwrites:
            draw_vertical()
            draw_horizontal()
        else:
            draw_horizontal()
            draw_vertical()
        return result

    return _decorate(
        transform,
        "oriented_marker_lines",
        vertical,
        horizontal,
        horizontal_overwrites,
        background,
    )


def marker_column_bands(
    terminal_color: int,
    background: int | None = None,
) -> Transform:
    """Fill marker columns in vertical bands ending at each marker.

    The terminal markers sit on the bottom edge and identify the active
    columns.  In each active column, a marker colors the band from the cell
    after the previous marker through itself.  Unrelated cells are preserved.
    """

    if terminal_color not in range(10):
        raise ValueError("terminal color must be in [0, 9]")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        columns = [int(value) for value in np.flatnonzero(source[-1] == terminal_color)]
        if not columns:
            raise ValueError("marker bands require terminal markers on the bottom edge")

        result = source.copy()
        for column in columns:
            marker_rows = [int(value) for value in np.flatnonzero(source[:, column] != bg)]
            if not marker_rows or marker_rows[-1] != source.shape[0] - 1:
                raise ValueError("each active column must end in a terminal marker")
            start = 0
            for row in marker_rows:
                result[start:row + 1, column] = int(source[row, column])
                start = row + 1
            if start != source.shape[0]:
                raise ValueError("marker bands do not cover the full column")
        if np.array_equal(result, source):
            raise ValueError("marker-band transform would be a no-op")
        return result

    return _decorate(transform, "marker_column_bands", terminal_color, background)


def periodic_pattern_repair(hole_color: int) -> Transform:
    """Repair holes from the unique smallest two-dimensional period.

    Known cells vote by residue class.  Ambiguous or degenerate patterns fail
    closed; only cells carrying ``hole_color`` may be changed.
    """

    if hole_color not in range(10):
        raise ValueError("hole color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        holes = np.argwhere(source == hole_color)
        if not len(holes):
            raise ValueError("periodic repair requires at least one hole")
        height, width = source.shape
        candidates: list[tuple[tuple[int, int], np.ndarray]] = []

        for row_period in range(2, height):
            for column_period in range(2, width):
                residues: dict[tuple[int, int], int] = {}
                compatible = True
                for row_value, column_value in np.argwhere(source != hole_color):
                    row, column = int(row_value), int(column_value)
                    key = (row % row_period, column % column_period)
                    value = int(source[row, column])
                    previous = residues.setdefault(key, value)
                    if previous != value:
                        compatible = False
                        break
                if not compatible or len(set(residues.values())) < 2:
                    continue
                if any(
                    (int(row) % row_period, int(column) % column_period) not in residues
                    for row, column in holes
                ):
                    continue

                repaired = source.copy()
                for row_value, column_value in holes:
                    row, column = int(row_value), int(column_value)
                    repaired[row, column] = residues[
                        (row % row_period, column % column_period)
                    ]
                if np.array_equal(repaired, source):
                    continue
                score = (row_period * column_period, row_period + column_period)
                candidates.append((score, repaired))

        if not candidates:
            raise ValueError("no non-degenerate periodic repair exists")
        best_score = min(score for score, _ in candidates)
        best = [candidate for score, candidate in candidates if score == best_score]
        unique: list[np.ndarray] = []
        for candidate in best:
            if not any(np.array_equal(candidate, previous) for previous in unique):
                unique.append(candidate)
        if len(unique) != 1:
            raise ValueError("periodic repair is ambiguous")
        return unique[0]

    return _decorate(transform, "periodic_pattern_repair", hole_color)


def _separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in separators:
        if start < separator:
            spans.append((start, separator))
        start = separator + 1
    if start < length:
        spans.append((start, length))
    return spans


def reflection_block_repair(background: int | None = None) -> Transform:
    """Complete damaged 2x2 block quartets by horizontal/vertical reflection."""

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        separator_rows = [
            row for row in range(source.shape[0])
            if np.all(source[row] == source[row, 0]) and int(source[row, 0]) != bg
        ]
        separator_columns = [
            column for column in range(source.shape[1])
            if np.all(source[:, column] == source[0, column])
            and int(source[0, column]) != bg
        ]
        if not separator_rows or not separator_columns:
            raise ValueError("block repair requires row and column separators")
        row_spans = _separator_spans(source.shape[0], separator_rows)
        column_spans = _separator_spans(source.shape[1], separator_columns)
        repairs: list[tuple[int, int, np.ndarray]] = []

        for row_index in range(len(row_spans) - 1):
            for column_index in range(len(column_spans) - 1):
                coordinates = [
                    (row_index, column_index),
                    (row_index, column_index + 1),
                    (row_index + 1, column_index),
                    (row_index + 1, column_index + 1),
                ]
                shapes = {
                    (
                        row_spans[row][1] - row_spans[row][0],
                        column_spans[column][1] - column_spans[column][0],
                    )
                    for row, column in coordinates
                }
                if len(shapes) != 1:
                    continue
                tiles = {
                    (row, column): source[
                        row_spans[row][0]:row_spans[row][1],
                        column_spans[column][0]:column_spans[column][1],
                    ]
                    for row, column in coordinates
                }
                occupied = [key for key, tile in tiles.items() if np.any(tile != bg)]
                empty = [key for key, tile in tiles.items() if np.all(tile == bg)]
                if len(occupied) != 3 or len(empty) != 1:
                    continue

                canonical: list[np.ndarray] = []
                for row, column in occupied:
                    tile = tiles[(row, column)]
                    if row == row_index + 1:
                        tile = np.flipud(tile)
                    if column == column_index + 1:
                        tile = np.fliplr(tile)
                    canonical.append(tile)
                if not all(np.array_equal(canonical[0], tile) for tile in canonical[1:]):
                    continue

                missing_row, missing_column = empty[0]
                prediction = canonical[0]
                if missing_row == row_index + 1:
                    prediction = np.flipud(prediction)
                if missing_column == column_index + 1:
                    prediction = np.fliplr(prediction)
                repairs.append((missing_row, missing_column, prediction.copy()))

        if not repairs:
            raise ValueError("no unique reflection-block repair found")
        result = source.copy()
        written: dict[tuple[int, int], np.ndarray] = {}
        for row, column, prediction in repairs:
            key = (row, column)
            previous = written.get(key)
            if previous is not None and not np.array_equal(previous, prediction):
                raise ValueError("overlapping block repairs disagree")
            written[key] = prediction
        for (row, column), prediction in written.items():
            r0, r1 = row_spans[row]
            c0, c1 = column_spans[column]
            if np.any(result[r0:r1, c0:c1] != bg):
                raise ValueError("block repair would overwrite existing content")
            result[r0:r1, c0:c1] = prediction
        return result

    return _decorate(transform, "reflection_block_repair", background)


def unify_shape_colors(anomaly_color: int, background: int | None = None) -> Transform:
    """Recolor anomalous objects from same-shape exemplars."""

    if anomaly_color not in range(10):
        raise ValueError("anomaly color must be in [0, 9]")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        scene = Scene(source)
        bg = scene.background_color() if background is None else background
        if anomaly_color == bg:
            raise ValueError("anomaly color may not be the background")
        groups: dict[tuple[tuple[int, int], ...], list] = {}
        for obj in scene.objects():
            groups.setdefault(obj.normalized_cells(), []).append(obj)

        result = source.copy()
        changed = False
        for objects in groups.values():
            colors = {obj.color for obj in objects}
            targets = colors - {anomaly_color}
            if len(objects) < 2 or anomaly_color not in colors or len(targets) != 1:
                continue
            target = next(iter(targets))
            for obj in objects:
                if obj.color != anomaly_color:
                    continue
                for row, column in obj.cells:
                    result[row, column] = target
                changed = True
        if not changed:
            raise ValueError("no same-shape anomaly can be unified")
        return result

    return _decorate(transform, "unify_shape_colors", anomaly_color, background)


def periodic_perimeter_segments(background: int | None = None) -> Transform:
    """Repeat one boundary run as equal on/off segments around the perimeter."""

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        height, width = source.shape
        if height < 2 or width < 2:
            raise ValueError("perimeter completion requires a two-dimensional grid")
        perimeter = (
            [(0, column) for column in range(width)]
            + [(row, width - 1) for row in range(1, height)]
            + [(height - 1, column) for column in range(width - 2, -1, -1)]
            + [(row, 0) for row in range(height - 2, 0, -1)]
        )
        perimeter_set = set(perimeter)
        if any(
            int(source[row, column]) != bg and (row, column) not in perimeter_set
            for row in range(height)
            for column in range(width)
        ):
            raise ValueError("perimeter completion cannot move interior content")
        active = [
            index for index, (row, column) in enumerate(perimeter)
            if int(source[row, column]) != bg
        ]
        if not active:
            raise ValueError("perimeter completion requires one active run")
        colors = {int(source[perimeter[index]]) for index in active}
        if len(colors) != 1:
            raise ValueError("perimeter run must have one color")
        active_set = set(active)
        starts = [
            index for index in active
            if (index - 1) % len(perimeter) not in active_set
        ]
        if len(starts) != 1:
            raise ValueError("perimeter input must contain one circular run")
        start = starts[0]
        run_length = len(active)
        expected = {(start + offset) % len(perimeter) for offset in range(run_length)}
        if expected != active_set:
            raise ValueError("perimeter foreground is not contiguous")
        if len(perimeter) % (2 * run_length):
            raise ValueError("perimeter cannot be tiled by equal on/off runs")

        result = source.copy()
        color = next(iter(colors))
        for segment_start in range(start, start + len(perimeter), 2 * run_length):
            for offset in range(run_length):
                result[perimeter[(segment_start + offset) % len(perimeter)]] = color
        if np.array_equal(result, source):
            raise ValueError("perimeter completion would be a no-op")
        return result

    return _decorate(transform, "periodic_perimeter_segments", background)


def complete_occluded_rectangles(background: int | None = None) -> Transform:
    """Restore solid rectangles or diagonal rays occluded by another color.

    A color is eligible only when its aggregate bounding box contains no
    background cells.  This distinguishes a genuinely occluded rectangle from
    an arbitrary sparse object.  Conflicting candidate rectangles fail closed.
    """

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        candidates: list[tuple[int, int, int, int, int]] = []
        for color_value in np.unique(source):
            color = int(color_value)
            if color == bg:
                continue
            cells = np.argwhere(source == color)
            r0, c0 = (int(value) for value in cells.min(axis=0))
            r1, c1 = (int(value) for value in cells.max(axis=0))
            window = source[r0:r1 + 1, c0:c1 + 1]
            if np.all(window == color) or np.any(window == bg):
                continue
            candidates.append((color, r0, c0, r1, c1))

        for index, first in enumerate(candidates):
            _, ar0, ac0, ar1, ac1 = first
            for second in candidates[index + 1:]:
                _, br0, bc0, br1, bc1 = second
                if max(ar0, br0) <= min(ar1, br1) and max(ac0, bc0) <= min(ac1, bc1):
                    raise ValueError("occluded rectangle candidates overlap ambiguously")

        writes: dict[tuple[int, int], int] = {}
        for color, r0, c0, r1, c1 in candidates:
            for row in range(r0, r1 + 1):
                for column in range(c0, c1 + 1):
                    if int(source[row, column]) != color:
                        writes[(row, column)] = color

        # A long diagonal can be read exactly like a one-cell-wide rectangle:
        # if every gap is occupied by another foreground color, restore it.
        for color_value in np.unique(source):
            color = int(color_value)
            if color == bg:
                continue
            cells = [tuple(int(value) for value in cell) for cell in np.argwhere(source == color)]
            if len(cells) < 3:
                continue
            expected_lines: list[list[tuple[int, int]]] = []
            if len({row - column for row, column in cells}) == 1:
                diagonal = cells[0][0] - cells[0][1]
                rows = range(min(row for row, _ in cells), max(row for row, _ in cells) + 1)
                expected_lines.append([(row, row - diagonal) for row in rows])
            if len({row + column for row, column in cells}) == 1:
                diagonal = cells[0][0] + cells[0][1]
                rows = range(min(row for row, _ in cells), max(row for row, _ in cells) + 1)
                expected_lines.append([(row, diagonal - row) for row in rows])
            for expected in expected_lines:
                missing = [cell for cell in expected if int(source[cell]) != color]
                if not missing or any(int(source[cell]) == bg for cell in missing):
                    continue
                for cell in missing:
                    previous = writes.setdefault(cell, color)
                    if previous != color:
                        raise ValueError("occluded shape repairs conflict")

        if not writes:
            raise ValueError("no occluded rectangle or diagonal found")
        result = source.copy()
        for cell, color in writes.items():
            result[cell] = color
        return result

    return _decorate(transform, "complete_occluded_rectangles", background)


def opposite_edge_anomaly_pairs(foreground: int, background: int | None = None) -> Transform:
    """Complete the opposite dents/bulges of a regular two-bar L shape."""

    if foreground not in range(10):
        raise ValueError("foreground color must be in [0, 9]")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        bg = Scene(source).background_color() if background is None else background
        if foreground == bg or set(int(value) for value in np.unique(source)) != {bg, foreground}:
            raise ValueError("opposite-edge repair requires one foreground color")
        mask = source == foreground
        height, width = mask.shape
        row_counts = np.count_nonzero(mask, axis=1)
        column_counts = np.count_nonzero(mask, axis=0)
        if not row_counts.max() or not column_counts.max():
            raise ValueError("foreground is empty")

        def longest_group(indices: list[int]) -> list[int]:
            groups: list[list[int]] = []
            for value in indices:
                if groups and value == groups[-1][-1] + 1:
                    groups[-1].append(value)
                else:
                    groups.append([value])
            if not groups:
                raise ValueError("regular bar not found")
            maximum = max(len(group) for group in groups)
            best = [group for group in groups if len(group) == maximum]
            if len(best) != 1:
                raise ValueError("regular bar axis is ambiguous")
            return best[0]

        horizontal_rows = longest_group([
            row for row, count in enumerate(row_counts)
            if count > float(row_counts.max()) * 0.65
        ])
        vertical_columns = longest_group([
            column for column, count in enumerate(column_counts)
            if count > float(column_counts.max()) * 0.65
        ])

        def middle(values: list[int]) -> int:
            ordered = sorted(values)
            size = len(ordered)
            if size % 2:
                return ordered[size // 2]
            return (ordered[size // 2 - 1] + ordered[size // 2]) // 2

        horizontal_left = middle([int(np.flatnonzero(mask[row])[0]) for row in horizontal_rows])
        horizontal_right = middle([int(np.flatnonzero(mask[row])[-1]) for row in horizontal_rows])
        vertical_top = middle([int(np.flatnonzero(mask[:, column])[0]) for column in vertical_columns])
        vertical_bottom = middle([int(np.flatnonzero(mask[:, column])[-1]) for column in vertical_columns])

        base = np.zeros_like(mask)
        base[min(horizontal_rows):max(horizontal_rows) + 1, horizontal_left:horizontal_right + 1] = True
        base[vertical_top:vertical_bottom + 1, min(vertical_columns):max(vertical_columns) + 1] = True
        differences = np.argwhere(mask != base)
        if not len(differences):
            raise ValueError("regular L shape has no dents or bulges")

        result = source.copy()
        written: dict[tuple[int, int], int] = {}
        for row_value, column_value in differences:
            row, column = int(row_value), int(column_value)
            extra = bool(mask[row, column])
            directions: list[tuple[int, int]] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if extra:
                    neighbor = (row - dr, column - dc)
                    if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width and base[neighbor]:
                        directions.append((dr, dc))
                else:
                    neighbor = (row + dr, column + dc)
                    if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width and base[neighbor]):
                        directions.append((dr, dc))
            if len(directions) != 1:
                raise ValueError("dent or bulge does not have one outward direction")
            dr, dc = directions[0]
            axis = 0 if dr else 1
            if axis == 0:
                line_column = column
                if extra:
                    line_column -= dc
                indices = [value for value in range(height) if base[value, line_column]]
            else:
                line_row = row
                if extra:
                    line_row -= dr
                indices = [value for value in range(width) if base[line_row, value]]
            if not indices:
                raise ValueError("dent or bulge has no opposite base edge")
            coordinate = row if axis == 0 else column
            sign = dr if axis == 0 else dc
            target_coordinate = min(indices) + max(indices) - coordinate + sign * (1 if extra else -1)
            target = [row, column]
            target[axis] = target_coordinate
            target_cell = (target[0], target[1])
            if not (0 <= target[0] < height and 0 <= target[1] < width):
                raise ValueError("opposite repair would leave the grid")
            value = bg if extra else foreground
            previous = written.setdefault(target_cell, value)
            if previous != value:
                raise ValueError("opposite repairs conflict")

        for cell, value in written.items():
            result[cell] = value
        if np.array_equal(result, source):
            raise ValueError("opposite-edge repair would be a no-op")
        return result

    return _decorate(transform, "opposite_edge_anomaly_pairs", foreground, background)


def directed_marker_ray_cleanup() -> Transform:
    """Erase the first border intrusion opposite a marker-attached line."""

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        scene = Scene(source)
        bg = scene.background_color()
        positions = {int(color): np.argwhere(source == color) for color in np.unique(source)}
        candidates: list[tuple[int, int, int]] = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ]
        for marker_color, marker_cells in positions.items():
            if marker_color == bg or len(marker_cells) != 1:
                continue
            marker = tuple(int(value) for value in marker_cells[0])
            for line_color, line_cells in positions.items():
                if line_color in (bg, marker_color) or len(line_cells) < 2:
                    continue
                cells = {tuple(int(value) for value in cell) for cell in line_cells}
                for dr, dc in directions:
                    expected = {(marker[0] + step * dr, marker[1] + step * dc) for step in range(1, len(cells) + 1)}
                    if expected != cells:
                        continue
                    row, column = marker[0] - dr, marker[1] - dc
                    if not (0 <= row < source.shape[0] and 0 <= column < source.shape[1]):
                        continue
                    fill = int(source[row, column])
                    if fill in (marker_color, line_color):
                        continue
                    while 0 <= row < source.shape[0] and 0 <= column < source.shape[1] and int(source[row, column]) == fill:
                        row -= dr
                        column -= dc
                    if 0 <= row < source.shape[0] and 0 <= column < source.shape[1]:
                        candidates.append((row, column, fill))
        unique = sorted(set(candidates))
        if len(unique) != 1:
            raise ValueError("marker ray is absent or ambiguous")
        row, column, fill = unique[0]
        result = source.copy()
        result[row, column] = fill
        return result

    return _decorate(transform, "directed_marker_ray_cleanup")


def legend_sample_erase(background: int | None = None) -> Transform:
    """Use the singleton inside an L-shaped legend to erase that color outside."""

    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        scene = Scene(source)
        bg = scene.background_color() if background is None else background
        matches: list[tuple[int, tuple[int, int, int, int]]] = []
        for obj in scene.objects():
            if obj.color == bg or obj.bbox.height != obj.bbox.width or obj.bbox.height < 3:
                continue
            n = obj.bbox.height
            cells = set(obj.cells)
            r0, c0, r1, c1 = obj.bbox.r0, obj.bbox.c0, obj.bbox.r1, obj.bbox.c1
            patterns = [
                ({(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}, (r0, c0, r1 - 1, c1 - 1)),
                ({(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}, (r0, c0 + 1, r1 - 1, c1)),
                ({(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}, (r0 + 1, c0, r1, c1 - 1)),
                ({(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}, (r0 + 1, c0 + 1, r1, c1)),
            ]
            for expected, interior in patterns:
                if cells != expected:
                    continue
                ir0, ic0, ir1, ic1 = interior
                values = source[ir0:ir1 + 1, ic0:ic1 + 1]
                samples = [(int(value), int(np.count_nonzero(values == value))) for value in np.unique(values) if int(value) not in (bg, obj.color)]
                if len(samples) == 1 and samples[0][1] == 1:
                    matches.append((samples[0][0], (r0, c0, r1, c1)))
        if len(matches) != 1:
            raise ValueError("L-shaped legend is absent or ambiguous")
        sample, box = matches[0]
        r0, c0, r1, c1 = box
        result = source.copy()
        outside = source == sample
        outside[r0:r1 + 1, c0:c1 + 1] = False
        if not np.any(outside):
            raise ValueError("legend sample has no external instances")
        result[outside] = bg
        return result

    return _decorate(transform, "legend_sample_erase", background)


def duplicate_template_refine(
    result_color: int,
    mask_color: int = 0,
    reference_color: int = 5,
    background: int | None = None,
) -> Transform:
    """Trim the mask whose rectangular body faces a reference object sideways."""

    for name, color in (
        ("result", result_color),
        ("mask", mask_color),
        ("reference", reference_color),
    ):
        if color not in range(10):
            raise ValueError(f"{name} color must be in [0, 9]")
    if background is not None and background not in range(10):
        raise ValueError("background color must be in [0, 9]")
    if len({result_color, mask_color, reference_color}) != 3:
        raise ValueError("result, mask, and reference colors must differ")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        scene = Scene(source)
        bg = scene.background_color() if background is None else background
        if bg in (result_color, mask_color, reference_color):
            raise ValueError("foreground roles may not use the background color")
        reference_cells = np.argwhere(source == reference_color)
        if not len(reference_cells):
            raise ValueError("reference object is absent")
        reference_top, reference_left = (int(value) for value in reference_cells.min(axis=0))
        reference_bottom, reference_right = (int(value) for value in reference_cells.max(axis=0))

        candidates: list[tuple[int, object, tuple[int, int, int, int]]] = []
        for obj in scene.components(color=mask_color):
            if obj.size < 2:
                continue
            object_cells = set(obj.cells)
            rectangles: list[tuple[int, int, int, int, int]] = []
            for r0 in range(obj.bbox.r0, obj.bbox.r1 + 1):
                for r1 in range(r0, obj.bbox.r1 + 1):
                    for c0 in range(obj.bbox.c0, obj.bbox.c1 + 1):
                        for c1 in range(c0, obj.bbox.c1 + 1):
                            cells = {
                                (row, column)
                                for row in range(r0, r1 + 1)
                                for column in range(c0, c1 + 1)
                            }
                            if cells <= object_cells:
                                rectangles.append((len(cells), r0, c0, r1, c1))
            if not rectangles:
                continue
            area, r0, c0, r1, c1 = max(rectangles)
            row_overlap = max(r0, reference_top) <= min(r1, reference_bottom)
            sideways = c1 < reference_left or c0 > reference_right
            if area <= 1 or not row_overlap or not sideways:
                continue
            distance = reference_left - c1 if c1 < reference_left else c0 - reference_right
            candidates.append((distance, obj, (r0, c0, r1, c1)))

        if not candidates:
            raise ValueError("no mask body faces the reference sideways")
        minimum = min(distance for distance, _, _ in candidates)
        best = [candidate for candidate in candidates if candidate[0] == minimum]
        if len(best) != 1:
            raise ValueError("side-facing mask selection is ambiguous")
        _, target, (r0, c0, r1, c1) = best[0]

        result = source.copy()
        for cell in target.cells:
            result[cell] = bg
        result[r0:r1 + 1, c0:c1 + 1] = mask_color
        facing_column = c1 if c1 < reference_left else c0
        result[r0:r1 + 1, facing_column] = result_color
        return result

    return _decorate(
        transform,
        "duplicate_template_refine",
        result_color,
        mask_color,
        reference_color,
        background,
    )


def extend_diagonal_lattice(
    marker_color: int,
    fill_color: int,
    background: int = 0,
) -> Transform:
    """Extend an equally spaced diagonal marker lattice toward its open end."""

    if any(color not in range(10) for color in (marker_color, fill_color, background)):
        raise ValueError("ARC colors must be in [0, 9]")
    if len({marker_color, fill_color, background}) != 3:
        raise ValueError("marker, fill, and background colors must differ")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape[0] < 3 or source.shape[1] < 3:
            return source.copy()
        if set(int(value) for value in np.unique(source)) - {background, marker_color}:
            return source.copy()
        points = np.argwhere(source == marker_color)
        if len(points) < 3:
            return source.copy()

        candidates: list[np.ndarray] = []
        for column_sign in (1, -1):
            diagonal = points[:, 0] - column_sign * points[:, 1]
            if len(set(int(value) for value in diagonal)) != 1:
                continue
            ordered = points[np.argsort(points[:, 0])]
            differences = np.diff(ordered, axis=0)
            if not len(differences) or not np.all(differences == differences[0]):
                continue
            row_step, column_step = (int(value) for value in differences[0])
            if row_step <= 0 or abs(column_step) != row_step:
                continue

            result = source.copy()
            position = ordered[-1].copy() + differences[0]
            changed = False
            while (
                0 <= position[0] < source.shape[0]
                and 0 <= position[1] < source.shape[1]
            ):
                cell = tuple(int(value) for value in position)
                if int(source[cell]) != background:
                    changed = False
                    break
                result[cell] = fill_color
                changed = True
                position += differences[0]
            if changed:
                candidates.append(result)

        if len(candidates) != 1:
            return source.copy()
        return candidates[0]

    return _decorate(
        transform,
        "extend_diagonal_lattice",
        marker_color,
        fill_color,
        background,
    )


def complete_solid_template(
    fill_color: int,
    background: int = 0,
) -> Transform:
    """Complete partial copies of a unique solid 3x3 template in the grid."""

    if any(color not in range(10) for color in (fill_color, background)):
        raise ValueError("ARC colors must be in [0, 9]")
    if fill_color == background:
        raise ValueError("fill and background colors must differ")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        height, width = source.shape
        if height < 3 or width < 3:
            return source.copy()

        templates: list[np.ndarray] = []
        for row in range(height - 2):
            for column in range(width - 2):
                window = source[row:row + 3, column:column + 3]
                if np.all(window != background) and np.any(window == fill_color):
                    if not any(np.array_equal(window, item) for item in templates):
                        templates.append(window.copy())
        if len(templates) != 1:
            return source.copy()
        template = templates[0]
        fill_mask = template == fill_color
        if np.all(fill_mask):
            return source.copy()

        proposals: dict[tuple[int, int], int] = {}
        for row in range(height - 2):
            for column in range(width - 2):
                window = source[row:row + 3, column:column + 3]
                if not np.all(window[~fill_mask] == template[~fill_mask]):
                    continue
                if not np.all((window[fill_mask] == fill_color) | (window[fill_mask] == background)):
                    continue
                missing = np.argwhere(fill_mask & (window == background))
                for local_row, local_column in missing:
                    proposals[(row + int(local_row), column + int(local_column))] = fill_color

        if not proposals:
            return source.copy()
        result = source.copy()
        for cell, color in proposals.items():
            result[cell] = color
        return result

    return _decorate(transform, "complete_solid_template", fill_color, background)


def complete_d4_orbits(background: int = 0) -> Transform:
    """Fill missing cells in same-color D4 orbits inside a square pattern."""

    if background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        points = np.argwhere(source != background)
        if not len(points):
            return source.copy()
        top, left = (int(value) for value in points.min(axis=0))
        bottom, right = (int(value) for value in points.max(axis=0))
        if bottom - top != right - left or (bottom - top) % 2:
            return source.copy()
        center_row, center_column = (top + bottom) // 2, (left + right) // 2
        if int(source[center_row, center_column]) == background:
            return source.copy()

        result = source.copy()
        seen: set[tuple[tuple[int, int], ...]] = set()
        changed = False
        for row, column in (tuple(int(value) for value in point) for point in points):
            dr, dc = row - center_row, column - center_column
            orbit = {
                (center_row + dr, center_column + dc),
                (center_row + dr, center_column - dc),
                (center_row - dr, center_column + dc),
                (center_row - dr, center_column - dc),
                (center_row + dc, center_column + dr),
                (center_row + dc, center_column - dr),
                (center_row - dc, center_column + dr),
                (center_row - dc, center_column - dr),
            }
            key = tuple(sorted(orbit))
            if key in seen:
                continue
            seen.add(key)
            colors = {
                int(source[cell])
                for cell in orbit
                if int(source[cell]) != background
            }
            if len(colors) != 1:
                return source.copy()
            color = next(iter(colors))
            for cell in orbit:
                if int(result[cell]) == background:
                    result[cell] = color
                    changed = True
        return result if changed else source.copy()

    return _decorate(transform, "complete_d4_orbits", background)


def mode_below_separator(background: int = 0) -> Transform:
    """Place the unique modal upper color below a full separator row."""

    if background not in range(10):
        raise ValueError("background color must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        height, width = source.shape
        if height < 3 or width < 3 or width % 2 == 0:
            return source.copy()
        separators: list[int] = []
        for row in range(1, height - 1):
            values = set(int(value) for value in source[row])
            if len(values) == 1 and next(iter(values)) != background:
                separators.append(row)
        if len(separators) != 1:
            return source.copy()
        separator_row = separators[0]
        separator_color = int(source[separator_row, 0])
        if np.any(source[separator_row + 1:] != background):
            return source.copy()
        if np.any(source[:separator_row] == background):
            return source.copy()
        values = [
            int(value)
            for value in source[:separator_row].flat
            if int(value) not in (background, separator_color)
        ]
        counts = Counter(values).most_common()
        if not counts or (len(counts) > 1 and counts[0][1] == counts[1][1]):
            return source.copy()
        result = source.copy()
        result[height - 1, width // 2] = counts[0][0]
        return result

    return _decorate(transform, "mode_below_separator", background)


def complete_local_symmetric_templates(background: int = 0) -> Transform:
    """Propagate isolated centered 3x3 symmetric motifs to partial copies."""

    if background not in range(10):
        raise ValueError("background color must be in [0, 9]")
    masks = (
        frozenset({(0, -1), (0, 1)}),
        frozenset({(-1, 0), (1, 0)}),
        frozenset({(-1, 0), (1, 0), (0, -1), (0, 1)}),
        frozenset({(-1, -1), (-1, 1), (1, -1), (1, 1)}),
    )

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        height, width = source.shape
        if height < 3 or width < 3:
            return source.copy()

        prototypes: list[tuple[int, int, frozenset[tuple[int, int]]]] = []
        for row in range(1, height - 1):
            for column in range(1, width - 1):
                center = int(source[row, column])
                if center == background:
                    continue
                for offsets in masks:
                    support_values = [
                        int(source[row + dr, column + dc]) for dr, dc in offsets
                    ]
                    if len(set(support_values)) != 1:
                        continue
                    support = support_values[0]
                    if support in (background, center):
                        continue
                    used = {(0, 0), *offsets}
                    isolated = all(
                        int(source[row + dr, column + dc]) == background
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if (dr, dc) not in used
                    )
                    prototype = (center, support, offsets)
                    if isolated and prototype not in prototypes:
                        prototypes.append(prototype)
        if not prototypes:
            return source.copy()

        proposals: dict[tuple[int, int], set[int]] = {}
        for center, support, offsets in prototypes:
            for row in range(1, height - 1):
                for column in range(1, width - 1):
                    center_value = int(source[row, column])
                    if center_value not in (background, center):
                        continue
                    support_values = {
                        offset: int(source[row + offset[0], column + offset[1]])
                        for offset in offsets
                    }
                    if any(value not in (background, support) for value in support_values.values()):
                        continue
                    used = {(0, 0), *offsets}
                    if any(
                        int(source[row + dr, column + dc]) != background
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if (dr, dc) not in used
                    ):
                        continue
                    center_present = center_value == center
                    complete_support = all(value == support for value in support_values.values())
                    if not center_present and not complete_support:
                        continue
                    if center_value == background:
                        proposals.setdefault((row, column), set()).add(center)
                    for (dr, dc), value in support_values.items():
                        if value == background:
                            proposals.setdefault((row + dr, column + dc), set()).add(support)

        if not proposals or any(len(colors) != 1 for colors in proposals.values()):
            return source.copy()
        result = source.copy()
        for cell, colors in proposals.items():
            result[cell] = next(iter(colors))
        return result

    return _decorate(transform, "complete_local_symmetric_templates", background)


def extract_full_span_color() -> Transform:
    """Return the unique color forming an uninterrupted full row or column."""

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape[0] < 3 or source.shape[1] < 3 or len(np.unique(source)) < 2:
            raise ValueError("a non-degenerate multi-color grid is required")
        background = Scene(source).background_color()
        colors: set[int] = set()
        for line in (*source, *source.T):
            values = set(int(value) for value in line)
            if len(values) == 1:
                color = next(iter(values))
                if color != background:
                    colors.add(color)
        if len(colors) != 1:
            raise ValueError("full-span color is absent or ambiguous")
        return np.asarray([[next(iter(colors))]], dtype=int)

    return _decorate(transform, "extract_full_span_color")


def quadrant_object_map() -> Transform:
    """Map isolated 2x2 objects inside a field to their 2x2 quadrants."""

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        height, width = source.shape
        if height < 4 or width < 4:
            raise ValueError("quadrant extraction requires a larger grid")
        border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
        border_counts = Counter(int(value) for value in border)
        outer, outer_count = border_counts.most_common(1)[0]
        if len(border_counts) > 1 and outer_count == border_counts.most_common(2)[1][1]:
            raise ValueError("outer background is ambiguous")

        field_cells = np.argwhere(source != outer)
        if not len(field_cells):
            raise ValueError("field is absent")
        top, left = (int(value) for value in field_cells.min(axis=0))
        bottom, right = (int(value) for value in field_cells.max(axis=0))
        inside = source[top:bottom + 1, left:right + 1]
        if np.any(inside == outer):
            raise ValueError("field rectangle contains outer-background holes")
        field_counts = Counter(int(value) for value in inside.flat)
        field, field_count = field_counts.most_common(1)[0]
        if len(field_counts) > 1 and field_count == field_counts.most_common(2)[1][1]:
            raise ValueError("field background is ambiguous")

        scene = Scene(source)
        result = np.full((2, 2), outer, dtype=int)
        object_count = 0
        row_midpoint = (top + bottom) / 2
        column_midpoint = (left + right) / 2
        for color in sorted(field_counts):
            if color == field:
                continue
            for obj in scene.components(color=color):
                if obj.size != 4 or obj.bbox.height != 2 or obj.bbox.width != 2:
                    raise ValueError("every mapped object must be a solid 2x2 block")
                if any(int(source[cell]) != color for cell in obj.cells):
                    raise ValueError("object is not solid")
                center_row = (obj.bbox.r0 + obj.bbox.r1) / 2
                center_column = (obj.bbox.c0 + obj.bbox.c1) / 2
                if center_row == row_midpoint or center_column == column_midpoint:
                    raise ValueError("object lies on a quadrant boundary")
                target = (
                    0 if center_row < row_midpoint else 1,
                    0 if center_column < column_midpoint else 1,
                )
                if int(result[target]) != outer:
                    raise ValueError("multiple objects occupy one quadrant")
                result[target] = color
                object_count += 1
        if not object_count:
            raise ValueError("no quadrant objects found")
        return result

    return _decorate(transform, "quadrant_object_map")


def encode_cell_count_3x3(
    output_color: int,
    output_background: int = 0,
) -> Transform:
    """Encode one to four foreground cells along a fixed 3x3 count path."""

    if any(color not in range(10) for color in (output_color, output_background)):
        raise ValueError("ARC colors must be in [0, 9]")
    if output_color == output_background:
        raise ValueError("output and background colors must differ")
    order = ((0, 0), (0, 1), (0, 2), (1, 1))

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape != (3, 3):
            raise ValueError("cell-count encoding requires a 3x3 input")
        background = Scene(source).background_color()
        foreground = source[source != background]
        if not len(foreground) or len(set(int(value) for value in foreground)) != 1:
            raise ValueError("exactly one foreground color is required")
        count = int(len(foreground))
        if count > len(order):
            raise ValueError("foreground count exceeds the encoding capacity")
        result = np.full((3, 3), output_background, dtype=int)
        for cell in order[:count]:
            result[cell] = output_color
        return result

    return _decorate(transform, "encode_cell_count_3x3", output_color, output_background)


def encode_square_component_count_3x3(
    output_color: int,
    output_background: int = 0,
) -> Transform:
    """Count isolated solid 2x2 components into a five-position 3x3 code."""

    if any(color not in range(10) for color in (output_color, output_background)):
        raise ValueError("ARC colors must be in [0, 9]")
    if output_color == output_background:
        raise ValueError("output and background colors must differ")
    order = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape == (1, 1):
            raise ValueError("degenerate input")
        scene = Scene(source)
        background = scene.background_color()
        colors = [int(value) for value in np.unique(source) if int(value) != background]
        if len(colors) != 1:
            raise ValueError("exactly one object color is required")
        components = scene.components(color=colors[0])
        if not 1 <= len(components) <= len(order):
            raise ValueError("component count exceeds the encoding capacity")
        for obj in components:
            if obj.size != 4 or obj.bbox.height != 2 or obj.bbox.width != 2:
                raise ValueError("every component must be a solid 2x2 block")
            if any(int(source[cell]) != colors[0] for cell in obj.cells):
                raise ValueError("component is not solid")
        result = np.full((3, 3), output_background, dtype=int)
        for cell in order[:len(components)]:
            result[cell] = output_color
        return result

    return _decorate(
        transform,
        "encode_square_component_count_3x3",
        output_color,
        output_background,
    )


def component_segment_histogram(
    rows: int,
    columns: int,
    output_background: int = 0,
) -> Transform:
    """Pack descending colors repeated by their horizontal segment counts."""

    if rows < 1 or columns < 1:
        raise ValueError("histogram dimensions must be positive")
    if output_background not in range(10):
        raise ValueError("output background must be in [0, 9]")

    def transform(grid: np.ndarray) -> np.ndarray:
        source = _grid(grid)
        if source.shape == (1, 1):
            raise ValueError("degenerate input")
        scene = Scene(source)
        background = scene.background_color()
        colors = sorted(
            (int(value) for value in np.unique(source) if int(value) != background),
            reverse=True,
        )
        if not colors:
            raise ValueError("segments are absent")
        encoded: list[int] = []
        for color in colors:
            components = scene.components(color=color)
            for obj in components:
                if obj.bbox.height != 1:
                    raise ValueError("every component must be a horizontal segment")
            encoded.extend([color] * len(components))
        capacity = rows * columns
        if len(encoded) > capacity:
            raise ValueError("segment count exceeds the histogram capacity")
        encoded.extend([output_background] * (capacity - len(encoded)))
        return np.asarray(encoded, dtype=int).reshape(rows, columns)

    return _decorate(
        transform,
        "component_segment_histogram",
        rows,
        columns,
        output_background,
    )
