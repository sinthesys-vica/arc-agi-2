"""Small, deterministic transformation DSL for ARC grids.

Owner: Atlas

Every primitive is a pure ``grid -> grid`` function. Functions carry a stable
program signature so search can deduplicate candidates without descriptions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .scene import Scene


Transform = Callable[[np.ndarray], np.ndarray]


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


def signature(transform: Transform) -> tuple:
    """Return a stable structural signature for a transform."""

    stored = getattr(transform, "_arc2_signature", None)
    if stored is not None:
        return tuple(stored)
    return (getattr(transform, "__qualname__", repr(transform)),)


def identity(grid: np.ndarray) -> np.ndarray:
    """Return an independent copy of the input."""

    return _grid(grid).copy()


_decorate(identity, "identity")


def compose(*transforms: Transform) -> Transform:
    """Compose transforms left-to-right: ``compose(f, g)(x) = g(f(x))``."""

    if not transforms:
        return identity

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
