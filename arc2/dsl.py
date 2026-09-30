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
