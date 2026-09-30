"""SceneGraph IR for ARC grids.

Owner: Atlas

The scene layer is deterministic and task-id agnostic. It turns an integer
grid into reusable structural facts; it does not guess a solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np


GridLike = np.ndarray | list[list[int]]
Cell = tuple[int, int]


@dataclass(frozen=True, slots=True, order=True)
class BBox:
    """Inclusive rectangular bounding box."""

    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    def touches_border(self, height: int, width: int) -> bool:
        return self.r0 == 0 or self.c0 == 0 or self.r1 == height - 1 or self.c1 == width - 1

    def as_list(self) -> list[int]:
        return [self.r0, self.c0, self.r1, self.c1]


@dataclass(frozen=True, slots=True)
class SceneObject:
    """A monochrome connected component."""

    color: int
    cells: tuple[Cell, ...]
    bbox: BBox
    connectivity: int = 4

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def density(self) -> float:
        return self.size / self.bbox.area

    @property
    def is_solid_rectangle(self) -> bool:
        return self.size == self.bbox.area

    def touches_border(self, height: int, width: int) -> bool:
        return self.bbox.touches_border(height, width)

    def normalized_cells(self) -> tuple[Cell, ...]:
        return tuple((r - self.bbox.r0, c - self.bbox.c0) for r, c in self.cells)

    def to_dict(self, height: int, width: int) -> dict:
        return {
            "color": self.color,
            "size": self.size,
            "bbox": self.bbox.as_list(),
            "density": self.density,
            "touching_border": self.touches_border(height, width),
            "solid_rectangle": self.is_solid_rectangle,
            "cells": [list(cell) for cell in self.cells],
        }


def _validated_grid(grid: GridLike) -> np.ndarray:
    raw = np.asarray(grid)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("ARC grid must be a non-empty rectangular 2D array")
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            integer = raw.astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("ARC grid must contain integers") from exc
        if not np.array_equal(raw, integer):
            raise ValueError("ARC grid must contain integers")
        raw = integer
    arr = np.array(raw, dtype=int, copy=True)
    if arr.min() < 0 or arr.max() > 9:
        raise ValueError("ARC colors must be integers in [0, 9]")
    arr.setflags(write=False)
    return arr


class Scene:
    """Immutable structured representation of an ARC grid."""

    def __init__(self, grid: GridLike):
        self.grid = _validated_grid(grid)
        self.height, self.width = self.grid.shape

    @classmethod
    def from_list(cls, grid_list: list[list[int]]) -> "Scene":
        return cls(grid_list)

    @cached_property
    def colors(self) -> tuple[int, ...]:
        return tuple(int(x) for x in np.unique(self.grid))

    @cached_property
    def color_counts(self) -> dict[int, int]:
        return {color: int(np.count_nonzero(self.grid == color)) for color in self.colors}

    @cached_property
    def _border_counts(self) -> dict[int, int]:
        border = np.concatenate((
            self.grid[0, :],
            self.grid[-1, :],
            self.grid[1:-1, 0],
            self.grid[1:-1, -1],
        ))
        return {color: int(np.count_nonzero(border == color)) for color in self.colors}

    def background_color(self) -> int:
        """Infer background from frequency, border frequency, then color 0."""

        return min(
            self.colors,
            key=lambda color: (
                -self.color_counts[color],
                -self._border_counts[color],
                color != 0,
                color,
            ),
        )

    @staticmethod
    def _neighbors(cell: Cell, connectivity: int) -> Iterable[Cell]:
        r, c = cell
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        if connectivity == 8:
            offsets += ((1, 1), (1, -1), (-1, 1), (-1, -1))
        elif connectivity != 4:
            raise ValueError("connectivity must be 4 or 8")
        for dr, dc in offsets:
            yield r + dr, c + dc

    def components(
        self,
        *,
        connectivity: int = 4,
        include_background: bool = False,
        color: int | None = None,
    ) -> list[SceneObject]:
        """Return monochrome connected components in stable reading order."""

        background = self.background_color()
        seen = np.zeros(self.grid.shape, dtype=bool)
        result: list[SceneObject] = []
        for r in range(self.height):
            for c in range(self.width):
                value = int(self.grid[r, c])
                if seen[r, c]:
                    continue
                if color is not None and value != color:
                    continue
                if not include_background and value == background:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells: list[Cell] = []
                while stack:
                    current = stack.pop()
                    cells.append(current)
                    for nr, nc in self._neighbors(current, connectivity):
                        if not (0 <= nr < self.height and 0 <= nc < self.width):
                            continue
                        if seen[nr, nc] or int(self.grid[nr, nc]) != value:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                ordered = tuple(sorted(cells))
                rows = [cell[0] for cell in ordered]
                cols = [cell[1] for cell in ordered]
                result.append(SceneObject(
                    color=value,
                    cells=ordered,
                    bbox=BBox(min(rows), min(cols), max(rows), max(cols)),
                    connectivity=connectivity,
                ))
        result.sort(key=lambda obj: (obj.bbox.r0, obj.bbox.c0, obj.color, -obj.size))
        return result

    def objects(self, connectivity: int = 4) -> list[SceneObject]:
        return self.components(connectivity=connectivity)

    def background_regions(self, connectivity: int = 4) -> list[SceneObject]:
        return self.components(
            connectivity=connectivity,
            include_background=True,
            color=self.background_color(),
        )

    def holes(self, connectivity: int = 4) -> list[SceneObject]:
        return [
            region for region in self.background_regions(connectivity)
            if not region.touches_border(self.height, self.width)
        ]

    def non_background_bbox(self) -> BBox | None:
        cells = np.argwhere(self.grid != self.background_color())
        if cells.size == 0:
            return None
        return BBox(
            int(cells[:, 0].min()),
            int(cells[:, 1].min()),
            int(cells[:, 0].max()),
            int(cells[:, 1].max()),
        )

    def crop(self, bbox: BBox) -> np.ndarray:
        return np.array(self.grid[bbox.r0:bbox.r1 + 1, bbox.c0:bbox.c1 + 1], copy=True)

    def uniform_rows(self, color: int | None = None) -> tuple[int, ...]:
        return tuple(
            index for index, row in enumerate(self.grid)
            if np.all(row == row[0]) and (color is None or int(row[0]) == color)
        )

    def uniform_cols(self, color: int | None = None) -> tuple[int, ...]:
        return tuple(
            index for index, column in enumerate(self.grid.T)
            if np.all(column == column[0]) and (color is None or int(column[0]) == color)
        )

    def symmetries(self) -> dict[str, bool]:
        """Exact finite-grid dihedral symmetries."""

        result = {
            "identity": True,
            "flip_lr": bool(np.array_equal(self.grid, self.grid[:, ::-1])),
            "flip_ud": bool(np.array_equal(self.grid, self.grid[::-1, :])),
            "rot180": bool(np.array_equal(self.grid, self.grid[::-1, ::-1])),
        }
        if self.height == self.width:
            result.update({
                "rot90": bool(np.array_equal(self.grid, np.rot90(self.grid, 1))),
                "rot270": bool(np.array_equal(self.grid, np.rot90(self.grid, 3))),
                "transpose": bool(np.array_equal(self.grid, self.grid.T)),
                "anti_transpose": bool(np.array_equal(self.grid, np.rot90(self.grid, 2).T)),
            })
        return result

    def nested_rectangular_palette(self) -> tuple[int, ...]:
        """Recover contracting rectangular color layers, preserving repeats."""

        top, left = 0, 0
        bottom, right = self.height - 1, self.width - 1
        palette: list[int] = []
        while True:
            current = int(self.grid[top, left])
            palette.append(current)
            window = self.grid[top:bottom + 1, left:right + 1]
            different = np.argwhere(window != current)
            if different.size == 0:
                return tuple(palette)
            next_top = top + int(different[:, 0].min())
            next_left = left + int(different[:, 1].min())
            next_bottom = top + int(different[:, 0].max())
            next_right = left + int(different[:, 1].max())
            if (next_top, next_left, next_bottom, next_right) == (top, left, bottom, right):
                raise ValueError("grid is not a contracting rectangular nesting")
            top, left, bottom, right = next_top, next_left, next_bottom, next_right

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "colors": list(self.colors),
            "color_counts": self.color_counts,
            "background": self.background_color(),
            "objects": [obj.to_dict(self.height, self.width) for obj in self.objects()],
            "holes": [hole.to_dict(self.height, self.width) for hole in self.holes()],
            "symmetries": self.symmetries(),
            "uniform_rows": list(self.uniform_rows()),
            "uniform_cols": list(self.uniform_cols()),
        }

    def __repr__(self) -> str:
        return f"Scene({self.height}x{self.width}, background={self.background_color()})"
