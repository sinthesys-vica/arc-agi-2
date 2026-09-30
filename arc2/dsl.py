"""DSL — composable transformation primitives.

Owner: Atlas
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# Type alias for a transformation: grid -> grid
Transform = Callable[[np.ndarray], np.ndarray]


def identity(grid: np.ndarray) -> np.ndarray:
    """No-op transform."""
    return grid.copy()


def compose(*transforms: Transform) -> Transform:
    """Compose transforms left-to-right: compose(f, g)(x) = g(f(x))."""
    def composed(grid: np.ndarray) -> np.ndarray:
        result = grid
        for t in transforms:
            result = t(result)
        return result
    return composed


# --- Atlas: add primitives below ---
# Examples: rotate, flip, recolor, crop, tile, scale, mask, fill, etc.
