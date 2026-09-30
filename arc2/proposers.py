"""Hypothesis proposers — generate candidate transforms from scene + DSL.

Owner: Atlas
"""

from __future__ import annotations

from typing import Any
from .scene import Scene
from .dsl import Transform


class Hypothesis:
    """A candidate transformation hypothesis."""

    def __init__(self, transform: Transform, description: str, confidence: float = 0.0):
        self.transform = transform
        self.description = description
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"Hypothesis({self.description}, conf={self.confidence:.2f})"


class Proposer:
    """Base class for hypothesis proposers."""

    def propose(self, scenes: list[Scene]) -> list[Hypothesis]:
        """Generate hypotheses from training examples."""
        raise NotImplementedError("Atlas: implement proposer")


# --- Atlas: add concrete proposers below ---
