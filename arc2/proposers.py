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
    """Base class for hypothesis proposers.

    The `tier` attribute controls search order (see search.ProposerTier):
        0 = fast (single primitives)
        1 = medium (object-based)
        2 = deep (compositional)
        3 = exotic (structural)
    """

    tier: int = 1  # default: medium

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses from training examples.

        Args:
            inputs: input scenes from training pairs
            outputs: output scenes (optional) — enables relation-driven
                     proposing (color maps, size ratios, structural diffs)

        Returns:
            list of candidate hypotheses
        """
        raise NotImplementedError("Atlas: implement proposer")


# --- Atlas: add concrete proposers below ---
