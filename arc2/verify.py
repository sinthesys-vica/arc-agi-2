"""Verification — validate hypotheses against training examples.

Owner: Dorian
"""

from __future__ import annotations

import numpy as np
from .scene import Scene
from .proposers import Hypothesis


def verify_hypothesis(hypothesis: Hypothesis, train_pairs: list[dict]) -> bool:
    """Check if a hypothesis correctly transforms all training inputs to outputs.

    Args:
        hypothesis: candidate transformation
        train_pairs: list of {"input": grid, "output": grid} dicts

    Returns:
        True if hypothesis produces exact output for all training pairs
    """
    for pair in train_pairs:
        input_grid = np.array(pair["input"], dtype=int)
        expected = np.array(pair["output"], dtype=int)
        result = hypothesis.transform(input_grid)
        if result.shape != expected.shape or not np.array_equal(result, expected):
            return False
    return True


def score_hypothesis(hypothesis: Hypothesis, train_pairs: list[dict]) -> float:
    """Score a hypothesis: fraction of training pairs correctly transformed."""
    if not train_pairs:
        return 0.0
    correct = sum(
        1 for pair in train_pairs
        if np.array_equal(
            hypothesis.transform(np.array(pair["input"], dtype=int)),
            np.array(pair["output"], dtype=int)
        )
    )
    return correct / len(train_pairs)


# --- Dorian: add encoder, writer-with-verify below ---
