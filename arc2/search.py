"""Search — hypothesis space exploration strategy.

Owner: Raiden
"""

from __future__ import annotations

from typing import Optional
from .scene import Scene
from .proposers import Hypothesis, Proposer
from .verify import verify_hypothesis, score_hypothesis


class SearchResult:
    """Result of hypothesis search for a single task."""

    def __init__(self, best: Optional[Hypothesis] = None, alt: Optional[Hypothesis] = None):
        self.best = best  # attempt 1: best hypothesis
        self.alt = alt    # attempt 2: best structurally different hypothesis

    @property
    def solved(self) -> bool:
        return self.best is not None

    def __repr__(self) -> str:
        if not self.solved:
            return "SearchResult(unsolved)"
        return f"SearchResult(best={self.best}, alt={self.alt})"


def search(
    proposers: list[Proposer],
    train_pairs: list[dict],
    max_hypotheses: int = 1000,
) -> SearchResult:
    """Search for valid hypotheses using all proposers.

    Strategy:
    1. Generate hypotheses from all proposers
    2. Verify each against training pairs
    3. Return best + best structurally different

    Args:
        proposers: list of hypothesis proposers
        train_pairs: training input/output pairs
        max_hypotheses: cap on total hypotheses to evaluate

    Returns:
        SearchResult with best and alt hypotheses
    """
    scenes = [Scene.from_list(pair["input"]) for pair in train_pairs]

    # Collect all hypotheses
    all_hypotheses: list[Hypothesis] = []
    for proposer in proposers:
        candidates = proposer.propose(scenes)
        all_hypotheses.extend(candidates)
        if len(all_hypotheses) >= max_hypotheses:
            all_hypotheses = all_hypotheses[:max_hypotheses]
            break

    # Verify and score
    valid: list[Hypothesis] = []
    for h in all_hypotheses:
        if verify_hypothesis(h, train_pairs):
            h.confidence = score_hypothesis(h, train_pairs)
            valid.append(h)

    if not valid:
        return SearchResult()

    # Sort by confidence (all should be 1.0 if verify passed)
    valid.sort(key=lambda h: h.confidence, reverse=True)

    best = valid[0]
    # Find structurally different alt
    alt = None
    for h in valid[1:]:
        if h.description != best.description:
            alt = h
            break

    return SearchResult(best=best, alt=alt)
