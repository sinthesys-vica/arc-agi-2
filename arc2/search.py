"""Search — hypothesis space exploration strategy.

Owner: Raiden

Search taxonomy derived from 20 solved tasks (rounds 1-6 + bonus):

GEOMETRIC: tiling, scaling (2x), rotation, reflection, D4-orbit
OBJECT:    extraction, counting, sorting, filtering by property
COLOR:     swapping, remapping, complementing, frequency-based
SPATIAL:   flood-fill, path-finding, region ops, obstacle avoidance
STRUCTURAL: fractal/self-similar, analogy, embedding/nesting, marker-based

The search is iterative-deepening: cheap proposers first (color, simple geometric),
then compositional, then structural. Early exit on verified solution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .scene import Scene
from .proposers import Hypothesis, Proposer
from .verify import verify_hypothesis, structural_distance as _structural_distance


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SearchStats:
    """Telemetry for a single task search."""
    hypotheses_generated: int = 0
    hypotheses_verified: int = 0
    proposers_tried: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class SearchResult:
    """Result of hypothesis search for a single task."""
    best: Optional[Hypothesis] = None
    alt: Optional[Hypothesis] = None
    stats: SearchStats = field(default_factory=SearchStats)

    @property
    def solved(self) -> bool:
        return self.best is not None

    def __repr__(self) -> str:
        if not self.solved:
            return f"SearchResult(unsolved, {self.stats.hypotheses_generated} tried)"
        alt_desc = f", alt={self.alt}" if self.alt else ""
        return f"SearchResult(best={self.best}{alt_desc})"


# ---------------------------------------------------------------------------
# Proposer priority tiers
# ---------------------------------------------------------------------------

class ProposerTier:
    """Tiered proposer ordering for iterative-deepening search.

    Tier 0 (fast):   single-primitive transforms (recolor, flip, crop, scale)
    Tier 1 (medium): object-based (extract, filter, sort, count)
    Tier 2 (deep):   compositional (compose 2-3 primitives)
    Tier 3 (exotic): structural (fractal, analogy, symmetry-orbit, path)
    """
    FAST = 0
    MEDIUM = 1
    DEEP = 2
    EXOTIC = 3


def classify_proposer(proposer: Proposer) -> int:
    """Classify a proposer into a priority tier.

    Uses the proposer's `tier` attribute if set, otherwise defaults to MEDIUM.
    """
    return getattr(proposer, "tier", ProposerTier.MEDIUM)


def sort_by_tier(proposers: list[Proposer]) -> list[Proposer]:
    """Sort proposers by tier (fast first)."""
    return sorted(proposers, key=classify_proposer)


# ---------------------------------------------------------------------------
# Structural diversity
# ---------------------------------------------------------------------------

STRUCTURAL_DISTANCE_THRESHOLD = 0.3  # minimum distance to count as "different"


def structurally_different(h1: Hypothesis, h2: Hypothesis) -> bool:
    """Check if two hypotheses are structurally different.

    Uses Dorian's structural_distance (Jaccard over description tokens
    and compose() component identities) rather than simple string compare.
    Threshold: >= 0.3 on [0, 1] scale.
    """
    return _structural_distance(h1, h2) >= STRUCTURAL_DISTANCE_THRESHOLD


def select_alt(best: Hypothesis, candidates: list[Hypothesis]) -> Optional[Hypothesis]:
    """Select the most structurally different alternative to `best`.

    Sorts candidates by structural distance (descending) and picks the
    most different one that still verifies.
    """
    scored = [(h, _structural_distance(best, h)) for h in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    for h, dist in scored:
        if dist >= STRUCTURAL_DISTANCE_THRESHOLD:
            return h
    return None


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search(
    proposers: list[Proposer],
    train_pairs: list[dict],
    max_hypotheses: int = 10000,
    time_budget_seconds: float = 300.0,
) -> SearchResult:
    """Search for valid hypotheses using tiered iterative-deepening.

    Strategy:
    1. Sort proposers by tier (fast → exotic)
    2. Generate hypotheses tier by tier
    3. Verify each immediately — early exit on first verified hit
    4. Continue through remaining tiers to find alt
    5. Return best + best structurally different alt

    Args:
        proposers: all available hypothesis proposers
        train_pairs: training input/output pairs
        max_hypotheses: hard cap on total hypotheses evaluated
        time_budget_seconds: wall-clock time limit for this task

    Returns:
        SearchResult with best, alt, and stats
    """
    t0 = time.monotonic()
    stats = SearchStats()

    scenes_input = [Scene.from_list(pair["input"]) for pair in train_pairs]
    scenes_output = [Scene.from_list(pair["output"]) for pair in train_pairs]

    ordered = sort_by_tier(proposers)
    valid: list[Hypothesis] = []
    found_best = False

    for proposer in ordered:
        # Time check
        elapsed = time.monotonic() - t0
        if elapsed >= time_budget_seconds:
            break

        stats.proposers_tried += 1
        try:
            candidates = proposer.propose(scenes_input)
        except Exception:
            continue

        for h in candidates:
            stats.hypotheses_generated += 1
            if stats.hypotheses_generated >= max_hypotheses:
                break

            # Verify
            if verify_hypothesis(h, train_pairs):
                stats.hypotheses_verified += 1
                valid.append(h)

                if not found_best:
                    found_best = True
                    # Don't stop — keep looking for alt

            # Time check inside inner loop (every 100 hypotheses)
            if stats.hypotheses_generated % 100 == 0:
                if time.monotonic() - t0 >= time_budget_seconds:
                    break

        if stats.hypotheses_generated >= max_hypotheses:
            break

        # If we have best + alt, we can stop early
        if len(valid) >= 2:
            best = valid[0]
            alt = select_alt(best, valid[1:])
            if alt is not None:
                # We have a structurally different pair — stop
                break

    stats.elapsed_seconds = time.monotonic() - t0

    if not valid:
        return SearchResult(stats=stats)

    best = valid[0]
    alt = select_alt(best, valid[1:])

    return SearchResult(best=best, alt=alt, stats=stats)


# ---------------------------------------------------------------------------
# Budget-aware multi-task search
# ---------------------------------------------------------------------------

def search_with_budget(
    proposers: list[Proposer],
    tasks: dict[str, dict],
    total_time_seconds: float = 36000.0,  # 10 hours (leave 2h margin from 12h)
) -> dict[str, SearchResult]:
    """Search all tasks with a shared time budget.

    Allocates time per task based on remaining budget and remaining tasks.
    Easy tasks (solved quickly) donate their leftover time to harder ones.

    Args:
        proposers: all available proposers
        tasks: dict of task_id -> task dict
        total_time_seconds: total wall-clock budget

    Returns:
        dict of task_id -> SearchResult
    """
    results: dict[str, SearchResult] = {}
    t0 = time.monotonic()
    remaining_ids = list(tasks.keys())

    for i, task_id in enumerate(remaining_ids):
        elapsed = time.monotonic() - t0
        remaining_time = total_time_seconds - elapsed
        remaining_tasks = len(remaining_ids) - i

        if remaining_time <= 0:
            # Out of time — mark remaining as unsolved
            results[task_id] = SearchResult()
            continue

        # Adaptive per-task budget: equal share of remaining time
        per_task = remaining_time / remaining_tasks

        result = search(
            proposers=proposers,
            train_pairs=tasks[task_id]["train"],
            time_budget_seconds=per_task,
        )
        results[task_id] = result

    return results
