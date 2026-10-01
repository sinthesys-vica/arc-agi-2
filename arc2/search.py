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

import numpy as np

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
# Feature-based pre-filtering
# ---------------------------------------------------------------------------

class TaskSignature:
    """Lightweight feature summary of a task's training pairs.

    Used to skip proposer families that cannot possibly match:
    e.g., ResizeProposer on same_size tasks, GeometricProposer on
    tasks where the color palette changes.
    """

    def __init__(self, train_pairs: list[dict]):
        # Lightweight: only compute what we need for filtering,
        # NOT full encode_pair (which runs expensive component extraction)
        self.pairs = []
        for p in train_pairs:
            inp = np.asarray(p["input"], dtype=int)
            out = np.asarray(p["output"], dtype=int)
            in_colors = set(int(x) for x in np.unique(inp))
            out_colors = set(int(x) for x in np.unique(out))
            self.pairs.append({
                "same_size": inp.shape == out.shape,
                "size_delta": [out.shape[0] - inp.shape[0], out.shape[1] - inp.shape[1]],
                "size_ratio": [
                    round(out.shape[0] / inp.shape[0], 4),
                    round(out.shape[1] / inp.shape[1], 4),
                ] if inp.shape[0] and inp.shape[1] else None,
                "colors_only_in": sorted(in_colors - out_colors),
                "colors_only_out": sorted(out_colors - in_colors),
            })

    @property
    def all_same_size(self) -> bool:
        return all(p["same_size"] for p in self.pairs)

    @property
    def all_different_size(self) -> bool:
        return all(not p["same_size"] for p in self.pairs)

    @property
    def has_color_change(self) -> bool:
        """True if any pair has colors appearing/disappearing."""
        return any(p["colors_only_in"] or p["colors_only_out"] for p in self.pairs)

    @property
    def consistent_size_ratio(self) -> tuple[float, float] | None:
        """If all pairs share the same integer size ratio, return it."""
        if not self.pairs or self.all_same_size:
            return None
        ratios = [tuple(p["size_ratio"]) for p in self.pairs if p["size_ratio"]]
        if not ratios:
            return None
        if all(r == ratios[0] for r in ratios):
            return ratios[0]
        return None

    @property
    def output_smaller(self) -> bool:
        """True if output is strictly smaller than input in all pairs."""
        return all(
            p["size_delta"][0] < 0 or p["size_delta"][1] < 0
            for p in self.pairs
        )


def should_skip_proposer(proposer: Proposer, sig: TaskSignature) -> bool:
    """Return True if this proposer cannot match the task signature.

    Conservative: only skips when we're CERTAIN the proposer can't help.
    False negatives are fine; false positives lose solutions.
    """
    name = type(proposer).__name__

    # ResizeProposer: skip if all pairs are same size
    if name == "ResizeProposer" and sig.all_same_size:
        return True

    # GeometricProposer (pure D4): skip if colors change between in/out
    # (D4 transforms don't change colors)
    if name == "GeometricProposer" and sig.has_color_change:
        return True

    # CropProposer: skip if output is larger than input
    if name == "CropProposer" and not sig.output_smaller and not sig.all_same_size:
        # Crop can only make things smaller or same
        if all(
            p["size_delta"][0] >= 0 and p["size_delta"][1] >= 0
            for p in sig.pairs
        ):
            # Output >= input in both dims, and not same size → growing, not cropping
            if sig.all_different_size:
                return True

    # Same-size repair families: skip when sizes differ
    if name in (
        "WallpaperRepairProposer",
        "SymmetryCompleterProposer",
        "ShapeUnifierProposer",
    ) and sig.all_different_size:
        return True

    # PixelExpansionProposer: skip same-size tasks (needs integer scaling)
    if name == "PixelExpansionProposer" and sig.all_same_size:
        return True

    return False


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

    # Pre-compute task signature for proposer filtering
    task_sig = TaskSignature(train_pairs)

    ordered = sort_by_tier(proposers)
    valid: list[Hypothesis] = []
    found_best = False

    for proposer in ordered:
        # Time check
        elapsed = time.monotonic() - t0
        if elapsed >= time_budget_seconds:
            break

        # Skip proposers that can't match this task's signature
        if should_skip_proposer(proposer, task_sig):
            continue

        stats.proposers_tried += 1
        try:
            # Atlas interface: propose(inputs, outputs=None)
            # With outputs: relation-driven candidates (color map, size ratio, etc.)
            # Without outputs: generic primitive enumeration
            candidates = proposer.propose(scenes_input, outputs=scenes_output) or []
        except TypeError:
            # Proposer doesn't accept outputs kwarg — fall back
            try:
                candidates = proposer.propose(scenes_input) or []
            except Exception:
                continue
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
