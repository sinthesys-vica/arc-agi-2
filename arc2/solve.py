"""Solve — end-to-end orchestration per task.

Owner: Raiden
"""

from __future__ import annotations

import logging
import numpy as np

from .proposers import Proposer
from .search import search, search_with_budget, SearchResult
from .verify import apply_with_verify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared prediction writer
# ---------------------------------------------------------------------------

def _write_predictions(
    sr: SearchResult,
    test_inputs: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Write predictions for all test inputs using a search result.

    Fail-closed via apply_with_verify. Returns (predictions, reports).
    """
    predictions = []
    reports = []

    for test_case in test_inputs:
        input_grid = np.array(test_case["input"], dtype=int)

        # Attempt 1: best hypothesis
        if sr.best is not None:
            grid1, report1 = apply_with_verify(sr.best, input_grid)
        else:
            grid1 = input_grid.copy()
            report1 = {"status": "no_hypothesis", "reason": "search found nothing"}

        # Attempt 2: structurally different alt
        if sr.alt is not None:
            grid2, report2 = apply_with_verify(sr.alt, input_grid)
        elif sr.best is not None:
            grid2 = grid1.copy()
            report2 = {"status": "duplicated_best", "reason": "no structural alt"}
        else:
            grid2 = input_grid.copy()
            report2 = {"status": "no_hypothesis", "reason": "search found nothing"}

        predictions.append({
            "attempt_1": grid1.tolist(),
            "attempt_2": grid2.tolist(),
        })
        reports.append({"attempt_1": report1, "attempt_2": report2})

    return predictions, reports


def _format_stats(sr: SearchResult) -> dict:
    """Format search stats for output."""
    return {
        "hypotheses_generated": sr.stats.hypotheses_generated,
        "hypotheses_verified": sr.stats.hypotheses_verified,
        "proposers_tried": sr.stats.proposers_tried,
        "elapsed_seconds": round(sr.stats.elapsed_seconds, 3),
    }


# ---------------------------------------------------------------------------
# Single task
# ---------------------------------------------------------------------------

def solve_task(task: dict, proposers: list[Proposer], time_budget: float = 300.0) -> dict:
    """Solve a single ARC task.

    Args:
        task: ARC task dict with "train" and "test" keys
        proposers: list of hypothesis proposers
        time_budget: seconds allowed for this task's search

    Returns:
        dict with solve status, descriptions, predictions, and stats
    """
    result = search(proposers, task["train"], time_budget_seconds=time_budget)
    predictions, reports = _write_predictions(result, task["test"])

    return {
        "solved": result.solved,
        "best_description": result.best.description if result.best else None,
        "alt_description": result.alt.description if result.alt else None,
        "predictions": predictions,
        "reports": reports,
        "stats": _format_stats(result),
    }


# ---------------------------------------------------------------------------
# All tasks with budget
# ---------------------------------------------------------------------------

def solve_all(
    tasks: dict[str, dict],
    proposers: list[Proposer],
    total_time_seconds: float = 36000.0,
) -> dict[str, dict]:
    """Solve all tasks with budget-aware scheduling.

    Args:
        tasks: dict of task_id -> task dict
        proposers: all available proposers
        total_time_seconds: total wall-clock budget (default 10h, 2h margin)

    Returns:
        dict of task_id -> solve result
    """
    search_results = search_with_budget(proposers, tasks, total_time_seconds)

    results = {}
    for task_id, sr in search_results.items():
        predictions, reports = _write_predictions(sr, tasks[task_id]["test"])
        results[task_id] = {
            "solved": sr.solved,
            "best_description": sr.best.description if sr.best else None,
            "alt_description": sr.alt.description if sr.alt else None,
            "predictions": predictions,
            "reports": reports,
            "stats": _format_stats(sr),
        }

    solved = sum(1 for r in results.values() if r["solved"])
    logger.info(f"Solved {solved}/{len(results)} tasks")

    return results
