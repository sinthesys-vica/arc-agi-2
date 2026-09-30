"""Solve — end-to-end orchestration per task.

Owner: Raiden
"""

from __future__ import annotations

import json
import logging
import numpy as np
from typing import Any, Optional

from .scene import Scene
from .proposers import Proposer
from .search import search, search_with_budget, SearchResult
from .verify import apply_with_verify

logger = logging.getLogger(__name__)


def solve_task(task: dict, proposers: list[Proposer], time_budget: float = 300.0) -> dict:
    """Solve a single ARC task.

    Uses Dorian's apply_with_verify for fail-closed prediction writing.

    Args:
        task: ARC task dict with "train" and "test" keys
        proposers: list of hypothesis proposers
        time_budget: seconds allowed for this task's search

    Returns:
        dict with solve status, descriptions, predictions, and stats
    """
    train_pairs = task["train"]
    test_inputs = task["test"]

    result = search(proposers, train_pairs, time_budget_seconds=time_budget)

    predictions = []
    reports = []
    for test_case in test_inputs:
        input_grid = np.array(test_case["input"], dtype=int)

        # Attempt 1: best hypothesis (via apply_with_verify)
        if result.best is not None:
            grid1, report1 = apply_with_verify(result.best, input_grid)
        else:
            grid1 = input_grid.copy()
            report1 = {"status": "no_hypothesis", "reason": "search found nothing"}

        # Attempt 2: structurally different alt (via apply_with_verify)
        if result.alt is not None:
            grid2, report2 = apply_with_verify(result.alt, input_grid)
        elif result.best is not None:
            # No alt found — duplicate best
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

    return {
        "solved": result.solved,
        "best_description": result.best.description if result.best else None,
        "alt_description": result.alt.description if result.alt else None,
        "predictions": predictions,
        "reports": reports,
        "stats": {
            "hypotheses_generated": result.stats.hypotheses_generated,
            "hypotheses_verified": result.stats.hypotheses_verified,
            "proposers_tried": result.stats.proposers_tried,
            "elapsed_seconds": round(result.stats.elapsed_seconds, 3),
        },
    }


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
        task = tasks[task_id]
        test_inputs = task["test"]
        predictions = []
        reports = []

        for test_case in test_inputs:
            input_grid = np.array(test_case["input"], dtype=int)

            if sr.best is not None:
                grid1, report1 = apply_with_verify(sr.best, input_grid)
            else:
                grid1 = input_grid.copy()
                report1 = {"status": "no_hypothesis", "reason": "search found nothing"}

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

        results[task_id] = {
            "solved": sr.solved,
            "best_description": sr.best.description if sr.best else None,
            "alt_description": sr.alt.description if sr.alt else None,
            "predictions": predictions,
            "reports": reports,
            "stats": {
                "hypotheses_generated": sr.stats.hypotheses_generated,
                "hypotheses_verified": sr.stats.hypotheses_verified,
                "proposers_tried": sr.stats.proposers_tried,
                "elapsed_seconds": round(sr.stats.elapsed_seconds, 3),
            },
        }

    solved = sum(1 for r in results.values() if r["solved"])
    total = len(results)
    logger.info(f"Solved {solved}/{total} tasks")

    return results
