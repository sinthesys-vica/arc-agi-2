"""Solve — end-to-end orchestration per task.

Owner: Raiden
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any, Optional

from .scene import Scene
from .proposers import Proposer
from .search import search, SearchResult


def solve_task(task: dict, proposers: list[Proposer]) -> dict:
    """Solve a single ARC task.

    Args:
        task: ARC task dict with "train" and "test" keys
        proposers: list of hypothesis proposers

    Returns:
        dict with "task_id" and "predictions" (list of attempt dicts)
    """
    train_pairs = task["train"]
    test_inputs = task["test"]

    result = search(proposers, train_pairs)

    predictions = []
    for test_case in test_inputs:
        input_grid = np.array(test_case["input"], dtype=int)
        attempts = []

        if result.best is not None:
            attempt1 = result.best.transform(input_grid)
            attempts.append(attempt1.tolist())
        else:
            # Fallback: return input unchanged
            attempts.append(input_grid.tolist())

        if result.alt is not None:
            attempt2 = result.alt.transform(input_grid)
            attempts.append(attempt2.tolist())
        elif result.best is not None:
            # No structurally different alt — duplicate best
            attempts.append(attempts[0])
        else:
            attempts.append(input_grid.tolist())

        predictions.append({
            "attempt_1": attempts[0],
            "attempt_2": attempts[1],
        })

    return {
        "solved": result.solved,
        "best_description": result.best.description if result.best else None,
        "alt_description": result.alt.description if result.alt else None,
        "predictions": predictions,
    }


def solve_all(tasks: dict[str, dict], proposers: list[Proposer]) -> dict[str, dict]:
    """Solve all tasks in a challenge set.

    Args:
        tasks: dict of task_id -> task dict
        proposers: all available proposers

    Returns:
        dict of task_id -> solve result
    """
    results = {}
    for task_id, task in tasks.items():
        results[task_id] = solve_task(task, proposers)
    return results
