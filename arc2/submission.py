"""Submission — Kaggle notebook wrapper.

Owner: Raiden

This module generates the submission.json for Kaggle.
The actual notebook is a thin wrapper that imports this.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def format_submission(results: dict[str, dict]) -> dict:
    """Format solve results into Kaggle submission format.

    Args:
        results: dict of task_id -> solve result from solve_all

    Returns:
        Kaggle-compatible submission dict
    """
    submission = {}
    for task_id, result in results.items():
        for i, pred in enumerate(result["predictions"]):
            key = f"{task_id}_{i}"
            submission[key] = {
                "attempt_1": pred["attempt_1"],
                "attempt_2": pred["attempt_2"],
            }
    return submission


def write_submission(results: dict[str, dict], output_path: str = "submission.json") -> Path:
    """Write submission file.

    Args:
        results: dict of task_id -> solve result
        output_path: where to write the JSON

    Returns:
        Path to written file
    """
    submission = format_submission(results)
    path = Path(output_path)
    with open(path, "w") as f:
        json.dump(submission, f)
    return path
