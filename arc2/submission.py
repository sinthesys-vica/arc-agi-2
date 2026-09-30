"""Submission — Kaggle notebook entry point.

Owner: Raiden

Generates submission.json from the evaluation set.
The Kaggle notebook imports this module and calls main().
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .proposers import default_proposers
from .solve import solve_all

logger = logging.getLogger(__name__)

# Kaggle constraints
TOTAL_TIME_BUDGET = 36000.0  # 10h (2h margin from 12h limit)
MAX_GRID_SIZE = 30


def load_tasks(task_dir: str | Path) -> dict[str, dict]:
    """Load ARC tasks from a directory of JSON files.

    Each file: {"train": [...], "test": [...]}
    """
    task_dir = Path(task_dir)
    tasks = {}
    for path in sorted(task_dir.glob("*.json")):
        with open(path) as f:
            task = json.load(f)
        tasks[path.stem] = task
    return tasks


def format_submission(results: dict[str, dict]) -> dict:
    """Format solve results into Kaggle submission format.

    Kaggle expects: {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
    """
    submission = {}
    for task_id, result in results.items():
        submission[task_id] = [
            {
                "attempt_1": pred["attempt_1"],
                "attempt_2": pred["attempt_2"],
            }
            for pred in result["predictions"]
        ]
    return submission


def write_submission(
    results: dict[str, dict],
    output_path: str | Path = "submission.json",
) -> Path:
    """Write submission.json."""
    submission = format_submission(results)
    path = Path(output_path)
    with open(path, "w") as f:
        json.dump(submission, f)
    logger.info(f"Wrote {len(submission)} tasks to {path}")
    return path


def main(
    task_dir: str | Path = "/kaggle/input/arc-prize-2026-arc-agi-2/challenges",
    output_path: str | Path = "submission.json",
    time_budget: float = TOTAL_TIME_BUDGET,
) -> Path:
    """Full pipeline: load → solve → write.

    Args:
        task_dir: directory of challenge JSON files
        output_path: where to write submission.json
        time_budget: total wall-clock seconds

    Returns:
        Path to submission.json
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    t0 = time.monotonic()

    logger.info(f"Loading tasks from {task_dir}")
    tasks = load_tasks(task_dir)
    logger.info(f"Loaded {len(tasks)} tasks")

    logger.info("Building proposers")
    proposers = default_proposers()
    logger.info(f"Using {len(proposers)} proposers")

    logger.info(f"Solving with {time_budget:.0f}s budget")
    results = solve_all(tasks, proposers, total_time_seconds=time_budget)

    solved = sum(1 for r in results.values() if r["solved"])
    logger.info(f"Solved {solved}/{len(results)} tasks")

    path = write_submission(results, output_path)

    elapsed = time.monotonic() - t0
    logger.info(f"Done in {elapsed:.1f}s")

    return path


if __name__ == "__main__":
    main()
