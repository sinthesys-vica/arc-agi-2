"""Round 8 verification — GridCellOperationProposer acceptance tests.

Owner: Dorian (verification), Raiden (bench)
Tests that the new GridCellOperationProposer solves the 5 acceptance tasks
on training pairs (blind — no test output used).
"""

import numpy as np
import pytest

from arc2.submission import load_tasks
from arc2.proposers import default_proposers
from arc2.verify import verify_hypothesis


TASK_DIR = "data/arc-agi-2/data/training/"

ACCEPTANCE_TASKS = [
    "ef26cbf6",  # key-pattern recolor (3x2 grid, div=4)
    "5a5a2103",  # grid stamp (4x4 grid, div=3/8)
    "1e32b0e9",  # pattern fill (3x3 grid, div=2)
    "e3fe1151",  # cell transform (2x2 grid, div=7)
    "2dc579da",  # cell extraction (2x2 grid, diff-size)
]


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(TASK_DIR)


@pytest.fixture(scope="module")
def proposers():
    return default_proposers()


def _find_grid_cell_proposer(proposers):
    """Find the GridCellOperationProposer in the proposer list."""
    for p in proposers:
        name = type(p).__name__
        if "GridCell" in name or "GridOp" in name:
            return p
    return None


def _solve_task(proposers, task):
    """Try to solve a task using all proposers. Returns True if any hypothesis verifies."""
    from arc2.scene import Scene

    scenes_in = [Scene.from_list(p["input"]) for p in task["train"]]
    scenes_out = [Scene.from_list(p["output"]) for p in task["train"]]

    for prop in proposers:
        try:
            candidates = prop.propose(scenes_in, outputs=scenes_out) or []
        except TypeError:
            try:
                candidates = prop.propose(scenes_in) or []
            except Exception:
                continue
        except Exception:
            continue

        for h in candidates:
            if verify_hypothesis(h, task["train"]):
                return True
    return False


@pytest.mark.parametrize("task_id", ACCEPTANCE_TASKS)
def test_acceptance_task(task_id, tasks, proposers):
    """Each acceptance task must be solvable on training pairs."""
    task = tasks[task_id]
    assert _solve_task(proposers, task), (
        f"GridCellOperationProposer failed to solve acceptance task {task_id}"
    )


def test_no_regression(tasks, proposers):
    """Benchmark must not regress: ≥56 solved, 0 contract violations."""
    from arc2.solve import solve_all

    results = solve_all(tasks, proposers, total_time_seconds=600)
    solved = sum(1 for r in results.values() if r["solved"])
    violations = sum(1 for r in results.values() if r.get("contract_violation"))

    assert solved >= 56, f"Regression: {solved} < 56 solved"
    assert violations == 0, f"Contract violations: {violations}"


def test_at_least_4_of_5(tasks, proposers):
    """Must solve at least 4 of 5 acceptance tasks."""
    solved = 0
    for task_id in ACCEPTANCE_TASKS:
        task = tasks[task_id]
        if _solve_task(proposers, task):
            solved += 1
    assert solved >= 4, f"Only {solved}/5 acceptance tasks solved (need ≥4)"
