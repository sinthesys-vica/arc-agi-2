"""Round 10 verification — MiscTransformProposer acceptance tests.

Owner: Raiden (bench), Dorian (verification)
Tests that the new MiscTransformProposer solves the 5 acceptance tasks
on training pairs (blind — no test output used).
"""

import numpy as np
import pytest

from arc2.submission import load_tasks
from arc2.proposers import default_proposers
from arc2.verify import verify_hypothesis


TASK_DIR = "data/arc-agi-2/data/training/"

ACCEPTANCE_TASKS = [
    "1e0a9b12",  # gravity down
    "3906de3d",  # gravity up
    "22168020",  # connect H
    "ded97339",  # connect H+V
    "e9afcf9a",  # diagonal column shift
]


@pytest.fixture(scope="module")
def tasks():
    return load_tasks(TASK_DIR)


@pytest.fixture(scope="module")
def proposers():
    return default_proposers()


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
        f"MiscTransformProposer failed to solve acceptance task {task_id}"
    )


def test_no_regression(tasks, proposers):
    """Benchmark must not regress: >=77 solved, 0 contract violations."""
    from arc2.solve import solve_all
    from arc2.verify import validate_prediction

    results = solve_all(tasks, proposers, total_time_seconds=600)
    solved = sum(1 for r in results.values() if r["solved"])
    violations = 0
    for r in results.values():
        for pred in r["predictions"]:
            for attempt in ("attempt_1", "attempt_2"):
                ok, _ = validate_prediction(pred[attempt])
                if not ok:
                    violations += 1

    assert solved >= 77, f"Regression: {solved} < 77 solved"
    assert violations == 0, f"Contract violations: {violations}"


def test_at_least_4_of_5(tasks, proposers):
    """Must solve at least 4 of 5 acceptance tasks."""
    solved = 0
    for task_id in ACCEPTANCE_TASKS:
        task = tasks[task_id]
        if _solve_task(proposers, task):
            solved += 1
    assert solved >= 4, f"Only {solved}/5 acceptance tasks solved (need >=4)"
