"""End-to-end pipeline benchmark on the ARC-AGI-2 training set.

Owner: Dorian (integration test)

Usage:
    python tests/bench_pipeline.py <data_dir> [--limit N] [--total-seconds S]

Reads task JSONs (``train``/``test``) from ``<data_dir>``, runs the full
pipeline (Scene -> Proposers -> Search -> Verify -> Solve) and reports the
baseline: solved fraction, prediction-contract violations, and timing.

This script contains no dataset; the data directory is passed in.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc2.proposers import default_proposers  # noqa: E402
from arc2.solve import solve_all  # noqa: E402
from arc2.verify import validate_prediction  # noqa: E402


def load_tasks(data_dir: str, limit: int | None) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    files = sorted(Path(data_dir).glob("*.json"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        tasks[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory with one task JSON per file")
    parser.add_argument("--limit", type=int, default=None, help="only first N tasks")
    parser.add_argument("--total-seconds", type=float, default=3600.0,
                        help="shared time budget for solve_all")
    args = parser.parse_args()

    tasks = load_tasks(args.data_dir, args.limit)
    if not tasks:
        print("no tasks found")
        return 1
    print(f"tasks loaded: {len(tasks)}")

    t0 = time.monotonic()
    results = solve_all(tasks, default_proposers(), total_time_seconds=args.total_seconds)
    elapsed = time.monotonic() - t0

    solved = sum(1 for r in results.values() if r["solved"])
    predictions_total = 0
    violations: list[tuple[str, str]] = []
    for task_id, result in results.items():
        for pred in result["predictions"]:
            for attempt in ("attempt_1", "attempt_2"):
                predictions_total += 1
                ok, reason = validate_prediction(pred[attempt])
                if not ok:
                    violations.append((task_id, reason))

    print(f"solved: {solved}/{len(tasks)} ({100.0 * solved / len(tasks):.1f}%)")
    print(f"predictions: {predictions_total}, contract violations: {len(violations)}")
    for task_id, reason in violations[:10]:
        print(f"  VIOLATION {task_id}: {reason}")
    print(f"elapsed: {elapsed:.1f}s")

    # fail-closed guarantee: the writer must never emit a malformed prediction
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
