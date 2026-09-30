"""Measure the program-space statistics behind structural_distance.

Owner: Dorian (threshold tuning data)

Usage:
    python tests/measure_signature_space.py <data_dir> [--limit N]

For every training task: generate candidates from the default proposers
(mirroring search's paired flow), verify each, and collect the valid
hypotheses. Then measure, over all tasks with 2+ valid hypotheses:

* pairwise structural_distance distribution among verified programs
* how many tasks get a structurally different alt under candidate
  thresholds (the select_alt decision surface)
* the most frequent program signatures (which families actually verify)

This script contains no dataset; the data directory is passed in.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from arc2.proposers import default_proposers, Hypothesis  # noqa: E402
from arc2.scene import Scene  # noqa: E402
from arc2.verify import verify_hypothesis, structural_distance  # noqa: E402


def load_tasks(data_dir: str, limit: int | None) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    files = sorted(Path(data_dir).glob("*.json"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        tasks[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return tasks


def collect_valid(task: dict) -> list[Hypothesis]:
    """Generate + verify candidates for one task; return the valid list."""
    train = task["train"]
    inputs = [Scene.from_list(p["input"]) for p in train]
    outputs = [Scene.from_list(p["output"]) for p in train]
    valid: list[Hypothesis] = []
    for proposer in default_proposers():
        try:
            candidates = proposer.propose(inputs, outputs=outputs) or []
        except TypeError:
            candidates = proposer.propose(inputs) or []
        except Exception:
            candidates = []
        for hypothesis in candidates:
            if verify_hypothesis(hypothesis, train):
                valid.append(hypothesis)
    return valid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = load_tasks(args.data_dir, args.limit)
    print(f"tasks loaded: {len(tasks)}")

    signature_counts: Counter[tuple] = Counter()
    tasks_with_alt_candidates = 0
    pairwise: list[float] = []
    per_task_max: list[float] = []
    per_task_min_nonzero: list[float] = []

    for task_id, task in tasks.items():
        valid = collect_valid(task)
        seen = set()
        for h in valid:
            key = h.program_signature
            if key not in seen:
                seen.add(key)
                signature_counts[key] += 1

        if len(valid) >= 2:
            tasks_with_alt_candidates += 1
            distances = [
                structural_distance(a, b)
                for i, a in enumerate(valid)
                for b in valid[i + 1:]
            ]
            pairwise.extend(distances)
            per_task_max.append(max(distances))
            per_task_min_nonzero.append(min(d for d in distances if d > 0))

    print(f"tasks with 2+ verified hypotheses: {tasks_with_alt_candidates}")
    print(f"pairwise distances measured: {len(pairwise)}")

    if pairwise:
        arr = np.array(pairwise)
        print("pairwise structural_distance distribution:")
        print(f"  min={arr.min():.3f} p25={np.percentile(arr, 25):.3f} "
              f"median={np.median(arr):.3f} p75={np.percentile(arr, 75):.3f} "
              f"max={arr.max():.3f}")
        hist, edges = np.histogram(arr, bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01])
        for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
            print(f"  [{lo:.1f},{hi:.1f}): {count}")

        print("select_alt decision surface (tasks where an alt EXISTS "
              "under threshold):")
        for threshold in (0.2, 0.3, 0.4, 0.5):
            with_alt = sum(1 for m in per_task_max if m >= threshold)
            print(f"  threshold {threshold}: {with_alt}/{tasks_with_alt_candidates} tasks")

        if per_task_min_nonzero:
            print(f"  smallest nonzero distance ever seen: "
                  f"{min(per_task_min_nonzero):.3f}")

    print("top verified program signatures:")
    for signature, count in signature_counts.most_common(12):
        print(f"  {count:4d}  {signature}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
