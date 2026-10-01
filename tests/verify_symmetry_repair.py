"""Train-only acceptance check for sparse repair rule families.

Usage:
    python tests/verify_symmetry_repair.py <data_dir>

Only ``train`` pairs are loaded.  Each target must have an exact proposal from
its task-agnostic family; test outputs are never read here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc2.proposers import (  # noqa: E402
    ShapeUnifierProposer,
    SymmetryCompleterProposer,
    WallpaperRepairProposer,
)
from arc2.scene import Scene  # noqa: E402


CASES = {
    "17b80ad2": WallpaperRepairProposer,
    "1d0a4b61": WallpaperRepairProposer,
    "2546ccf6": SymmetryCompleterProposer,
    "2a5f8217": ShapeUnifierProposer,
    "30f42897": SymmetryCompleterProposer,
}


def _load_train(data_dir: Path, task_id: str) -> tuple[list[Scene], list[Scene]]:
    task = json.loads((data_dir / f"{task_id}.json").read_text(encoding="utf-8"))
    pairs = task["train"]
    return (
        [Scene(pair["input"]) for pair in pairs],
        [Scene(pair["output"]) for pair in pairs],
    )


def _exact(proposal, inputs: list[Scene], outputs: list[Scene]) -> bool:
    return all(
        np.array_equal(proposal.transform(source.grid), target.grid)
        for source, target in zip(inputs, outputs)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    all_ok = True
    for task_id, proposer_type in CASES.items():
        inputs, outputs = _load_train(data_dir, task_id)
        proposals = proposer_type().propose(inputs, outputs)
        ok = any(_exact(proposal, inputs, outputs) for proposal in proposals)
        print(
            f"{'PASS' if ok else 'FAIL'}  {task_id}: "
            f"{len(inputs)}/{len(inputs)} train exact; {len(proposals)} proposal(s)"
        )
        all_ok &= ok
    print("ALL SPARSE REPAIRS VERIFIED" if all_ok else "SPARSE REPAIR VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
