"""Train-only acceptance check for grid-cell operations.

Usage:
    python tests/verify_grid_cell_ops_train.py <data_dir>

Only training pairs are loaded. Test outputs are never read here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc2.proposers import GridCellOperationProposer  # noqa: E402
from arc2.scene import Scene  # noqa: E402


CASES = ("ef26cbf6", "5a5a2103", "1e32b0e9", "e3fe1151", "2dc579da")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    solved = 0
    for task_id in CASES:
        task = json.loads((data_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        inputs = [Scene(pair["input"]) for pair in task["train"]]
        outputs = [Scene(pair["output"]) for pair in task["train"]]
        proposals = GridCellOperationProposer().propose(inputs, outputs)
        exact = [
            proposal
            for proposal in proposals
            if all(
                np.array_equal(proposal.transform(source.grid), target.grid)
                for source, target in zip(inputs, outputs)
            )
        ]
        ok = bool(exact)
        solved += int(ok)
        print(
            f"{'PASS' if ok else 'FAIL'}  {task_id}: "
            f"{len(inputs)}/{len(inputs)} train exact; {len(exact)} proposal(s)"
        )
    print(f"GRID-CELL ACCEPTANCE: {solved}/{len(CASES)}")
    return 0 if solved >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
