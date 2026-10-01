"""Train-only acceptance check for contextual background fills.

Usage:
    python tests/verify_fill_from_bg.py <data_dir>

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

from arc2.proposers import FillFromBackgroundProposer  # noqa: E402
from arc2.scene import Scene  # noqa: E402


CASES = ("0b17323b", "bb52a14b", "11852cab", "27a77e38", "72322fa7")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    all_ok = True
    for task_id in CASES:
        task = json.loads((data_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        inputs = [Scene(pair["input"]) for pair in task["train"]]
        outputs = [Scene(pair["output"]) for pair in task["train"]]
        proposals = FillFromBackgroundProposer().propose(inputs, outputs)
        exact = [
            proposal
            for proposal in proposals
            if all(
                np.array_equal(proposal.transform(source.grid), target.grid)
                for source, target in zip(inputs, outputs)
            )
        ]
        ok = bool(exact)
        print(
            f"{'PASS' if ok else 'FAIL'}  {task_id}: "
            f"{len(inputs)}/{len(inputs)} train exact; {len(exact)} proposal(s)"
        )
        all_ok &= ok
    print("ALL FILL-FROM-BACKGROUND REPAIRS VERIFIED" if all_ok else "FILL VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
