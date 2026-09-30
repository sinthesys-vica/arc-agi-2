"""Train-only acceptance check for the grid-partitioning family.

Usage:
    python tests/verify_grid_partitioning.py <data_dir>

The task JSONs are external to this repository.  Only ``train`` pairs are
loaded.  Exit code 0 means all four partition tasks have an exact proposal,
the background-gap counterexample is rejected by every partition proposer,
and the separate framed-object family solves it exactly.
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
    FramedObjectProposer,
    PartitionAnchorRelocateProposer,
    PartitionBlueprintProposer,
    PartitionMarkerRouteProposer,
    PartitionMaxCountProposer,
)
from arc2.scene import Scene  # noqa: E402


PARTITION_CASES = {
    "09629e4f": PartitionBlueprintProposer,
    "29623171": PartitionMaxCountProposer,
    "6d0160f0": PartitionMarkerRouteProposer,
    "dc46ea44": PartitionAnchorRelocateProposer,
}


def _load_train(data_dir: Path, task_id: str) -> tuple[list[Scene], list[Scene]]:
    task = json.loads((data_dir / f"{task_id}.json").read_text(encoding="utf-8"))
    pairs = task["train"]
    return (
        [Scene(pair["input"]) for pair in pairs],
        [Scene(pair["output"]) for pair in pairs],
    )


def _all_exact(proposals, inputs: list[Scene], outputs: list[Scene]) -> bool:
    return any(
        all(
            np.array_equal(proposal.transform(source.grid), target.grid)
            for source, target in zip(inputs, outputs)
        )
        for proposal in proposals
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    all_ok = True
    for task_id, proposer_type in PARTITION_CASES.items():
        inputs, outputs = _load_train(data_dir, task_id)
        proposals = proposer_type().propose(inputs, outputs)
        ok = _all_exact(proposals, inputs, outputs)
        print(
            f"{'PASS' if ok else 'FAIL'}  {task_id}: "
            f"{len(inputs)}/{len(inputs)} train exact; {len(proposals)} proposal(s)"
        )
        all_ok &= ok

    reject_inputs, reject_outputs = _load_train(data_dir, "543a7ed5")
    partition_proposers = [
        PartitionBlueprintProposer(),
        PartitionMaxCountProposer(),
        PartitionMarkerRouteProposer(),
        PartitionAnchorRelocateProposer(),
    ]
    partition_hits = sum(
        len(proposer.propose(reject_inputs, reject_outputs))
        for proposer in partition_proposers
    )
    rejected = partition_hits == 0
    print(
        f"{'PASS' if rejected else 'FAIL'}  543a7ed5: "
        f"partition family emitted {partition_hits} verified proposal(s)"
    )
    all_ok &= rejected

    framed = FramedObjectProposer().propose(reject_inputs, reject_outputs)
    framed_ok = _all_exact(framed, reject_inputs, reject_outputs)
    print(
        f"{'PASS' if framed_ok else 'FAIL'}  543a7ed5: "
        f"framed-object family {len(reject_inputs)}/{len(reject_inputs)} train exact"
    )
    all_ok &= framed_ok

    print("ALL GRID RULES VERIFIED" if all_ok else "GRID RULE VERIFICATION FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
