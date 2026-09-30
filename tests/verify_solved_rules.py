"""Verify Dorian's solved rules as parameterized programs on real training pairs.

Owner: Dorian

Usage:
    python tests/verify_solved_rules.py <data_dir>

Blind discipline: TRAIN pairs only — no test input/output is ever read or
written here. Rules were derived from training examples during the family
blind rounds; this script re-checks them as executable programs against the
real training data. The data directory is passed in; no dataset lives in
the repo.

Exit code 0: every rule verified exact on all its training pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Rule programs
# ---------------------------------------------------------------------------

def _nonzero_components(grid: np.ndarray) -> list[tuple[int, list, list]]:
    """4-connected nonzero components: (color, cells, bbox[r0,c0,r1,c1])."""
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [x[0] for x in cells]
            cs = [x[1] for x in cells]
            comps.append((color, cells, [min(rs), min(cs), max(rs), max(cs)]))
    return comps


def rule_48131b3c(pairs):
    """Inversion (0<->color) then 2x2 block tiling. The color is inferred
    PER GRID: each input carries its own single nonzero color."""

    def transform(grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=int)
        nonzero = {int(x) for x in np.unique(g) if x != 0}
        if len(nonzero) != 1:
            raise ValueError(f"expected exactly 1 nonzero color, found {sorted(nonzero)}")
        color = nonzero.pop()
        inverted = np.where(g == 0, color, 0)
        return np.tile(inverted, (2, 2))

    return transform, "per-grid inversion (0<->color), tile 2x2"


def rule_1c0d0a4b(pairs):
    """Band-complement inversion: within the band of input rows/cols that
    contain any nonzero cell, 0 -> 2 and 8 -> 0; separators stay 0."""

    def transform(grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=int)
        rows = [r for r in range(g.shape[0]) if (g[r] != 0).any()]
        cols = [c for c in range(g.shape[1]) if (g[:, c] != 0).any()]
        if not rows or not cols:
            return g.copy()
        out = g.copy()
        band = g[np.ix_(rows, cols)]
        out[np.ix_(rows, cols)] = np.where(band == 0, 2, np.where(band == 8, 0, band))
        return out

    return transform, "band-complement inversion (0->2, 8->0 inside band)"


def rule_e9b4f6fc(pairs):
    """Largest nonzero component is the target; horizontal adjacent color
    pairs OUTSIDE it define recolor rules (right -> left); output is the
    target's bbox crop with the rules applied inside."""

    def transform(grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=int)
        comps = _nonzero_components(g)
        if not comps:
            return g.copy()
        target = max(comps, key=lambda comp: len(comp[1]))
        r0, c0, r1, c1 = target[2]
        mapping: dict[int, int] = {}
        h, w = g.shape
        for r in range(h):
            for c in range(w - 1):
                if g[r, c] == 0 or g[r, c + 1] == 0:
                    continue
                # pairs fully INSIDE the target's bbox are content, not rules
                if r0 <= r <= r1 and c0 <= c and c + 1 <= c1:
                    continue
                left, right = int(g[r, c]), int(g[r, c + 1])
                if left != right and right not in mapping:
                    mapping[right] = left
        crop = g[r0:r1 + 1, c0:c1 + 1].copy()
        for source, dest in mapping.items():
            crop[crop == source] = dest
        return crop

    return transform, "largest-component crop + outside-pair color rules"


def rule_8ba14f53(pairs):
    """Frame-cavity summary: enclosed 0-regions attribute their cell count
    to the bordering frame color; colors in first-appearance order; each
    color starts a NEW row in the 3x3 output, left-aligned."""

    def transform(grid: np.ndarray) -> np.ndarray:
        g = np.asarray(grid, dtype=int)
        h, w = g.shape
        first_seen: list[int] = []
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v != 0 and v not in first_seen:
                    first_seen.append(v)
        seen = np.zeros_like(g, dtype=bool)
        attrib: dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                if g[r, c] != 0 or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                touches_border = False
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    if cr in (0, h - 1) or cc in (0, w - 1):
                        touches_border = True
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == 0:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                if touches_border:
                    continue
                neighbors: set[int] = set()
                for (cr, cc) in cells:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != 0:
                            neighbors.add(int(g[nr, nc]))
                if len(neighbors) == 1:
                    color = neighbors.pop()
                    attrib[color] = attrib.get(color, 0) + len(cells)
        out = np.zeros((3, 3), dtype=int)
        row = 0
        for color in first_seen:
            remaining = attrib.get(color, 0)
            # each color starts a NEW row; if its count exceeds 3, it
            # spills into following rows
            while remaining > 0 and row < 3:
                fill = min(remaining, 3)
                out[row, :fill] = color
                remaining -= fill
                row += 1
        return out

    return transform, "frame-cavity summary, per-color new row"


RULES: dict[str, tuple[str, object]] = {
    "48131b3c": ("inversion + 2x2 block tile", rule_48131b3c),
    "8ba14f53": ("frame-cavity summary", rule_8ba14f53),
    "e9b4f6fc": ("outside pairs + largest component crop", rule_e9b4f6fc),
    "1c0d0a4b": ("band-complement inversion", rule_1c0d0a4b),
}


# ---------------------------------------------------------------------------
# Verification driver
# ---------------------------------------------------------------------------

def verify_task(task_id: str, rule_fn, data_dir: Path) -> tuple[bool, str]:
    path = data_dir / f"{task_id}.json"
    if not path.exists():
        return False, f"task file missing: {path}"
    task = json.loads(path.read_text(encoding="utf-8"))
    pairs = task["train"]
    transform, params = rule_fn(pairs)
    if transform is None:
        return False, f"parameter inference failed: {params}"
    failures = []
    for i, pair in enumerate(pairs):
        inp = np.asarray(pair["input"], dtype=int)
        expected = np.asarray(pair["output"], dtype=int)
        try:
            result = transform(inp)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"train{i}: raised {exc}")
            continue
        if result.shape != expected.shape or not np.array_equal(result, expected):
            failures.append(f"train{i}: mismatch")
    if failures:
        return False, f"{params} -> {len(failures)}/{len(pairs)} failed ({failures[:3]})"
    return True, f"{params} -> {len(pairs)}/{len(pairs)} exact"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="directory with one task JSON per file")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("blind discipline: train pairs only")
    all_ok = True
    for task_id, (label, rule_fn) in RULES.items():
        ok, detail = verify_task(task_id, rule_fn, data_dir)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {task_id}  {label}: {detail}")
        all_ok = all_ok and ok
    print("ALL RULES VERIFIED" if all_ok else "SOME RULES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
