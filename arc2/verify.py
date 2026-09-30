"""Verification — validate hypotheses against training examples.

Owner: Dorian

This module is fail-closed by design: a hypothesis that raises, returns a
malformed grid, or produces anything outside the ARC color contract is
never treated as valid. All functions are deterministic and offline-safe
(no network, no I/O beyond what the caller passes in).
"""

from __future__ import annotations

import numpy as np

from .scene import Scene
from .proposers import Hypothesis
from .dsl import signature as _dsl_signature

MAX_COLOR = 9
MIN_COLOR = 0


# ---------------------------------------------------------------------------
# Low-level safety helpers
# ---------------------------------------------------------------------------

def _safe_apply(hypothesis: Hypothesis, input_grid: np.ndarray) -> np.ndarray | None:
    """Apply a hypothesis defensively; return None on any failure.

    Hypotheses come from search and may be partially constructed. A raised
    exception must degrade to "invalid", never crash the search loop.
    """
    try:
        result = hypothesis.transform(np.asarray(input_grid, dtype=int))
        return np.asarray(result, dtype=int)
    except Exception:
        return None


def validate_prediction(grid: np.ndarray | list) -> tuple[bool, str]:
    """Validate a predicted grid against the ARC output contract.

    Args:
        grid: candidate prediction (ndarray or nested list)

    Returns:
        (ok, reason) — ok is True only if the grid is a non-empty 2D
        rectangular integer array with all values in [0, MAX_COLOR].
    """
    try:
        arr = np.asarray(grid, dtype=int)
    except Exception:
        return False, "not convertible to int array"
    if arr.ndim != 2:
        return False, f"expected 2D grid, got {arr.ndim}D"
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return False, "empty grid"
    if arr.size != np.asarray(grid).size:
        return False, "ragged rows"
    if arr.min() < MIN_COLOR or arr.max() > MAX_COLOR:
        return False, f"color out of range [{MIN_COLOR},{MAX_COLOR}]"
    return True, "ok"


# ---------------------------------------------------------------------------
# Hypothesis verification
# ---------------------------------------------------------------------------

def verify_hypothesis(hypothesis: Hypothesis, train_pairs: list[dict]) -> bool:
    """Check if a hypothesis correctly transforms all training inputs to outputs.

    Fail-closed: exceptions during transform, shape mismatches, and value
    mismatches all count as invalid.

    Args:
        hypothesis: candidate transformation
        train_pairs: list of {"input": grid, "output": grid} dicts

    Returns:
        True if hypothesis produces exact output for all training pairs
    """
    for pair in train_pairs:
        expected = np.asarray(pair["output"], dtype=int)
        result = _safe_apply(hypothesis, np.asarray(pair["input"], dtype=int))
        if result is None:
            return False
        if result.shape != expected.shape or not np.array_equal(result, expected):
            return False
    return True


def score_hypothesis(hypothesis: Hypothesis, train_pairs: list[dict]) -> float:
    """Score a hypothesis: fraction of training pairs correctly transformed."""
    if not train_pairs:
        return 0.0
    correct = 0
    for pair in train_pairs:
        expected = np.asarray(pair["output"], dtype=int)
        result = _safe_apply(hypothesis, np.asarray(pair["input"], dtype=int))
        if result is not None and result.shape == expected.shape and np.array_equal(result, expected):
            correct += 1
    return correct / len(train_pairs)


def structural_distance(h1: Hypothesis, h2: Hypothesis) -> float:
    """Distance between two hypotheses on the program space, in [0, 1].

    The attempt-2 rule needs the *best structurally different* hypothesis,
    not merely one with a different output. Output grids may coincide while
    the programs differ — and the structural diversity is what matters for
    the second attempt. The norm here is program-level:

    * the program norm is PRIMARY: if both transforms carry program keys
      (compose() component signatures, or the transform's own dsl signature),
      the distance is the Jaccard distance over those key sets — semantic,
      not object identity: two separately built instances of the same
      program are distance 0;
    * otherwise (not introspectable), the description tokens serve as the
      fallback norm (Jaccard distance).

    0 = structurally identical, 1 = disjoint. Program identity beats wording:
    two identical programs with differently-worded descriptions are still
    distance 0.
    """
    d_prog = _program_jaccard_distance(h1.transform, h2.transform)
    if d_prog is not None:
        return d_prog
    return _token_jaccard_distance(h1.description, h2.description)


def _token_jaccard_distance(a: str, b: str) -> float:
    ta = set(a.lower().replace("_", " ").replace("-", " ").split())
    tb = set(b.lower().replace("_", " ").replace("-", " ").split())
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def _program_key_set(f) -> set[tuple] | None:
    """The semantic key set of a transform in the program space.

    * compose() closures: the dsl.signature of each component;
    * other transforms carrying a dsl signature: {signature(f)};
    * otherwise None (not introspectable).

    Keys are signatures, never id(): two separately built instances of the
    same program must be distance 0.
    """
    components = _compose_components(f)
    if components is not None:
        return {_dsl_signature(c) for c in components}
    sig = getattr(f, "_arc2_signature", None)
    if sig is not None:
        return {tuple(sig)}
    return None


def _program_jaccard_distance(f1, f2) -> float | None:
    """Jaccard distance over semantic program key sets.

    Returns None when either side is not introspectable — the caller then
    falls back to the description norm.
    """
    s1 = _program_key_set(f1)
    s2 = _program_key_set(f2)
    if s1 is None or s2 is None:
        return None
    if not s1 and not s2:
        return 0.0
    return 1.0 - len(s1 & s2) / len(s1 | s2)


def _compose_components(f) -> tuple | None:
    """Extract component transforms of a dsl.compose() function.

    Prefers the explicit ``_arc2_components`` attribute set by dsl.compose();
    falls back to closure scanning for plain closures built elsewhere.
    """
    explicit = getattr(f, "_arc2_components", None)
    if isinstance(explicit, tuple) and explicit and all(callable(x) for x in explicit):
        return explicit
    closure = getattr(f, "__closure__", None)
    if not closure:
        return None
    for cell in closure:
        value = cell.cell_contents
        if isinstance(value, tuple) and all(callable(x) for x in value):
            return value
    return None


# ---------------------------------------------------------------------------
# Encoder — deterministic feature extraction for proposers
# ---------------------------------------------------------------------------

def _components(grid: np.ndarray) -> list[dict]:
    """4-connected components of non-background cells, largest first.

    Each component: color, size, bbox (r0, c0, r1, c1 inclusive),
    touching_border flag. Deterministic order: by (-size, first cell).
    """
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            color = grid[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
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
            comps.append({
                "color": int(color),
                "size": len(cells),
                "bbox": [min(rs), min(cs), max(rs), max(cs)],
                "touching_border": min(rs) == 0 or min(cs) == 0 or max(rs) == h - 1 or max(cs) == w - 1,
            })
    comps.sort(key=lambda x: (-x["size"], x["bbox"][0], x["bbox"][1]))
    return comps


def encode_grid(grid: np.ndarray | list) -> dict:
    """Deterministic, JSON-serializable feature vector for one grid.

    Proposers consume these features to build candidate hypotheses. The
    encoder is deliberately cheap and rule-agnostic: it measures structure
    (colors, components, symmetries, uniformity, border) without deciding
    what the transformation is.
    """
    arr = np.asarray(grid, dtype=int)
    h, w = arr.shape
    colors = sorted(int(x) for x in np.unique(arr))
    counts = {int(color): int((arr == color).sum()) for color in colors}
    nonzero = arr != 0

    def uniform_lines(axis: int) -> list[int]:
        uniform = []
        for i in range(arr.shape[axis]):
            line = arr.take(i, axis=axis)
            if len(set(int(x) for x in line)) == 1:
                uniform.append(i)
        return uniform

    return {
        "height": h,
        "width": w,
        "area": int(arr.size),
        "colors": colors,
        "color_counts": counts,
        "background": int(counts[max(counts, key=lambda k: counts[k])]) and
                      max(counts, key=lambda k: counts[k]),
        "nonzero_fraction": float(nonzero.mean()),
        "components": _components(arr),
        "n_components": sum(1 for _ in _components(arr)),
        "border_colors": sorted({int(arr[r, c]) for r in range(h) for c in range(w)
                                 if r in (0, h - 1) or c in (0, w - 1)}),
        "uniform_rows": uniform_lines(0),
        "uniform_cols": uniform_lines(1),
        "sym_h": bool(np.array_equal(arr, arr[:, ::-1])),
        "sym_v": bool(np.array_equal(arr, arr[::-1, :])),
        "sym_rot180": bool(np.array_equal(arr, arr[::-1, ::-1])),
    }


def encode_pair(input_grid: np.ndarray | list, output_grid: np.ndarray | list) -> dict:
    """Feature vector for one train pair: both grids plus their relation."""
    inp = np.asarray(input_grid, dtype=int)
    out = np.asarray(output_grid, dtype=int)
    in_colors = set(int(x) for x in np.unique(inp))
    out_colors = set(int(x) for x in np.unique(out))
    size_ratio = (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1]) \
        if inp.shape[0] and inp.shape[1] else None
    return {
        "input": encode_grid(inp),
        "output": encode_grid(out),
        "size_delta": [out.shape[0] - inp.shape[0], out.shape[1] - inp.shape[1]],
        "size_ratio": [round(size_ratio[0], 4), round(size_ratio[1], 4)] if size_ratio else None,
        "colors_only_in": sorted(in_colors - out_colors),
        "colors_only_out": sorted(out_colors - in_colors),
        "colors_shared": sorted(in_colors & out_colors),
        "same_size": inp.shape == out.shape,
    }


# ---------------------------------------------------------------------------
# Writer-with-verify — apply hypothesis to test input, verify before writing
# ---------------------------------------------------------------------------

def apply_with_verify(
    hypothesis: Hypothesis,
    input_grid: np.ndarray | list,
    fallback: str = "input",
) -> tuple[np.ndarray, dict]:
    """Apply a hypothesis to a test input and verify the result immediately.

    Fail-closed writer: nothing malformed ever reaches submission.json.

    Args:
        hypothesis: candidate transformation
        input_grid: test input grid
        fallback: what to return if the hypothesis fails —
            "input" (default): the unchanged input grid

    Returns:
        (grid, report) — grid is the verified prediction (or fallback);
        report holds status and failure reason for logging.
    """
    inp = np.asarray(input_grid, dtype=int)
    result = _safe_apply(hypothesis, inp)
    if result is None:
        return inp.copy(), {"status": "fallback", "reason": "transform raised"}
    ok, reason = validate_prediction(result)
    if not ok:
        return inp.copy(), {"status": "fallback", "reason": reason}
    return result, {"status": "verified", "reason": "ok"}


# ---------------------------------------------------------------------------
# Self-tests (run: python -m arc2.verify)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    from .dsl import identity, compose

    def rot180(g: np.ndarray) -> np.ndarray:
        return g[::-1, ::-1]

    def bad_raise(g: np.ndarray) -> np.ndarray:
        raise RuntimeError("boom")

    def bad_shape(g: np.ndarray) -> np.ndarray:
        return np.zeros((2, 2, 2), dtype=int)

    def bad_color(g: np.ndarray) -> np.ndarray:
        return np.full(g.shape, 42, dtype=int)

    h_good = Hypothesis(identity, "identity transform")
    h_wrong = Hypothesis(rot180, "rotate 180")
    h_raise = Hypothesis(bad_raise, "raising transform")
    h_shape = Hypothesis(bad_shape, "3d output")
    h_color = Hypothesis(bad_color, "out of range colors")

    # Note: inputs are deliberately asymmetric so rot180 actually changes them.
    train = [
        {"input": [[1, 0], [0, 0]], "output": [[1, 0], [0, 0]]},
        {"input": [[0, 2], [0, 0]], "output": [[0, 2], [0, 0]]},
        {"input": [[3, 3], [3, 0]], "output": [[3, 3], [3, 0]]},
    ]

    assert verify_hypothesis(h_good, train) is True, "identity must verify"
    assert verify_hypothesis(h_wrong, train) is False, "rot180 must fail"
    assert verify_hypothesis(h_raise, train) is False, "raising must fail closed"
    assert verify_hypothesis(h_shape, train) is False, "3d output must fail closed"
    assert score_hypothesis(h_good, train) == 1.0, "score 3/3"
    assert abs(score_hypothesis(h_wrong, train) - 0.0) < 1e-9, "score 0/3"

    feats = encode_grid([[1, 0], [0, 1]])
    assert feats["colors"] == [0, 1], feats
    assert feats["n_components"] == 2, feats
    # anti-diagonal: 180-deg symmetric, not mirror symmetric
    assert feats["sym_h"] is False and feats["sym_v"] is False, feats
    assert feats["sym_rot180"] is True, feats

    pair_feats = encode_pair([[1]], [[2, 2]])
    assert pair_feats["size_delta"] == [0, 1], pair_feats
    assert pair_feats["colors_only_in"] == [1], pair_feats

    grid_ok, rep = apply_with_verify(h_good, [[5, 0], [0, 5]])
    assert rep["status"] == "verified", rep
    assert np.array_equal(grid_ok, [[5, 0], [0, 5]])
    grid_fb, rep = apply_with_verify(h_raise, [[5, 0], [0, 5]])
    assert rep["status"] == "fallback" and rep["reason"] == "transform raised", rep
    assert np.array_equal(grid_fb, [[5, 0], [0, 5]]), "fallback must be unchanged input"
    _, rep = apply_with_verify(h_color, [[5, 0], [0, 5]])
    assert rep["status"] == "fallback" and "range" in rep["reason"], rep

    ok, reason = validate_prediction([[1, 2], [3, 4]])
    assert ok, reason
    ok, reason = validate_prediction([1, 2, 3])
    assert not ok and "2D" in reason, reason

    d_same = structural_distance(h_good, Hypothesis(identity, "identity transform"))
    d_diff = structural_distance(h_good, Hypothesis(rot180, "rotate 180 degrees"))
    assert d_same == 0.0, d_same
    assert d_diff > 0.0, d_diff
    d_comp = structural_distance(
        Hypothesis(compose(identity), "a"), Hypothesis(compose(rot180), "b"))
    assert d_comp == 1.0, d_comp
    # same program, differently-worded descriptions -> program norm wins -> 0
    d_same_prog = structural_distance(
        Hypothesis(compose(identity), "identity transform"),
        Hypothesis(compose(identity), "do absolutely nothing at all"))
    assert d_same_prog == 0.0, d_same_prog

    # regression (Atlas catch): separately built instances of the SAME
    # program must be distance 0 — the norm is semantic, not id()-based
    from .dsl import recolor, tile
    def build_same_program():
        return compose(recolor({1: 2}), tile(2, 2))
    d_rebuilt = structural_distance(
        Hypothesis(build_same_program(), "recolor then tile"),
        Hypothesis(build_same_program(), "recolor then tile, other instance"),
    )
    assert d_rebuilt == 0.0, f"same program, separate instances: {d_rebuilt}"
    # positive control: a different mapping in the same shape must stay apart
    d_other_map = structural_distance(
        Hypothesis(compose(recolor({1: 2}), tile(2, 2)), "a"),
        Hypothesis(compose(recolor({1: 3}), tile(2, 2)), "b"),
    )
    assert d_other_map > 0.0, d_other_map

    # single-primitive transforms: their own dsl signature is the program key
    from .dsl import rotate
    d_single_diff = structural_distance(
        Hypothesis(identity, "identity transform"),
        Hypothesis(rotate(2), "rotate 180 degrees"),
    )
    assert d_single_diff == 1.0, d_single_diff
    d_single_same = structural_distance(
        Hypothesis(identity, "identity transform"),
        Hypothesis(identity, "leave everything exactly as it is"),
    )
    assert d_single_same == 0.0, d_single_same

    print("verify.py self-tests: ALL PASS")


if __name__ == "__main__":
    _self_test()
