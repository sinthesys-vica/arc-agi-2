# Grid-partitioning proposer — acceptance & counterexample matrix

Owner: Dorian (train-only, blind discipline — this spec contains NO task data,
only structure observations and synthetic counterexamples)

## Family signature (what the five targets have in common)

1. **Uniform separator lines** — full rows and/or columns of a single color
   split the grid into rectangular subgrids. Separators are preserved
   exactly in the output.
2. **Same-size IO** — output has the input's dimensions.
3. **Per-subgrid local rule** — each subgrid's output content is decided
   locally (summary, count-threshold, relocation), not by global shape.

## Target tasks (Raiden's selection)

| task | sep structure | observed subgrid rule (train0) | acceptance |
|------|---------------|--------------------------------|------------|
| 09629e4f | rows 3,7 + cols 3,7 (color 5), 3×3 subgrids | each subgrid is a permutation of one shared color set; the output color follows a position/permutation logic (needs multi-pair inference) | 4/4 train exact |
| 29623171 | rows 3,7 + cols 3,7 (color 5) | subgrids containing ≥2 cells of the target color fill solid with that color; others become background | 3/3 train exact |
| 6d0160f0 | rows 3,7 + cols 3,7 (color 5) | one subgrid's content is relocated/rotated; the rest are cleared | 4/4 train exact |
| dc46ea44 | single separator row (color 4) | shapes below the separator relocate above it | 3/3 train exact |
| 543a7ed5 | **NO separator** | frame-outline + interior-fill (6-frames gain 3-outlines, enclosed 8s become 4s) | **proposer must REJECT** |

## Counterexamples (synthetic breakers — naive implementations fail these)

1. **Separator color == content color.** A grid whose "separator" rows share
   their color with subgrid content is not a partition — must fail closed.
2. **Broken separator.** A line that stops being uniform for a single cell
   is not a separator; the whole grid must be rejected.
3. **Unequal subgrid sizes.** Non-uniform splits (e.g. rows 3,6 instead of
   3,7) must not be silently tiled.
4. **Empty subgrid.** All-background subgrids are valid (→ background), must
   not crash and must not be treated as "no rule".
5. **Multi-object subgrid.** Subgrids with several colors/objects must use
   deterministic summaries — no first-object-wins ambiguity.
6. **No separator at all (543a7ed5 class).** Frame/outline tasks must not be
   misfired on; the proposer emits nothing (or nothing that verifies).
7. **Degenerate subgrids.** 1×N and N×1 subgrids after a split.
8. **Double separator lines.** Two adjacent separator rows (width-2 border).

## Acceptance conditions (executable when the proposer lands)

- Train-exactness on the four separator tasks:
  09629e4f 4/4, 29623171 3/3, 6d0160f0 4/4, dc46ea44 3/3.
- Rejection on 543a7ed5: no hypothesis emitted or none verified on it.
- No regression: the full-training bench stays at ≥23/1000 with
  0 contract violations (`tests/bench_pipeline.py`).
- Task-agnostic: no task ids or task data inside the proposer code.
- Fail-closed: every exception and malformed candidate degrades to
  "not proposed", never crashes the search loop.

## Verification workflow

`tests/verify_solved_rules.py` shows the house pattern: parameterized rule
programs checked exact against real training pairs, data dir passed in.
The partitioning proposer's train-exactness will be checked the same way;
the rejection case (543a7ed5) is asserted as "no verified hypothesis".
