# Symmetry-repair proposer — acceptance & counterexample matrix

Owner: Dorian (train-only, blind discipline — structure observations only,
no test data; synthetic counterexamples are constructed, not dataset-derived)

## Honest family verdict first

The "symmetry_repair" label covers THREE different repair mechanisms:

| task | observed mechanism (train0) | literal symmetry? |
|------|------------------------------|-------------------|
| 17b80ad2 | marker columns divide into bands; each marker fills the band from the previous marker down to itself | NO — band logic, not mirror symmetry |
| 1d0a4b61 | periodic wallpaper (7x7 period, symmetric motif); 0-holes repaired by pattern continuation | yes (the wallpaper motif itself is symmetric) |
| 2546ccf6 | damaged/empty 4x4 block repaired from the D4-symmetric block tiling | yes (block tiling symmetric) |
| 2a5f8217 | duplicate SHAPES unify their color: a shape matching another shape's pattern gets recolored | NO — shape-class unification |
| 30f42897 | the output completes a grid symmetry; the line segment is copied to mirror positions | yes (output becomes vertically symmetric) |

Shared skeleton: (a) same-size IO; (b) only damaged/duplicate elements change,
unrelated content preserved; (c) background color is task-specific (0 in
four tasks, 8 in 30f42897) — never hardcode 0.

## Risk points

1. **Repair ambiguity.** A hole compatible with two candidate patterns/
   symmetries must fail closed, not pick silently.
2. **Multi-axis symmetry.** A grid already symmetric on both axes must not
   be "repaired" into double-fill.
3. **Existing-cell respect.** Repair fills must not overwrite non-background
   cells (except the explicit recolor rules of the 2a5f8217 class).
4. **Background generalization.** 30f42897 uses 8 as background; the
   detector must infer the background, not assume 0.
5. **Marker/edge placement.** A marker or damaged region touching the grid
   border (17b80ad2's markers are interior; 30f42897's line touches the
   right edge) — boundary handling must be per-rule, never out-of-bounds.

## Synthetic counterexamples (must NOT misfire / must fail closed)

1. Ambiguous hole: a 0-run compatible with two wallpaper periods -> reject.
2. Already-symmetric grid with an unrelated dot -> dot preserved, nothing
   "repaired" silently.
3. Damage containing partial pattern cells -> fill must respect them.
4. Background != 0 (e.g. 8) with a hole -> repair in the task's own
   background, no 0-assumption.
5. Line segment at a corner -> mirror/copy must clip to the grid.
6. Three identical shapes (multi-twin) -> deterministic unification.
7. Uniform wallpaper (period 1) -> degenerate, reject.
8. No pattern at all (random noise) -> reject.

## Acceptance conditions (executable when the proposer lands)

- Train-exactness: 17b80ad2 4/4, 1d0a4b61 3/3, 2546ccf6 2/2, 2a5f8217 3/3,
  30f42897 3/3.
- No regression: full-training bench stays at >= 34/1000 with 0 contract
  violations (`tests/bench_pipeline.py`).
- Task-agnostic: no task ids or dataset content in the proposer code.
- Fail-closed: exceptions and malformed candidates degrade to
  "not proposed", never crash the search loop.
- The shared detector: same-size IO + background inference + damaged/
  duplicate identification; per-target rule transforms (house pattern).
