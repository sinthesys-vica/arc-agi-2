# Symmetry-partial proposer — acceptance & counterexample matrix

Owner: Dorian (train-only, blind discipline — structure observations only,
no test data; synthetic counterexamples are constructed, not dataset-derived)

## Family signature (shared skeleton)

1. **Same-size IO** — the repair never resizes the grid.
2. **Tiny delta** — change ratios 0.005–0.021: single-digit cell repairs.
3. **Mirror-axis driven** — the bulk content determines a dominant axis
   (H / V / D4); the minority asymmetric cells are the repair targets.
4. **Everything else preserved** — untouched cells are copied exactly.
5. **Non-zero backgrounds are the norm here** — 52df9849: 7, 8dab14c2: 8,
   9f8de559: 7-field with 5/8 borders — background inference is mandatory,
   never hardcode 0.

## Per-target observations (train0, honest limits)

| task | axis | observation | open question |
|------|------|-------------|---------------|
| 52df9849 | H | two rectangles; the 4-block extends into the 1-block's columns | exact extension rule needs pair 2 |
| 8dab14c2 | V | scattered 1-cells move to their V-mirror positions (e.g. (2,5) clears, (10,10) fills) | move-vs-copy semantics |
| 9f8de559 | D4 | ONE cell changes in train0 ((13,9) 8->7 under the 6-column) | single-anomaly mechanism |
| 1e81d6f9 | D4 | dots around a partial 5-cross | mirror-completion vs relocation |
| 3d588dc9 | D4 | (2 pairs) | needs full read |

Honest note: train0 alone does not pin three of the five mechanisms — the
acceptance remains train-exactness across ALL pairs; the matrix below is
what the shared detector must guarantee regardless of mechanism.

## Risk points

1. **Axis ambiguity.** A grid symmetric (or near-symmetric) on TWO axes
   must not pick silently; fail closed or require the tie to break by
   evidence.
2. **Mirror conflicts.** A cell whose mirror target is already occupied by
   a DIFFERENT color is a conflict — no silent overwrite.
3. **Multi-cluster anomalies.** Several independent repairs in one grid
   must all be found; a single-repair proposer must reject the task.
4. **Border anomalies.** An asymmetric cell ON the border (e.g. 8dab14c2
   row4 col14) — mirror handling must clip and stay in-grid.
5. **Already-symmetric grids.** Delta 0 must not be "repaired".
6. **Background inference.** All five targets have non-zero backgrounds;
   the axis detection must be background-aware.

## Synthetic counterexamples (must NOT misfire / must fail closed)

1. Two-axis symmetric grid + one off-axis dot -> reject (ambiguous axis).
2. Mirror target occupied by a different color -> reject, no overwrite.
3. Two separate anomaly clusters -> reject unless both are found.
4. Anomaly on the border -> clip, no out-of-bounds.
5. Fully symmetric grid -> no-op, no repair.
6. Non-zero background with one anomaly -> repair in the task's own
   background.
7. 1xN / Nx1 degenerate grid -> axis undefined, reject.
8. Random noise (no dominant axis) -> reject.

## Acceptance conditions (executable when the proposer lands)

- Train-exactness: 52df9849 2/2, 8dab14c2 4/4, 9f8de559 4/4, 1e81d6f9 3/3,
  3d588dc9 2/2.
- No regression: full-training bench stays at >= 40/1000 with 0 contract
  violations, run with a KAGGLE-LIKE per-task budget (~36s/task = 10h/1000,
  the wallpaper lesson) — NOT a tight total budget.
- Task-agnostic: no task ids or dataset content in the proposer code.
- Fail-closed: exceptions and malformed candidates degrade to
  "not proposed", never crash the search loop.
