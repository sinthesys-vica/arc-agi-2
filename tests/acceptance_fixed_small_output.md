# Fixed-small-output proposer — acceptance & counterexample matrix

Owner: Dorian (train-only, blind discipline — structure observations only,
no test data; synthetic counterexamples are constructed, not dataset-derived)

## Family signature (shared skeleton)

1. **Fixed small output** — 1x1 to 3x3, never varies within a task.
2. **Extraction, not transformation** — the output summarizes a PROPERTY of
   the input: classification, position map, count histogram, or pattern
   completion. The input's cells are not moved or recolored into the output.
3. **The property must be pinned by ALL train pairs** — a mechanism that
   fits train0 but not the rest is an overfit trap, the #1 risk here.
4. **Background varies** — 0 (1a2e2828, 794b24be, ff28f65a), 8 (19bb5feb),
   1 (5289ad53). Background inference is mandatory.

## Per-target observations (train0, honest limits)

| task | output | observed mechanism | open question |
|------|--------|--------------------|---------------|
| 1a2e2828 | 1x1 | classification: the grid reduces to a single value (band/bar structure; modal-color hypothesis with tie-break) | exact property needs the other 4 pairs |
| 19bb5feb | 2x2 | quadrant map: each 2x2 object's quadrant becomes its color in a 2x2 mini-map, empty quadrants 0 | grid-to-quadrant split rule |
| 794b24be | 3x3 | pattern completion: the input's partial pattern implies the output shape (10 pairs — strongest signal) | completion rule |
| ff28f65a | 3x3 | object-position encoding: two 2x2 blocks map to two 1-cells (8 pairs) | encoding rule open |
| 5289ad53 | 2x3 | segment-count histogram: 3 has 3 segments -> 3 cells; 2 has 2 segments -> 2 cells; rows by first appearance | tie handling |

## Risk points

1. **Overfit trap.** A property that matches pair 0 but fails the rest —
   the proposer must require ALL pairs, never a single-pair shortcut.
2. **Output shape fixedness.** The output dims are a task constant; a
   mechanism that would emit a different shape must fail closed.
3. **Counting ties.** Equal segment counts / equal-size objects — the
   ordering must be deterministic (first appearance, reading order).
4. **Empty cases.** No objects / all background — the extraction must
   produce the task's empty form, never crash.
5. **Position ties.** Objects in the same quadrant / on quadrant borders —
   the map rule must be deterministic.
6. **Border objects.** Objects touching the grid border must still be
   counted and placed.

## Synthetic counterexamples (must NOT misfire / must fail closed)

1. One-pair shortcut: an input where two different properties agree on the
   first pair but diverge later -> all-pairs verification required.
2. Zero objects (all background) -> empty output, no crash.
3. Tie counts (two colors, equal segment counts) -> deterministic order.
4. Objects on quadrant borders -> deterministic quadrant assignment.
5. Non-zero background with objects -> background-aware extraction.
6. 1x1 input -> degenerate, reject.
7. Two identical objects -> position encoding must not collapse them.
8. Background-only 3x3 -> reject or the task's empty form.

## Acceptance conditions (executable when the proposer lands)

- Train-exactness: 1a2e2828 5/5, 19bb5feb 3/3, 794b24be 10/10, ff28f65a 8/8,
  5289ad53 4/4.
- No regression: full-training bench stays at >= 51/1000 with 0 contract
  violations, KAGGLE-LIKE per-task budget (~36s/task = 10h/1000).
- Task-agnostic: no task ids or dataset content in the proposer code.
- Fail-closed: exceptions and malformed candidates degrade to
  "not proposed", never crash the search loop.
