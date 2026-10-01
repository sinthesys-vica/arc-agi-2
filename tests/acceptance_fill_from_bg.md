# Fill-from-bg proposer — acceptance & counterexample matrix

Owner: Dorian (train-only, blind discipline — structure observations only,
no test data; synthetic counterexamples are constructed, not dataset-derived)

## Family signature (shared skeleton)

1. **Same-size IO** — filling never resizes.
2. **Only 0 -> color changes** — existing non-zero cells are never moved,
   recolored, or deleted (verified on the three observed train0 pairs).
3. **The 0s are missing markers** — which 0s get filled and with what color
   is decided by CONTEXT: lattice continuation, neighborhood triggers, or
   pattern-symmetry completion.
4. **Everything else preserved.**

## Per-target observations (train0, honest limits)

| task | colors | observed mechanism |
|------|--------|--------------------|
| 0b17323b | [1], fill 2 | markers on the main diagonal at stride 4; the next diagonal position gets the fill color (2) |
| bb52a14b | [1,8], fill 4 | 0s flanking 8-cells that are adjacent to 1-clusters become 4 |
| 11852cab | [2,3,4] | partial symmetric pattern completed: missing mirror positions filled with their counterpart colors |
| 27a77e38 | [1,2,4] | 1 cell per pair, small grids — mechanism needs the other pairs |
| 72322fa7 | [1,2,4,6,8] | pairwise fill by neighbor color — needs the other pairs |

## Risk points

1. **Fill ambiguity.** A 0-cell with TWO different candidate colors in its
   context must fail closed, not pick silently.
2. **Which 0s?** Most 0s stay 0 — the selector (lattice position, neighbor
   trigger, symmetry counterpart) must be exact; filling ALL 0s is wrong.
3. **Existing-cell respect.** Never overwrite or move non-zero cells.
4. **Border and corner 0s** — context windows must clip, never go
   out-of-bounds.
5. **Dense neighborhoods** — conflicting signals (multiple neighbor colors)
   must be detected as ambiguity.

## Synthetic counterexamples (must NOT misfire / must fail closed)

1. A 0-cell with two different-colored neighbors -> reject (ambiguous).
2. A 0-cell with NO non-zero context at all -> stays 0, no invention.
3. Grid where all 0s would satisfy the selector -> the selector must be
   position-specific, never "fill everything".
4. A 0 on the border with one-sided context -> clip, no crash.
5. Multi-color dense cluster around a 0 -> reject.
6. Existing markers NOT on the lattice/pattern -> no silent extension.
7. 1xN / Nx1 degenerate -> reject.
8. Empty grid -> reject.

## Acceptance conditions (executable when the proposer lands)

- Train-exactness: 0b17323b 2/2, bb52a14b 3/3, 11852cab 3/3, 27a77e38 3/3,
  72322fa7 3/3.
- No regression: full-training bench stays at >= 46/1000 with 0 contract
  violations, KAGGLE-LIKE per-task budget (~36s/task = 10h/1000).
- Task-agnostic: no task ids or dataset content in the proposer code.
- Fail-closed: exceptions and malformed candidates degrade to
  "not proposed", never crash the search loop.
