# Line-drawing / point-connection proposer — acceptance & counterexample matrix

Owner: Dorian (train-only, blind discipline — structure observations only,
no test data; synthetic counterexamples are constructed, not dataset-derived)

## Honest family verdict first

The "line drawing / point connection" label is LOOSE. The five targets are
five different sub-rules that only share a thin skeleton:

| task | observed rule (train0) | shared? |
|------|------------------------|---------|
| 23581191 | singleton dot -> full row + full column in the dot's color; crossings of DIFFERENT colors become 2 | skeleton |
| 5ad8a7c0 | dots aligned in a row -> horizontal connection only; vertical alignments present but NOT drawn | skeleton |
| 4258a5f9 | singleton dot -> 3x3 solid block (color 1) with the dot color at the center — expansion, not a line | skeleton |
| 0a938d79 | two reference rows -> full-width stripes, alternating colors every 2 rows, periodic to the bottom | skeleton |
| 178fcbfb | per-dot full line; direction selector row-vs-column per dot; crossings keep the ROW color | skeleton |

Shared skeleton: (a) singleton nonzero dots in the input; (b) same-size IO;
(c) background preserved; (d) the dots' colors drive the drawn geometry.
Everything else differs. Recommendation: implement per-target sub-rules
(house pattern from the partitioning round: strict shared detector +
separate rule transforms), do NOT force one mega-rule.

## Risk points (Raiden's flags, sharpened with data)

1. **Crossing priority — CONTRADICTS between tasks.** 23581191: crossing of
   two different-colored lines = 2 (both train pairs confirm). 178fcbfb:
   crossing keeps the horizontal line's color. A shared proposer must not
   hardcode either; the rule must be per-target.
2. **Singleton condition.** Dots are single cells. A 2+ cell blob of the
   same color must fail closed (never treated as a dot).
3. **Boundary clipping.** 4258a5f9's block at center (1,7) touches row 0 —
   fitted exactly, nothing overflows. A dot AT the border (row 0 / col 0 /
   last row / last col) must clip, never raise out-of-bounds or draw outside.
4. **Same-color pairing ambiguity.** 5ad8a7c0 has FOUR dots of one color —
   pairing must be deterministic (observed: row alignment drawn, column
   alignment NOT drawn; the other 4 pairs are the discriminator source).

## Synthetic counterexamples (must NOT misfire / must fail closed)

1. Vertical-only pair: two dots sharing a column, nothing else — a
   23581191-style rule draws both arms; a 5ad8a7c0-style rule draws nothing.
   The proposer must not silently merge these semantics.
2. Crossing grid: two dots, different rows AND columns (plus-through
   situation) — the crossing cell value must match the target's rule.
3. Blob input: a 2-cell domino of one color — singleton check must reject.
4. Corner dot at (0,0) — boundary clipping, no exception, no overflow.
5. Three collinear same-color dots — deterministic pairing required;
   middle-dot ambiguity must not produce a wrong line silently.
6. Mixed content: one singleton dot plus one pre-existing line segment —
   not a pure dot scene; reject.
7. No dots (empty grid) — reject, no crash.
8. Touching dots (adjacent same-color cells) — they form a blob by
   4-connectivity; singleton check must reject.

## Acceptance conditions (executable when the proposer lands)

- Train-exactness: 23581191 2/2, 5ad8a7c0 5/5, 4258a5f9 2/2, 0a938d79 4/4,
  178fcbfb 3/3.
- No regression: full-training bench stays at >= 28/1000 with 0 contract
  violations (`tests/bench_pipeline.py`).
- Task-agnostic: no task ids or dataset content in the proposer code.
- Fail-closed: exceptions and malformed candidates degrade to
  "not proposed", never crash the search loop.
- Same-size IO and background preservation hold on every accepted output.
