# Round 10 — MiscTransformProposer Acceptance Matrix

**Target cluster**: Tasks solvable by simple well-defined pixel-level transformations that no existing proposer handles.
**Candidate pool**: 6 confirmed unsolved tasks across 3 mechanism families.
**Mechanism**: Try portfolio of simple transforms, verify against training pairs, early-exit on match.

## Acceptance Criteria

The proposer MUST:
1. Return `Hypothesis` objects with `.apply(input_grid) → output_grid`
2. Solve ≥4 of the 5 acceptance tasks below on training pairs (train-only verification)
3. Not regress any currently-solved tasks (0 contract violations)
4. Complete within 36s/task budget

## Key Implementation Requirements

### Background Detection
- Background = most common color in input (Counter.most_common)
- Consistent across all strategies

### Strategy Portfolio (try in order, early-exit on verified)

**Strategy A: Gravity (4 directions)**
- Move all foreground (non-bg) pixels in a cardinal direction until they hit the grid edge or another fg pixel
- Preserve relative order of fg pixels along each row/column
- Try: down, up, left, right
- Implementation per direction (e.g., "down"):
  - For each column: collect all non-bg pixels top-to-bottom
  - Place them bottom-to-top at the bottom of the column
  - Fill remaining positions with bg
- Verify against ALL training pairs

**Strategy B: Connect Same-Color Markers (H, V, H+V)**
- Find pairs of same-color non-bg pixels in the same row (H) or column (V)
- Fill all bg pixels BETWEEN them with that color (only between the leftmost and rightmost / topmost and bottommost occurrences)
- Try: H-only, V-only, H-then-V (H+V)
- Important: only fill bg pixels, never overwrite existing fg pixels
- Verify against ALL training pairs

**Strategy C: Diagonal Shift (row or column)**
- Each row is circularly shifted by `shift * row_index` positions
- Or: each column is circularly shifted by `shift * col_index` positions
- Try shift values: 1, -1, 2, -2
- Try both row-shift and column-shift variants
- Total: 8 combinations (2 axes × 4 shifts)
- Verify against ALL training pairs

## Acceptance Tasks

### Task 1: 1e0a9b12 (Gravity Down)
- Same-size task
- Mechanism: all non-bg pixels "fall" downward in each column
- Relative order preserved within each column
- Train pairs verify consistently

### Task 2: 3906de3d (Gravity Up)
- Same-size task
- Mechanism: all non-bg pixels "rise" upward in each column
- Same as gravity down but reversed direction

### Task 3: 22168020 (Connect H)
- Same-size task
- Mechanism: same-color non-bg pixels in the same row connected by filling bg between them
- Only horizontal connections (no vertical)

### Task 4: ded97339 (Connect H+V)
- Same-size task
- Mechanism: same-color markers connected both horizontally AND vertically
- H connections applied first, then V connections

### Task 5: e9afcf9a (Diagonal Column Shift)
- Same-size task
- Mechanism: each column circularly shifted by `1 * column_index` positions
- col_shift_1 variant

## Additional Target Tasks (beyond acceptance 5)

- 22eb0ac0: Connect H (same as Task 3 mechanism)

## Pre-filter Rule (for search.py)

```python
# MiscTransformProposer: skip different-size tasks
if name == "MiscTransformProposer" and sig.all_different_size:
    return True
```

## Verification Protocol

1. Atlas implements → pushes commit
2. Raiden runs `python -m pytest tests/ -q` (must stay at 53+ PASS, 0 FAIL)
3. Raiden runs benchmark: `solve_all(tasks, proposers, 600)` — must be ≥77, 0 contract violations
4. Dorian runs independent verification on acceptance tasks
