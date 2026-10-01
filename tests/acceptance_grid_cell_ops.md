# Round 8 — GridCellOperationProposer Acceptance Matrix

**Target cluster**: Tasks with rectangular grid dividers (uniform rows/cols of a single non-zero color forming H+V cross pattern).
**Candidate pool**: 40 same-size + 37 different-size = 77 unsolved tasks with strict grid structure.
**Mechanism**: Detect grid → partition cells → learn cell-level transforms from training → apply to test.

## Acceptance Criteria

The proposer MUST:
1. Return `Hypothesis` objects with `.apply(input_grid) → output_grid`
2. Solve ≥4 of the 5 acceptance tasks below on training pairs (train-only verification)
3. Not regress any currently-solved tasks (0 contract violations)
4. Complete within 36s/task budget

## Key Implementation Requirements

### Grid Detection
- Find all **uniform rows** (every cell same non-zero color) and **uniform columns** (same)
- A valid grid has ≥1 horizontal divider AND ≥1 vertical divider of the **same** color
- Cells are the rectangular regions BETWEEN dividers (excluding divider rows/cols themselves)
- Divider color is constant across all dividers

### Strategy Portfolio (try in order, early-exit on verified)

**Strategy A: Key-Pattern Recolor**
- Identify "key cells" containing a single non-zero non-background pixel (the color marker)
- Identify "pattern cells" containing a shape drawn in a "neutral" color
- Output: replace neutral color in pattern cells with corresponding key cell's color
- Key-pattern pairing: same grid row, or same grid column

**Strategy B: Grid Stamp / Broadcast**
- Find cells with content (non-empty, non-background)
- One cell per grid-row (or grid-col) serves as template
- Broadcast template to all cells in same grid-row/col
- Color may come from a separate key cell

**Strategy C: Cell Extraction (different-size tasks)**
- Output shape matches one cell's dimensions
- Try each cell as candidate output
- Selection criteria: unique content, non-empty, most colorful, specific position

**Strategy D: Cell Reflection/Symmetry**  
- Fill empty cells by reflecting non-empty cells across divider lines
- Try all D4 reflections per cell

## Acceptance Tasks

### Task 1: ef26cbf6 (Key-Pattern Recolor)
- Grid: 3 grid-rows × 2 grid-cols, divider color 4
- Cell size: 3×3 (uniform)
- Mechanism: left column = color key (single pixel: 7, 3, 8), right column = pattern in color 1
- Output: replace 1s in right cells with corresponding left cell's key color
- Train pairs: 2

### Task 2: 5a5a2103 (Grid Stamp)
- Grid: 4×4, divider color 3 (or 8)
- Cell size: 4×4 (uniform)
- Mechanism: one cell per row has a 2×2 color block (key), one cell has a shape template
- Output: ALL cells get the template stamped in their row's key color
- Train pairs: 2

### Task 3: 1e32b0e9 (Pattern Fill)
- Grid: 3×3, divider color 2
- Cell size: 5×5 (uniform)
- Mechanism: some cells have colored patterns, others are empty
- Output: empty cells get filled based on patterns from occupied cells
- Train pairs: 3

### Task 4: e3fe1151 (Cell Transform)
- Grid: 2×2, divider color 7
- Cell size: 2×2 (uniform)
- Mechanism: cells contain colored pixels, one cell has a unique value
- Output: the unique value propagates to specific positions
- Train pairs: 2

### Task 5: 2dc579da (Cell Extraction — different-size)
- Grid: 2×2, divider color with 4 cells
- Input: 7×7, Output: 3×3
- Mechanism: output = content of one specific cell
- Selection criterion: determined by training examples
- Train pairs: 3

## Pre-filter Rule (for search.py)

```python
# GridCellOperationProposer: no pre-filter needed
# Works on both same-size and different-size tasks
# The grid detection itself is the filter
```

## Verification Protocol

1. Atlas implements → pushes commit
2. Raiden runs `python -m pytest tests/ -q` (must stay at 50+ PASS, 0 FAIL)
3. Raiden runs benchmark: `solve_all(tasks, proposers, 600)` — must be ≥56, 0 contract violations
4. Dorian runs independent verification on acceptance tasks
