# Round 9 — PixelExpansionProposer Acceptance Matrix

**Target cluster**: Tasks where the output is an integer-scaled expansion of the input, with each input pixel mapping to an NxN block in the output.
**Candidate pool**: 14 unsolved tasks (6 fractal + 2 complement + 5 mirror4 + 1 color-template).
**Mechanism**: Detect integer scale factor → determine pixel-to-block rule → expand each pixel → verify.

## Acceptance Criteria

The proposer MUST:
1. Return `Hypothesis` objects with `.apply(input_grid) → output_grid`
2. Solve ≥4 of the 5 acceptance tasks below on training pairs (train-only verification)
3. Not regress any currently-solved tasks (0 contract violations)
4. Complete within 36s/task budget

## Key Implementation Requirements

### Scale Detection
- Check if `output.shape == (N * input.shape[0], N * input.shape[1])` for N in {2, 3, 4, 5}
- Both dimensions must share the same integer scale factor
- If no integer scale factor exists, skip this task

### Strategy Portfolio (try in order, early-exit on verified)

**Strategy A: Fractal Self-Substitution (Nx)**
- For each unique color c in the input:
  - Build candidate output: pixels with color c → replaced by a COPY of the entire input grid; all other pixels → block of all zeros (same size as input)
  - Verify against all training pairs
- The "key" color changes between pairs — must find it per-pair:
  - Try EVERY color in the input as the key; verify consistency
- Important: the key color is NOT always the most/least common — try all colors
- Covers: 007bbfb7, 15696249, 27f8ce4f, 48f8583b, c3e719e8, cce03e0d

**Strategy B: Complement Substitution (Nx)**
- Only works on inputs with exactly 2 distinct colors (binary grids)
- For each of the 2 colors (c1, c2):
  - Build candidate: pixels with c1 → complement of input (swap c1↔c2), pixels with c2 → all zeros
  - Verify against all training pairs
- Complement = swap the two colors in the input grid
- Covers: 0692e18c, 8e2edd66

**Strategy C: Mirror4 Tiling (2x only)**
- Output is exactly 2H × 2W
- Try 4 quadrant arrangements:
  - `mirror4`: TL=input, TR=fliplr(input), BL=flipud(input), BR=fliplr(flipud(input))
  - `mirror4_rev`: TL=rot180, TR=flipud, BL=fliplr, BR=input
  - `mirror4_h`: TL=fliplr, TR=input, BL=rot180, BR=flipud
  - `mirror4_v`: TL=flipud, TR=rot180, BL=input, BR=fliplr
- Verify the chosen arrangement against ALL training pairs
- Covers: 0c786b71, 3af2c5a8, 62c24649, 67e8384a, 833dafe3

**Strategy D: Learned Color-Block Template (Nx)**
- From training pair 0: for each color c, extract the NxN block at the first occurrence of c
- Build a color→block mapping
- Verify this mapping is consistent:
  - Same color always maps to same block within a pair
  - Same mapping works across ALL training pairs
- If consistent, apply to test input
- Covers: 2072aba6 (and catches any Strategy A/B misses)

## Acceptance Tasks

### Task 1: 007bbfb7 (Fractal Self-Substitution, 3x)
- Input: 3×3, Output: 9×9, Scale: 3
- Binary colors (one non-zero + zero per pair)
- Mechanism: non-zero color → COPY OF INPUT, zero → all-zeros block
- Train pairs: 5
- Example: input has color 6 pattern → each 6 becomes the full 3×3 input, each 0 becomes 3×3 zeros

### Task 2: 15696249 (Fractal Self-Sub with multi-color input, 3x)
- Input: 3×3, Output: 9×9, Scale: 3
- Multi-color inputs (3+ colors per pair)
- Mechanism: ONE specific color → COPY OF INPUT, ALL other colors → all-zeros block
- Key color identification: must try all colors (not always most common)
- Train pairs: 4
- Harder test: proposer must try each color as "key" and verify

### Task 3: 0692e18c (Complement Substitution, 3x)
- Input: 3×3, Output: 9×9, Scale: 3
- Binary colors (exactly 2 colors per pair)
- Mechanism: non-bg → COMPLEMENT (swap the two colors), bg → all-zeros
- Complement = input with color A↔B swapped
- Train pairs: 3

### Task 4: 3af2c5a8 (Mirror4 Tiling, 2x)
- Input: 3×4, Output: 6×8, Scale: 2
- Multi-color
- Mechanism: TL=input, TR=fliplr(input), BL=flipud(input), BR=rot180(input)
- Standard mirror4 arrangement — verified on all 3 training pairs
- Train pairs: 3

### Task 5: 2072aba6 (Learned Color-Block Template, 2x)
- Input: 3×3, Output: 6×6, Scale: 2
- Input colors: {0, 5}. Output colors: {0, 1, 2} (NEW colors appear!)
- Mechanism: color 5 → fixed 2×2 block [[1,2],[2,1]], color 0 → [[0,0],[0,0]]
- Template is consistent across ALL 3 training pairs
- This template must be LEARNED from training (cannot be derived from input alone)
- Train pairs: 3

## Pre-filter Rule (for search.py)

```python
# PixelExpansionProposer: skip same-size tasks
if name == "PixelExpansionProposer" and sig.all_same_size:
    return True
```

## Verification Protocol

1. Atlas implements → pushes commit
2. Raiden runs `python -m pytest tests/ -q` (must stay at 51+ PASS, 0 FAIL)
3. Raiden runs benchmark: `solve_all(tasks, proposers, 600)` — must be ≥61, 0 contract violations
4. Dorian runs independent verification on acceptance tasks
