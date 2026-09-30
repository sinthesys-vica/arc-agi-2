# ARC-AGI-2 — Team Sinthesys

Kaggle ARC Prize 2026 submission.

## Team

| Member | Role | Module |
|--------|------|--------|
| Atlas | SceneGraph IR + DSL | `arc2/scene.py`, `arc2/dsl.py`, `arc2/proposers.py` |
| Dorian | Encoder + Verifier | `arc2/verify.py` |
| Raiden | Search + Orchestration | `arc2/search.py`, `arc2/solve.py`, `arc2/submission.py` |

## Architecture

Modular hypothesis-search kernel:
1. **Scene** — grid → structured scene representation
2. **DSL** — composable transformation primitives
3. **Proposers** — hypothesis generation from scene + DSL
4. **Search** — hypothesis space exploration strategy
5. **Verify** — hypothesis validation against examples
6. **Solve** — end-to-end orchestration per task
7. **Submission** — Kaggle notebook wrapper

## Submission strategy

- Attempt 1: best hypothesis
- Attempt 2: best structurally different hypothesis

## Deadlines

- Oct 26: entry gate
- Nov 2: final submission
