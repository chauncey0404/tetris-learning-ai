# Architecture Consolidation — V9

## Intent

`tetris_ai/` is now the game-independent engine/library.
Concrete game or mode packages live beside it at repository root.

```text
tetris_ai/          shared mechanics, neural primitives, reusable learning
singleplayer/       V8.x final-placement/Teacher/DDQN lineage
tetrio/             TETR.IO-specific SRS+, spins, future attack/garbage/self-play
```

## Important splits

- One authoritative tetromino geometry: `tetris_ai/core/tetrominoes.py`.
- Shared board collision/lock/clear primitives: `tetris_ai/core/board.py`.
- Shared path-aware movement/SRS/reachability: `tetris_ai/core/`.
- Shared neural architecture primitive: `tetris_ai/networks/CandidateQNetwork`.
- V8 state243/candidate215 encoders remain under `singleplayer/network/`.
- Root `tools/`, `tests/`, `experiments/` were moved under `singleplayer/`.
- TETR.IO tests stay under `tetrio/tests/`.

## Checkpoint compatibility

`singleplayer.network.ObservableSafeQNetwork` preserves the old module names
(`state_encoder`, `candidate_encoder`, `joint`, `q_head`), so existing V8.x
state_dict checkpoint keys remain compatible.

## Commands

```bat
.venv\Scripts\python.exe -m unittest discover -s tetris_ai\core\tests -t . -v
.venv\Scripts\python.exe -m unittest discover -s tetrio\tests -t . -v
.venv\Scripts\python.exe -m unittest discover -s singleplayer\tests -t . -v

.venv\Scripts\python.exe -m singleplayer.tools.watch_models --help
```

The V8 formal Champion and training semantics are not changed by this
architecture-only migration.
