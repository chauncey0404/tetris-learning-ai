# TETR.IO Expert v0

Expert v0 is the first GPU imitation-learning baseline built on the validated
historical top-player corpus.

## What is frozen before this stage

- historical corpus: 7,716,524 rows / 76,692 games
- game-level train/val/test isolation
- placement coordinate mapping
- hold/preview timing
- TETR.IO-specific entry state
- expert final placement ↔ current reachability: 6,537 / 6,537 exact

## Staged v0 policy

Reference BFS currently runs only about ~10 rows/s on the validated Windows CPU
path.  Enumerating both no-hold and hold branches for millions of states would
make preprocessing the dominant cost.

Therefore v0 deliberately uses two supervised components:

```text
state
├─ hold head -> use_hold
└─ candidate scorer -> rank placements inside the expert-selected hold branch
```

This is a research baseline, not the final unified battle policy.  Once the fast
candidate backend exists, both hold branches can be generated and scored in one
shared list.

## Leakage-safe state contract

State features:

```text
40x10 binary board      400
active piece              7
hold piece                7
first 5 preview pieces   35
----------------------------
STATE_SIZE               449
```

Only previously validated pre-action fields are used.  Rating, win result,
attack, clear result, T-spin label, combo, B2B and garbage fields are not model
inputs in v0.

Candidate features:

```text
after-board             400
piece one-hot             7
rotation one-hot          4
x/y normalized            2
use_hold                  1
lines cleared             1
----------------------------
CANDIDATE_SIZE           415
```

Path-sensitive duplicates are deduplicated by final canonical landing geometry.
Expert v0 learns placement geometry; exact spin-path imitation is a later gate.

## 1. Unit tests

```bat
.venv\Scripts\python.exe -m unittest tetrio.tests.test_expert_v0_network -v
```

Also rerun the existing V8 ranking/checkpoint regression because the patch adds
shared network exports but does not modify `CandidateQNetwork`:

```bat
.venv\Scripts\python.exe -m singleplayer.tests.test_v8_8_7_ranking_aux
.venv\Scripts\python.exe -m singleplayer.tests.validate_champion_checkpoint
```

## 2. Build a practical first cache

Do not start by preprocessing all 6.6M training rows with the slow reference
BFS.  First establish the end-to-end learning baseline:

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v0_candidate_cache ^
  --input data\tetrio\expert\top_players_s1_train.parquet ^
  --output-dir data\tetrio\expert_v0\train_20k ^
  --rows 20000 ^
  --workers 16

.venv\Scripts\python.exe -m tetrio.tools.build_expert_v0_candidate_cache ^
  --input data\tetrio\expert\top_players_s1_val.parquet ^
  --output-dir data\tetrio\expert_v0\val_5k ^
  --rows 5000 ^
  --workers 16
```

Both manifests must report `Result: PASS` / `failed_rows: 0` before training.

Large cache files should remain local. Add:

```gitignore
data/tetrio/expert_v0/
```

## 3. RTX 5070 training

```bat
.venv\Scripts\python.exe -m tetrio.tools.train_expert_v0 ^
  --train-cache data\tetrio\expert_v0\train_20k ^
  --val-cache data\tetrio\expert_v0\val_5k ^
  --batch-size 256 ^
  --epochs 8 ^
  --device cuda
```

The trainer uses BF16 when supported by the installed CUDA/PyTorch stack,
otherwise FP16 AMP. Candidate scoring is batched on the GPU; no per-candidate
GPU calls are used.

Checkpoint:

```text
models/tetrio_expert_v0.pt
```

This checkpoint is explicitly a **RESEARCH BASELINE**, not a Champion.

## 4. Scale only after the first learning curve is healthy

If validation Top-1/Top-3 improve normally, build a larger reference cache, e.g.
50k-100k train / 10k-20k val.  Do not attempt to cache all 6.6M rows until the
fast reachability backend exists.

The next performance milestone is a semantics-matched fast candidate generator
(bitboard/precomputed geometry) so the full expert corpus and later self-play
can feed the RTX 5070 without starving it on CPU BFS.
