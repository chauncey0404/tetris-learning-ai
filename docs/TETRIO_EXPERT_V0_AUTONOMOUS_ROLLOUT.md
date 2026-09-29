# TETR.IO Expert-v0 Autonomous Rollout

This is the first closed-loop Expert-v0 test.

Unlike held-out imitation evaluation, the model now creates its own future
states:

```text
empty 40x10 board
    -> deterministic TETR.IO-style 7-bag
    -> Active / Hold / Next 5
    -> Hold head chooses branch
    -> enumerate legal TETR.IO placements
    -> CandidateScoringNetwork ranks the chosen branch
    -> lock model Top-1
    -> clear lines
    -> advance Hold / bag
    -> repeat until top-out or max-pieces
```

Expert-v0 v0 still uses its trained staged policy:

```text
state-only Hold head
then placement ranking inside the selected Hold branch
```

It is not yet the future Expert-v1 unified no-hold/hold counterfactual scorer.

## GUI

```bat
.venv\Scripts\python.exe -m tetrio.tools.watch_expert_v0 ^
  --checkpoint models\tetrio_expert_v0_full.pt ^
  --seed 9001 ^
  --max-pieces 5000 ^
  --device cuda ^
  --backend fast
```

Controls follow the existing V3.4 viewer lineage:

```text
Space       Play / Pause
Right       single-step one piece
R           reset same seed
N           next seed
1..9        speed presets
+ / -       speed
D           details
S           screenshot
Esc / Q     quit
```

The board shows the model's chosen landing as an outline ghost before the piece
is committed.

## Headless development block

```bat
.venv\Scripts\python.exe -m tetrio.tools.watch_expert_v0 ^
  --checkpoint models\tetrio_expert_v0_full.pt ^
  --headless ^
  --seeds 9001-9020 ^
  --max-pieces 5000 ^
  --device cuda ^
  --backend fast ^
  --reference-audit-every 250 ^
  --save-json artifacts\tetrio\expert_v0_rollout_9001_9020.json
```

The periodic reference audit compares the *entire* Fast candidate geometry set
against the path-sensitive reference engine on self-generated states. Any
difference is a hard error.

Metrics:

- pieces survived
- lines / singles / doubles / triples / tetrises
- hold rate
- current / max holes
- average / max height
- average candidate count
- Fast/Reference audits and fallbacks
- terminal reason

These are development seeds. This rollout is not a Champion promotion gate.

## Targeted cleanup

The cleanup script removes only temporary diagnostics that are no longer part
of the frozen raise=1 + reference-exclusion pipeline, plus the contradicted
raise=2 migration document. It deliberately keeps production and safety tools.

```bat
powershell -NoProfile -ExecutionPolicy Bypass ^
  -File project_admin\tools\cleanup_tetrio_expert_v0_obsolete.ps1
```
