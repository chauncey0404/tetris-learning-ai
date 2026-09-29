# Expert v1 — Unified Hold/No-Hold + Conservative Quality Supervision

## Why V1 exists

Expert-v0 passed held-out imitation but closed-loop rollout exposed a different
failure mode: the model can repeatedly create avoidable holes.

V0 has two limitations:

1. a separate state-only Hold head chooses HOLD/NO-HOLD before placement
   ranking, so the scorer never compares placements across both branches;
2. replay labels can contain misdrops, speed compromises, or battle-context
   decisions whose missing context is not present in the validated v0 state.

V1 changes the decision contract without touching the validated state:

```text
State = board + Active + Hold + Next5

NO-HOLD reachable candidates ─┐
                              ├─ one CandidateScoringNetwork ─ Top-1
HOLD reachable candidates ────┘
```

There is no binary Hold head.

## Disk/performance design

Do **not** duplicate the already-large V0 expert-branch cache.

V1 builds only the opposite branch as a sidecar. During training:

```text
existing V0 selected branch
+
V1 counterfactual sidecar branch
=
joint candidate set
```

This roughly halves the extra V1 cache storage compared with writing a second
full unified cache.

## Conservative label-quality treatment

V1 does not hard-ban holes.

For every row it computes holes before and holes after every candidate.

If the **expert label itself** creates a new hole while another legal candidate
does not, that row is flagged `expert_risky`. Its imitation weight is reduced
(default 0.35) instead of pretending a different move is definitely correct.
This covers possible human misdrops and missing battle context.

If the expert label is structurally safe, a small margin auxiliary encourages
the expert score above candidates that create new holes. This does not conflict
with the expert label.

Validation reports:

- Top1 / Top3 / MRR over the joint candidate set
- `branch_acc`: whether Top1 chose the expert HOLD/NO-HOLD branch
- `expert_risky_rate`: how often the dataset label itself is structurally risky
- `top1_hole_creation_rate`
- `top1_avoidable_hole_rate`

## Recommended first pilot: 100K / 10K

Build train sidecar:

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_counterfactual_cache ^
  --v0-cache data\tetrio\expert_v0\train_full_fast_s8192 ^
  --output-dir data\tetrio\expert_v1_cf\train_100k ^
  --rows 100000 ^
  --workers 16 ^
  --backend fast ^
  --reference-audit-every 5000
```

Build validation sidecar:

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_counterfactual_cache ^
  --v0-cache data\tetrio\expert_v0\val_10k ^
  --output-dir data\tetrio\expert_v1_cf\val_10k ^
  --rows 0 ^
  --workers 16 ^
  --backend fast ^
  --reference-audit-every 1000
```

Train V1, warm-starting from the frozen full V0 scorer:

```bat
.venv\Scripts\python.exe -m tetrio.tools.train_expert_v1 ^
  --train-v0-cache data\tetrio\expert_v0\train_full_fast_s8192 ^
  --train-cf-cache data\tetrio\expert_v1_cf\train_100k ^
  --val-v0-cache data\tetrio\expert_v0\val_10k ^
  --val-cf-cache data\tetrio\expert_v1_cf\val_10k ^
  --init-v0 models\tetrio_expert_v0_full.pt ^
  --output models\tetrio_expert_v1_joint_100k.pt ^
  --metrics artifacts\tetrio\expert_v1_joint_100k_training.json ^
  --batch-size 2048 ^
  --epochs 10 ^
  --device cuda
```

Then inspect closed-loop behavior with fresh V1 development seed 9031:

```bat
.venv\Scripts\python.exe -m tetrio.tools.watch_expert_v1 ^
  --checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --seed 9031 ^
  --max-pieces 5000 ^
  --device cuda ^
  --backend fast
```

Use 9031+ for the V1 development block. Keep 9001-9020 as consumed V0 rollout
development seeds, and keep protected 6-20 untouched.

## Promotion discipline

This is a research pilot, not a Champion candidate yet.

Do not inspect the held-out test split for V1 tuning. Compare on validation and
closed-loop development rollouts first. Only after architecture/weights are
frozen should the held-out test be used again as a final generalization report.
