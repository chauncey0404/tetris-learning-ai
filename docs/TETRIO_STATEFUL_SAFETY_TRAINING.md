# Safety-Constrained Stateful Residual

This is the follow-up to the V1.2A 100K non-promotion result.

V1.2A proved that audited combo/B2B/history state contains real expert-policy
signal:

```text
V1.1 / ZERO Top1  = 0.5541
V1.2A TRUE Top1   = 0.5659
decision changes  ~= 5%
```

but structural quality became worse. Therefore this stage does not add more
data or relax the promotion gate. It changes only how the state residual is
allowed to learn from an expert disagreement.

## Frozen baseline and data

Unchanged:

```text
Base             = V1.1 500K epoch 7
Train future     = existing 100K cache
Val future       = existing 10K cache
Battle state     = existing audited state cache
State features   = unchanged
Network          = unchanged V1.2 state interaction residual
Recovery pairs   = 1,500 provenance only; no invented battle state
Fresh seeds      = untouched
```

## Expert-vs-baseline relation

For every row with expert recall, let:

```text
B = frozen V1.1 Top-1
E = historical expert target
```

The existing conservative structural dominance contract classifies the row as:

```text
agreement  : E == B
safe       : E safely dominates B
unsafe     : B safely dominates E
ambiguous  : neither safely dominates the other
```

Default imitation weights are fixed before running the pilot:

```text
agreement = 0.10
safe      = 1.00
unsafe    = 0.00
ambiguous = 0.20
```

Unsafe expert rows receive no imitation reward. Instead a targeted pair loss
requires the frozen baseline to remain above the structurally dominated expert
target.

The existing global dominance loss is also retained.

## Promotion contract

A checkpoint is written only when all conditions hold:

```text
TRUE Top1 >= ZERO Top1 + 0.0020
TRUE quality cost <= ZERO quality cost - 0.0005
decision_change_rate > 0
unsafe_change_rate <= 0.0050
ZERO residual == 0 exactly
ZERO score delta vs frozen V1.1 == 0 exactly
```

No post-hoc epoch selection is allowed.

## Run tests

```bat
.venv\Scripts\python.exe -m unittest ^
  tetrio.tests.test_expert_stateful_safety ^
  -v
```

## Train 100K pilot

No cache rebuild is needed:

```bat
.venv\Scripts\python.exe -m tetrio.tools.train_expert_stateful_safety ^
  --train-cache data\tetrio\expert_v1_1_future\train_100k ^
  --train-state-cache data\tetrio\expert_v1_2_state\train_100k ^
  --val-cache data\tetrio\expert_v1_1_future\val_10k ^
  --val-state-cache data\tetrio\expert_v1_2_state\val_10k ^
  --init-v1-1 models\tetrio_expert_v1_1_future_500k.pt ^
  --recovery-jsonl artifacts\tetrio\expert_v1_recovery_states.jsonl ^
  --recovery-weight 0.25 ^
  --output models\tetrio_expert_stateful_safety_100k.pt ^
  --metrics artifacts\tetrio\expert_stateful_safety_100k_training.json ^
  --epochs 10 ^
  --batch-size 1024 ^
  --device cuda
```

Each epoch reports:

```text
A/S/U/?       agreement / safe expert / unsafe expert / ambiguous rows
change        total TRUE-vs-ZERO Top-1 change rate
safechg       changed candidate safely dominates frozen baseline
unsafechg     frozen baseline safely dominates changed candidate
toExpert      state residual changed Top-1 to the expert target
```

If no epoch satisfies the frozen contract:

```text
Best epoch : NONE
Checkpoint : NOT WRITTEN
Status     : NON_PROMOTION
```

Do not expand to 500K in that case.

## Independent ablation after promotion

Only when a checkpoint was written:

```bat
.venv\Scripts\python.exe -m tetrio.tools.evaluate_expert_stateful_safety ^
  --checkpoint models\tetrio_expert_stateful_safety_100k.pt ^
  --future-cache data\tetrio\expert_v1_1_future\val_10k ^
  --state-cache data\tetrio\expert_v1_2_state\val_10k ^
  --device cuda
```
