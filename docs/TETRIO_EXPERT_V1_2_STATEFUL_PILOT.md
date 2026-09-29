# Expert V1.2A Stateful Residual — 100K Pilot

V1.2A is deliberately narrow. It keeps the frozen V1.1 500K policy and adds a
small candidate-specific interaction residual driven only by battle state that
passed the causality audit.

## Inputs

The first pilot uses only:

```text
raw_combo_before
raw_btb_before
previous_cleared
previous_t_spin_any
previous_t_spin_mini
previous_attack
previous_garbage_cleared
```

Current `incoming_garbage`, `immediate_garbage`, `won`, and current-action
`cleared/attack/t_spin` remain blocked.

The residual has no candidate-only shortcut. ZERO STATE always produces exactly
zero V1.2 residual even after training. This makes TRUE vs ZERO a real state
ablation rather than an extra-capacity ablation.

## Recovery corpus

The frozen V1.1 500K base already includes the 1,500 recovery-pair training.
The historical recovery JSONL does not contain audited battle state, so V1.2A
records its count/weight for provenance but does not fabricate state or apply a
new recovery gradient. Changing recovery data is deferred to a separate
experiment.

## Build 100K state adapters

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_2_state_cache ^
  --future-cache data\tetrio\expert_v1_1_future\train_100k ^
  --battle-state-parquet data\tetrio\expert_stateful\top_players_s1_train_battle_state.parquet ^
  --output-dir data\tetrio\expert_v1_2_state\train_100k ^
  --threads 20

.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_2_state_cache ^
  --future-cache data\tetrio\expert_v1_1_future\val_10k ^
  --battle-state-parquet data\tetrio\expert_stateful\top_players_s1_val_battle_state.parquet ^
  --output-dir data\tetrio\expert_v1_2_state\val_10k ^
  --threads 20
```

The adapter is fail-closed if the selected future rows intersect an ambiguous
`(game_id,subframe)` duplicate key.

## E00 parity

```bat
.venv\Scripts\python.exe -m tetrio.tools.check_expert_v1_2_e00_parity ^
  --base models\tetrio_expert_v1_1_future_500k.pt ^
  --future-cache data\tetrio\expert_v1_1_future\val_10k ^
  --state-cache data\tetrio\expert_v1_2_state\val_10k ^
  --device cuda
```

Required:

```text
Max TRUE residual = 0
Max ZERO residual = 0
Max ZERO delta score vs V1.1 = 0
Decision change = 0
PASS
```

## Train 100K

```bat
.venv\Scripts\python.exe -m tetrio.tools.train_expert_v1_2 ^
  --train-cache data\tetrio\expert_v1_1_future\train_100k ^
  --train-state-cache data\tetrio\expert_v1_2_state\train_100k ^
  --val-cache data\tetrio\expert_v1_1_future\val_10k ^
  --val-state-cache data\tetrio\expert_v1_2_state\val_10k ^
  --init-v1-1 models\tetrio_expert_v1_1_future_500k.pt ^
  --recovery-jsonl artifacts\tetrio\expert_v1_recovery_states.jsonl ^
  --recovery-weight 0.25 ^
  --output models\tetrio_expert_v1_2_stateful_100k.pt ^
  --metrics artifacts\tetrio\expert_v1_2_stateful_100k_training.json ^
  --epochs 10 ^
  --batch-size 1024 ^
  --device cuda
```

Selection is conservative: the TRUE-state quality cost must beat frozen V1.1
by at least 0.0005 while Top1 stays within 0.005 absolute and ZERO STATE stays
exactly inert. If nothing passes, no checkpoint is written and status is
`NON_PROMOTION`.

## Post-training ablation

```bat
.venv\Scripts\python.exe -m tetrio.tools.evaluate_expert_v1_2_ablation ^
  --checkpoint models\tetrio_expert_v1_2_stateful_100k.pt ^
  --future-cache data\tetrio\expert_v1_1_future\val_10k ^
  --state-cache data\tetrio\expert_v1_2_state\val_10k ^
  --device cuda
```

Do not spend fresh seeds 9111-9130 at the 100K pilot stage. First decide
whether TRUE STATE materially beats ZERO STATE offline. Only a passing 100K
pilot should scale to 500K.
