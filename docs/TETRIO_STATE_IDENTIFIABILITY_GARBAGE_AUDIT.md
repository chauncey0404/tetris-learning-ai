# Stateful Identifiability + Garbage Context Audit

This stage deliberately does **not** train a promotable model and does not use
fresh closed-loop seeds.

It answers two questions before any new battle-context architecture is built.

## 1. Does the exact row-specific state matter?

The previous stateful pilots showed that non-zero state can change the policy,
but that is not enough. A residual network could simply learn that "some state
is present" and apply a generic correction.

The identifiability audit retrains the original V1.2A diagnostic residual in
memory and evaluates the same validation rows under:

```text
TRUE ALL
SHUFFLED ALL
ZERO

TRUE combo/B2B only
SHUFFLED combo/B2B only

TRUE previous-outcome only
SHUFFLED previous-outcome only
```

SHUFFLED uses a deterministic global derangement: the marginal state
distribution is unchanged, but no validation row receives its own state.

Primary gate, fixed before running:

```text
TRUE_ALL Top1 - SHUFFLED_ALL Top1 >= 0.0020
for at least 3 consecutive epochs
```

No checkpoint is written by this audit.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.audit_state_identifiability ^
  --train-cache data\tetrio\expert_v1_1_future\train_100k ^
  --train-state-cache data\tetrio\expert_v1_2_state\train_100k ^
  --val-cache data\tetrio\expert_v1_1_future\val_10k ^
  --val-state-cache data\tetrio\expert_v1_2_state\val_10k ^
  --init-v1-1 models\tetrio_expert_v1_1_future_500k.pt ^
  --epochs 10 ^
  --batch-size 1024 ^
  --device cuda
```

Interpretation:

```text
TRUE >> SHUFFLED
    row-specific state identity is useful

TRUE ~= SHUFFLED > ZERO
    non-zero state mostly acts as a generic residual gate

combo/B2B PASS, previous FAIL
    prioritize combo/B2B

previous PASS, combo/B2B FAIL
    prioritize lagged outcome history
```

## 2. What can the historical garbage columns actually prove?

The historical placement corpus contains:

```text
incoming_garbage
immediate_garbage
```

but does not contain an authoritative opponent send/arrival/activation/tank
event stream. Therefore the garbage audit is intentionally evidence-only.

It reports:

- field distributions;
- whether one field is consistently nested inside the other;
- lead/lag between positive `incoming_garbage` and `immediate_garbage` changes;
- whether own positive attack is more associated with a queue drop on the
  current transition (post-action hypothesis) or next transition (pre-action
  hypothesis).

These are hypothesis-ranking signals only. The fields remain blocked for model
input until real replay/capture parity confirms the semantics.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.audit_garbage_context ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --games 5000 ^
  --threads 20 ^
  --output artifacts\tetrio\garbage_context_audit.json
```

If the result remains blocked, use the existing V9.3B replay tools on a real
`.ttrm` or controlled capture:

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_ttrm "C:\path\match.ttrm" ^
  --output artifacts\tetrio\parity\match_inventory.json

.venv\Scripts\python.exe -m tetrio.tools.validate_garbage_parity ^
  artifacts\tetrio\parity\match_trace.json
```

Fresh development seeds `9111-9130` remain unused throughout this stage.
