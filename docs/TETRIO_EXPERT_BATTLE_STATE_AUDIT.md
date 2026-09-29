# Historical Expert Battle-State Causality Audit

This stage exists before V1.2 model work. It does not train a model.

The historical top-player corpus contains:

```text
cleared, garbage_cleared, attack, t_spin,
btb, combo, immediate_garbage, incoming_garbage
```

but the existence of a column does not prove that it is safe to expose at the
decision time of the same row.

## What is reconstructed without leakage

For row `t`, the audit reconstructs only from rows `< t`:

- `combo_chain_before`
- `combo_index_before`
- `difficult_chain_before`
- previous clear
- previous T-spin label
- previous attack
- previous garbage cleared

The difficult-clear chain is a historical-corpus diagnostic. It is not treated
as proof that the 2024 dataset uses current Season-2 B2B Charging semantics.

## Raw combo / B2B timing

The audit compares the raw `combo` and `btb` values against both:

```text
state before row t
state after row t
```

using only informative rows where the state actually changes.

It also tries small integer offsets because historical counters may use:

```text
inactive=-1, first=0
```

while the normalized reconstruction uses:

```text
inactive length=0, first length=1
```

A raw field is approved only when one timing direction passes the configured
match threshold and clearly beats the other timing direction.

If timing is unresolved, the field stays blocked.

## Garbage fields

`incoming_garbage` and `immediate_garbage` are profiled but deliberately remain:

```text
BLOCKED_PENDING_GARBAGE_TIMING_PARITY
```

The placement-only dataset does not contain enough opponent/transport events to
silently declare either snapshot to be exact action-time incoming state.

## Run the audit

```bat
.venv\Scripts\python.exe -m unittest ^
  tetrio.tests.test_expert_battle_state ^
  tetrio.tests.test_battle_state_audit ^
  -v

.venv\Scripts\python.exe -m tetrio.tools.audit_expert_battle_state ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --games 5000 ^
  --threads 20 ^
  --output artifacts\tetrio\expert_battle_state_audit.json
```

Review the terminal summary and JSON before building V1.2 inputs.

## Optional next step after review

The patch also includes a fail-closed sidecar builder:

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_battle_state_cache ^
  --source data\tetrio\processed\top_players_s1.parquet ^
  --expert-dir data\tetrio\expert ^
  --audit artifacts\tetrio\expert_battle_state_audit.json ^
  --output-dir data\tetrio\expert_stateful ^
  --threads 20
```

It reconstructs history from the full source game first, then joins by
`game_id, subframe` to the existing leakage-safe train/val/test splits.

It will include raw combo/B2B only if the audit explicitly approved their
timing. Current incoming/immediate garbage never enters the state sidecar in
this stage.

## V1.2 gate

Do not implement the V1.2 residual network until the audit is reviewed.

The intended E00 contract remains:

```text
V1.2 E00 policy == frozen V1.1 500K policy
```

and fresh development seeds `9111-9130` stay unused until a V1.2 candidate
passes offline gates.
