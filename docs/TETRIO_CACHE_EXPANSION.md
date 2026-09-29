# Cache Expansion and Safe Performance Work

This patch prepares the Expert-v1.1 500K experiment without changing the model,
shortlist contract, recovery corpus, or policy semantics.

## Safe optimizations applied now

### Cross-directory shard reuse

Both builders accept:

```text
--resume
--reuse-dir <smaller completed cache>
```

A shard is reused only when its expected row count and row identities match the
current authoritative cache.

This matters when expanding 100K -> 500K. A 100K cache ends with a partial
shard; all complete shards are reused, while the partial final shard is
automatically rejected and rebuilt at full size.

### Vectorized structural metrics

Future lookahead still enumerates exactly the same reachable placements. Only
the integer board metrics after those placements are computed in a NumPy batch
instead of repeated Python column scans.

The values are required to be bit-for-bit integer-equivalent to the previous
scalar implementation. Run the supplied test and the strict rollout parity
gate before accepting the optimization.

## Deliberately deferred

Do NOT change these for the 500K experiment:

- Top-8 overall + Top-4/branch shortlist.
- 1,500 recovery pairs.
- model architecture / max adjustment.
- cross-state BF16 neural batching.
- approximate future search.
- pruning next-envelope candidates.

Those would change either the experiment variable or policy semantics.

## Expand counterfactual sidecar to 500K

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_counterfactual_cache ^
  --v0-cache data\tetrio\expert_v0\train_full_fast_s8192 ^
  --output-dir data\tetrio\expert_v1_cf\train_500k ^
  --reuse-dir data\tetrio\expert_v1_cf\train_100k ^
  --rows 500000 ^
  --workers 16 ^
  --backend fast ^
  --reference-audit-every 5000
```

## Expand future cache to 500K

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_1_future_cache ^
  --v0-cache data\tetrio\expert_v0\train_full_fast_s8192 ^
  --cf-cache data\tetrio\expert_v1_cf\train_500k ^
  --checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --output-dir data\tetrio\expert_v1_1_future\train_500k ^
  --reuse-dir data\tetrio\expert_v1_1_future\train_100k ^
  --rows 500000 ^
  --batch-size 1024 ^
  --workers 16 ^
  --device cuda
```

## Train 500K

Keep the same recovery file and weight:

```bat
.venv\Scripts\python.exe -m tetrio.tools.train_expert_v1_1 ^
  --train-cache data\tetrio\expert_v1_1_future\train_500k ^
  --val-cache data\tetrio\expert_v1_1_future\val_10k ^
  --init-v1 models\tetrio_expert_v1_joint_100k.pt ^
  --recovery-jsonl artifacts\tetrio\expert_v1_recovery_states.jsonl ^
  --recovery-weight 0.25 ^
  --output models\tetrio_expert_v1_1_future_500k.pt ^
  --metrics artifacts\tetrio\expert_v1_1_future_500k_training.json ^
  --epochs 10 ^
  --batch-size 1024 ^
  --device cuda
```

After training, use fresh development seeds 9091-9110. Do not reuse 9051-9090
for tuning.
