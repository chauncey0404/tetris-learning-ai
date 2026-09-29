# Future Search Performance Optimization

The Expert-v1.1 500K model is frozen. This patch changes execution only.

Safe optimizations included:

- bounded process-local Fast Reachability LRU keyed by exact board occupancy + piece + max_states;
- per-row exact memoization of structural metrics, NextEnvelope, and T tactical proxy;
- adaptive ProcessPool scheduling that splits a shortlist into contiguous chunks only when too few trajectories remain to occupy all workers;
- the existing exact NumPy batch structural metrics remain enabled.

Explicitly unchanged:

- model checkpoint/weights/architecture;
- Top-8 overall + Top-4/branch shortlist;
- recovery pairs;
- reachability rules;
- future feature definitions;
- BF16 neural call shapes;
- candidate pruning or approximate search.

Cross-state BF16 neural batching remains forbidden because it previously changed policy Top-1.

## Validation

```bat
.venv\Scripts\python.exe -m unittest ^
  tetrio.tests.test_future_structure_vectorization ^
  tetrio.tests.test_future_search_cache ^
  tetrio.tests.test_future_feature_cache_parity ^
  tetrio.tests.test_future_scheduling ^
  tetrio.tests.test_batched_rollout ^
  -v
```

Then run a longer strict parity gate on already-consumed seeds:

```bat
.venv\Scripts\python.exe -m tetrio.tools.compare_expert_models ^
  --baseline-checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --candidate-checkpoint models\tetrio_expert_v1_1_future_500k.pt ^
  --seeds 9071-9074 ^
  --parity-pieces 250 ^
  --parity-only ^
  --workers 16 ^
  --state-batch 20 ^
  --reference-audit-every 0 ^
  --device cuda
```

All V1 and V1.1 traces must match exactly. Any divergence rejects the optimization.

For throughput benchmarking, use already-consumed seeds:

```bat
.venv\Scripts\python.exe -m tetrio.tools.compare_expert_models ^
  --baseline-checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --candidate-checkpoint models\tetrio_expert_v1_1_future_500k.pt ^
  --seeds 9091-9094 ^
  --max-pieces 5000 ^
  --workers 16 ^
  --state-batch 20 ^
  --progress-every 512 ^
  --reference-audit-every 0 ^
  --device cuda ^
  --save-json artifacts\tetrio\expert_model_performance_benchmark.json
```

If parity is exact and throughput improves materially, keep the 500K checkpoint frozen and expand training next to 2M. Do not change recovery-pair count at the same time.
