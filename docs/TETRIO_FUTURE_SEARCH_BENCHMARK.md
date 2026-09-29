# Exact Future Search Performance Benchmark

The observed 25.85 states/s (4 seeds, 2000 pieces) cannot be directly compared
with the earlier 27.67 states/s (20 seeds, 5000 pieces). The workloads have
different concurrency and trajectory distributions.

This tool runs the same Expert V1.1 checkpoint, seeds and horizon under four
execution-only modes and requires exact placement-trace parity:

- baseline: immediate pre-optimization execution path;
- cache_memo: process-local reachability cache + per-row exact memoization;
- chunk2: cache/memo + adaptive scheduling capped at 2 chunks per state;
- chunk4: cache/memo + adaptive scheduling capped at 4 chunks per state.

The previously accepted NumPy structural vectorization remains enabled in every
mode, because it predates this optimization phase.

First run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.benchmark_future_search ^
  --checkpoint models\tetrio_expert_v1_1_future_500k.pt ^
  --seeds 9091-9094 ^
  --max-pieces 1000 ^
  --workers 16 ^
  --state-batch 20 ^
  --reference-audit-every 0 ^
  --device cuda
```

Then confirm only baseline + the winner at 2000 pieces.
