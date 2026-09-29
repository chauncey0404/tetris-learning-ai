# TETR.IO Strict-Parity Parallel Headless Rollout

The first cross-state CUDA batching attempt failed the required policy parity
gate. Seed 9052 diverged at move 32 even though seed 9051 passed.

The cause is the neural execution shape. Expert-v1 uses BF16 autocast. Merging
multiple states/candidates into a different GEMM shape can slightly change
near-tied logits. One tiny Top-1 flip changes the entire closed-loop trajectory.

For A/B evaluation, policy identity is more important than maximizing Task
Manager GPU utilization.

## Retained hardware optimization

```text
up to 20 deterministic trajectories
          |
16 CPU processes enumerate HOLD / NO-HOLD reachability in parallel
          |
for each ready state:
    state encoder B=1                <- same as legacy
    NO-HOLD scorer call              <- same candidate shape as legacy
    HOLD scorer call                 <- same candidate shape as legacy
          |
commit exact policy Top-1
```

For Expert-v1.1, future feature search still runs in the 16-process pool, while
the residual reranker preserves the legacy `[1,K]` neural call shape.

This means CPU utilization should remain much better than the old sequential
runner. GPU utilization will be bursty/modest by design because cross-state
BF16 batching is not allowed to change the policy being measured.

## Required parity gate

```bat
.venv\Scripts\python.exe -m tetrio.tools.compare_expert_models ^
  --baseline-checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --candidate-checkpoint models\tetrio_expert_v1_1_future_20k.pt ^
  --seeds 9051-9054 ^
  --parity-pieces 40 ^
  --parity-only ^
  --workers 16 ^
  --state-batch 20 ^
  --reference-audit-every 0 ^
  --device cuda
```

All V1 and V1.1 checks must pass before running the 9051-9070 development A/B.

## Paired A/B

```bat
.venv\Scripts\python.exe -m tetrio.tools.compare_expert_models ^
  --baseline-checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --candidate-checkpoint models\tetrio_expert_v1_1_future_20k.pt ^
  --seeds 9051-9070 ^
  --max-pieces 5000 ^
  --workers 16 ^
  --state-batch 20 ^
  --progress-every 512 ^
  --reference-audit-every 0 ^
  --device cuda ^
  --save-json artifacts\tetrio\expert_model_comparison_9051_9070.json
```

New persistent program filenames remain functionality-based and version-neutral.
