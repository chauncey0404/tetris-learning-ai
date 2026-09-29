# Expert-v0 Held-out Evaluation

Expert-v0 is an imitation-placement model, not yet a complete autonomous
TETR.IO battle agent. Its first formal test is therefore a held-out
game-disjoint test split.

Default test source:

```text
data/tetrio/expert/top_players_s1_test.parquet
```

Default cache:

```text
data/tetrio/expert_v0/test_full_fast_s8192
```

If the cache does not exist, evaluation builds it once with the same formal
reachability/exclusion contract used for training, then evaluates the checkpoint
without optimizer updates.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.eval_expert_v0 ^
  --checkpoint models\tetrio_expert_v0_full.pt ^
  --batch-size 8192 ^
  --device cuda ^
  --pipeline compact_gpu ^
  --prefetch-shards 4
```

Primary metrics:

- Top-1: model's first-ranked landing equals the expert landing.
- Top-3: expert landing appears in model's first three.
- MRR: reciprocal rank of the expert landing.
- Hold accuracy: separate Expert-v0 hold head.

The evaluator also prints the difference between held-out test metrics and the
validation metrics stored in the best checkpoint.

Do not tune hyperparameters on the held-out test split after reading the result.
Use validation for future model changes; reserve test as the generalization
report.
