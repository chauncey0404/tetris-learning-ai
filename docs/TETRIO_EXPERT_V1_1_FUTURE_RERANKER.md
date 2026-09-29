# TETR.IO Expert v1.1 — Future-Aware Residual Reranker

## Motivation

Expert-v1 fixed the staged Hold-head problem by jointly ranking HOLD and
NO-HOLD candidates. Closed-loop rollout still exposed failures such as:

- using J to seal a T-slot even though T is next;
- holding an active T instead of cashing out a reachable tactical opportunity;
- creating several avoidable holes and entering unfamiliar recovery states.

V1.1 does **not** blindly increase the hole penalty and does **not** add the
absolute score of a future neural state. Candidate-scoring logits are calibrated
inside one state, not across unrelated future states.

Instead:

```text
all V1 joint candidates
        |
frozen V1 base scorer
        |
Top-8 overall + Top-4 HOLD + Top-4 NO-HOLD
        |
one-step exact queue transition
        |
future structural/tactical envelope
        |
small bounded residual reranker
        |
final Top-1
```

The residual starts at exactly zero, so epoch 0 reproduces V1.

## Future features

For every shortlisted current candidate:

- immediate holes / height / bumpiness / wells / line clear;
- exact next Active/Hold state from the exposed queue;
- legal next-branch candidate count;
- best one-step recoverable holes / height / bumpiness / lines;
- fraction of next placements that do not add holes;
- T proximity in Active/Hold/next two known preview pieces;
- fast 3-corner T-slot proxy before/after;
- path-sensitive exact T-spin classification when T is immediate and a proxy
  opportunity exists;
- whether a T opportunity was destroyed or created;
- whether an active T with a reachable opportunity was deferred into Hold;
- dead-end flag.

No unseen sixth preview piece is invented.

## Conservative dominance

The residual receives a small pairwise auxiliary only when candidate A is no
worse than B on:

- holes;
- height;
- one-step recoverability;
- immediate lines;
- dead-end risk;
- T tactical value;

and is strictly better on at least one dimension.

This intentionally avoids the rule "fewer holes always wins".


## Bulk-cache speed contract (V1.1 fast-proxy v2)

The first implementation ran path-sensitive reference T-spin reconstruction for
many shortlisted candidates. On Windows this measured only ~2.5 rows/s, which
would make the 20K smoke cache take ~2.2 hours and a 100K cache ~11 hours.

Bulk-cache V2 therefore follows a two-tier contract:

```text
bulk train/cache:
    Fast Reachability + T 3-corner tactical proxy
    NO per-candidate Reference BFS

sampled/final diagnostics:
    path-sensitive Reference T-spin classification
```

Exact-only T feature columns remain in the file schema for compatibility, but
the V1.1 model explicitly neutralizes them. Therefore an already completed
old `shard_00000.npz` can be resumed safely.

The builder now defaults to `--resume` and prints progress/ETA after every GPU
batch. Do not use `--overwrite` if you want to keep an already completed first
shard.

## Phase 1: build a 20K smoke cache

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_1_future_cache ^
  --v0-cache data\tetrio\expert_v0\train_full_fast_s8192 ^
  --cf-cache data\tetrio\expert_v1_cf\train_100k ^
  --checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --output-dir data\tetrio\expert_v1_1_future\train_20k ^
  --rows 20000 ^
  --batch-size 1024 ^
  --workers 16 ^
  --device cuda
```

Validation:

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_expert_v1_1_future_cache ^
  --v0-cache data\tetrio\expert_v0\val_10k ^
  --cf-cache data\tetrio\expert_v1_cf\val_10k ^
  --checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --output-dir data\tetrio\expert_v1_1_future\val_10k ^
  --rows 0 ^
  --batch-size 1024 ^
  --workers 16 ^
  --device cuda
```

Inspect `shortlist_recall` in both manifests. If recall is unexpectedly low,
increase `--top-overall` / `--top-per-branch` before training.

## Collect self-generated recovery pairs

Use already-consumed V1 development seeds, not fresh V1.1 evaluation seeds:

```bat
.venv\Scripts\python.exe -m tetrio.tools.collect_expert_v1_recovery_states ^
  --checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --seeds 9031-9050 ^
  --max-pieces 1000 ^
  --max-events 2000 ^
  --device cuda ^
  --output artifacts\tetrio\expert_v1_recovery_states.jsonl
```

Each pair is generated only when the base Top-1 is structurally/tactically bad
and another shortlisted candidate conservatively dominates it. These remain
pairwise recovery preferences, not oracle absolute labels.

## Train 20K smoke

```bat
.venv\Scripts\python.exe -m tetrio.tools.train_expert_v1_1 ^
  --train-cache data\tetrio\expert_v1_1_future\train_20k ^
  --val-cache data\tetrio\expert_v1_1_future\val_10k ^
  --init-v1 models\tetrio_expert_v1_joint_100k.pt ^
  --recovery-jsonl artifacts\tetrio\expert_v1_recovery_states.jsonl ^
  --recovery-weight 0.25 ^
  --output models\tetrio_expert_v1_1_future_20k.pt ^
  --metrics artifacts\tetrio\expert_v1_1_future_20k_training.json ^
  --epochs 10 ^
  --batch-size 1024 ^
  --device cuda
```

Epoch 0 is the exact frozen-V1 baseline on the same shortlist.

A trained checkpoint is eligible only if validation Top1 remains within 1.5
percentage points of epoch 0. Among eligible epochs the saved model minimizes:

```text
avoidable
+ 0.50 * T opportunity destroyed
+ 0.25 * active-T cashout deferred
```

This is a research selection rule, not a Champion gate.

## Closed-loop V1.1

Use fresh development seeds 9051–9070:

```bat
.venv\Scripts\python.exe -m tetrio.tools.watch_expert_v1_1 ^
  --checkpoint models\tetrio_expert_v1_1_future_20k.pt ^
  --seed 9051 ^
  --max-pieces 5000 ^
  --device cuda ^
  --backend fast
```

The viewer inherits the current V3.4 visible-rotation + history-replay UI.

Headless block:

```bat
.venv\Scripts\python.exe -m tetrio.tools.watch_expert_v1_1 ^
  --checkpoint models\tetrio_expert_v1_1_future_20k.pt ^
  --headless ^
  --seeds 9051-9070 ^
  --max-pieces 5000 ^
  --device cuda ^
  --save-json artifacts\tetrio\expert_v1_1_rollout_9051_9070.json
```

## Recovery pair training semantics

The trainer runs one supervised future-cache phase and one recovery-pair phase
per epoch. The recovery loss is only:

```text
score(dominant recovery) > score(base bad choice) + margin
```

No recovery candidate is declared globally optimal, and no self-generated state
is mixed into the expert listwise labels.

## Scale only after the pilot passes

If V1.1 improves closed-loop avoidable-hole / T-opportunity metrics without a
material validation collapse:

```text
20K smoke
 -> 100K future cache/train
 -> 500K
 -> full corpus
```

Do not inspect the held-out test split while tuning V1.1.
