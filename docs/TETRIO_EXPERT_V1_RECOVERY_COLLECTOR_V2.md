# Expert-v1 Recovery Collector V2

The original recovery collector advanced one seed at a time:

```text
one state
 -> CPU candidate generation
 -> tiny GPU inference
 -> serial future analysis
 -> repeat
```

That left most i5-13500 cores idle and fed the RTX 5070 with tiny inference
batches.

V2 advances many independent V1 trajectories in lockstep:

```text
up to 20 live seeds
        |
        v
16-process CPU candidate generation
        |
        v
ONE merged GPU batch for every candidate from all live states
        |
        v
shortlist each state
        |
        v
16-process CPU future-feature analysis
        |
        v
dominance / recovery-pair extraction
        |
        v
commit ORIGINAL V1 Top-1 and repeat
```

The collector deliberately commits the base V1 choice, not the recovery
candidate. This keeps the collected trajectory on the original V1 policy and
prevents the labels from changing the state distribution during collection.

Bulk future analysis uses the Speed-V2 contract:

```text
Fast Reachability + T tactical proxy
no per-candidate path-sensitive Reference T BFS
```

Reference fallback is used only if Fast returns no candidates for a branch.

## Recommended command

For the first V1.1 pilot, 750 conservative pairs are enough:

```bat
.venv\Scripts\python.exe -m tetrio.tools.collect_expert_v1_recovery_states ^
  --checkpoint models\tetrio_expert_v1_joint_100k.pt ^
  --seeds 9031-9050 ^
  --max-pieces 1000 ^
  --max-events 750 ^
  --workers 16 ^
  --state-batch 20 ^
  --progress-every 256 ^
  --device cuda ^
  --output artifacts\tetrio\expert_v1_recovery_states.jsonl
```

Progress example:

```text
states=2048 events=181/750 event_rate=8.84%
rate=27.4 states/s meanK=69.2 shortK=9.1 active=20 ETA=21s
```

The actual throughput depends on the self-generated boards.

The JSONL is line-buffered. If interrupted with Ctrl+C, already-written events
remain valid, although rerunning without a separate output path starts a fresh
collection.

## Hardware behavior

CPU should be substantially busier because both reachability phases use the
ProcessPool. GPU utilization will appear in bursts rather than as continuous
training-style 90-100% load because the pipeline alternates CPU search and GPU
ranking. Each GPU burst is now a merged batch across many independent states,
rather than one tiny state at a time.

## Event semantics

A pair is saved only when:

1. Base V1 Top-1 either adds holes or destroys a near-term T opportunity; and
2. another shortlisted legal candidate conservatively dominates it on the
   structural/tactical feature set.

The saved pair means only:

```text
dominant recovery > this particular bad V1 choice
```

It does not declare the recovery candidate globally optimal.
