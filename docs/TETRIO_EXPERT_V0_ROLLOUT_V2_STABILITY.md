# Expert-v0 Rollout V2 — V3.4 Hold/Next + Stability Audit

This patch addresses two separate observations from autonomous rollout.

## 1. HOLD / NEXT UI

The previous rollout viewer used text chips for Active / Hold / Next. The older
project viewer (`watch_models_v3_4.py`) used a dedicated HOLD preview box and a
vertical NEXT queue of miniature tetrominoes.

The rollout viewer now restores that visual grammar:

```text
HOLD
[ mini piece ]

NEXT
[ mini piece ]
[ mini piece ]
[ mini piece ]
[ mini piece ]
[ mini piece ]
```

ACTIVE is also shown as a mini-piece box beside the queue.

## 2. Strategy instability / apparently pointless holes

Do not immediately conclude that every odd placement is a human mistake copied
from the replay corpus. There are at least four plausible causes:

1. genuine human label noise / misdrops;
2. battle-specific intent such as T-spin setup or garbage interaction that the
   current v0 state does not fully observe;
3. autonomous distribution shift: one imperfect action creates states unlike
   the expert state distribution and errors can compound;
4. the v0 Hold architecture is staged: a state-only Hold head chooses the
   branch first, then the scorer can rank only that branch.

Therefore this patch adds a diagnostic stability audit without changing the
policy.

For every self-generated move it computes:

```text
holes before
holes after chosen move
minimum holes among all candidates in the selected Hold branch
```

It flags:

```text
HOLE +N
```

when the chosen move creates new holes, and:

```text
HOLE +N  AVOIDABLE
```

when another legal candidate in the same branch could have avoided those new
holes.

The GUI shows the network's highest-scoring safer alternative and the score gap.
The rollout JSON records every avoidable-hole event.

Headless output now reports per game:

```text
newHole=<count>
avoidable=<count>
```

and aggregate fields include:

```text
mean_hole_creation_rate
mean_avoidable_hole_rate
```

This is diagnostic only. It does not override the model, add a heuristic safety
filter, or change the trained checkpoint.

Recommended next measurement:

```bat
.venv\Scripts\python.exe -m tetrio.tools.watch_expert_v0 ^
  --checkpoint models\tetrio_expert_v0_full.pt ^
  --headless ^
  --seeds 9001-9020 ^
  --max-pieces 5000 ^
  --device cuda ^
  --backend fast ^
  --reference-audit-every 250 ^
  --save-json artifacts\tetrio\expert_v0_rollout_9001_9020_stability.json
```

If avoidable-hole events are frequent, the next model step should be Expert v1:
joint no-hold/hold candidate scoring plus controlled data-quality / structural
regularization experiments. Do not tune against the held-out test split.
