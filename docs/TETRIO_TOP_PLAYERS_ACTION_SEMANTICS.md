# Top-Player Corpus Hold / Preview Semantics

Board/placement mapping has reached 100% mapped, legal and line-clear agreement
on 5,000 sampled games, with 99.9997% exact continuity on conservative quiet
transitions.

Before imitation learning, the remaining action-state timing must be validated:
specifically whether `hold` and `next` are post-placement state.

The first observed rows strongly suggest that they are.

Example:

```text
row 1 after placing I:
  hold = N
  next = JZSOT...

row 2:
  placed = Z
  hold = J
  next = SOT...
```

This is exactly the standard empty-hold sequence:

```text
active J -> HOLD
J enters hold
next Z becomes active and locks
preview consumes J and Z
```

The validator checks every adjacent transition in a deterministic game sample
against three modes:

- `no_hold`
- `hold_swap`
- `hold_empty`

Run:

```bat
.venv\Scripts\python.exe -m unittest tetrio.tests.test_top_players_action_semantics -v

.venv\Scripts\python.exe -m tetrio.tools.validate_top_players_action_semantics ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --games 5000
```

A very high classified rate confirms that previous-row `next[0]` is the next
active piece and lets the corpus builder derive an explicit `use_hold` label.

Do not train on `placed` alone before this timing relationship is validated.
