# TETR.IO Historical Expert Corpus

The 5,000-game validation established:

- placement mapping: 100% mapped
- geometry legality: 100%
- line-clear agreement: 100%
- quiet board continuity: 99.9997%
- hold/preview classification: 99.9996%
- unique hold label: 96.4637%
- ambiguous hold label: 3.5359%

This is sufficient to build a first leakage-safe expert placement corpus.

## Important state timing

For row `t`:

```text
board_before = row[t].playfield

active_piece = row[t-1].next[0]
hold_piece   = row[t-1].hold
preview      = row[t-1].next[1:]
```

The row's own `hold` and `next` are post-placement state.

## Ambiguous hold transitions

About 3.5% of transitions are observationally ambiguous between hold/no-hold
because identical piece identities can make both actions produce the same
observable state.

They are NOT discarded.

They are written to:

```text
data/tetrio/expert/top_players_s1_ambiguous_hold.parquet
```

Their placement geometry remains useful. Later training may include them while
masking the hold-head loss.

The tiny unmatched set is quarantined separately and must not enter training.

## Leakage-safe split

Train/validation/test are split by `game_id`, never by individual placement.

Default split:

```text
train 90%
val    5%
test   5%
```

The builder verifies that there is zero game overlap between the three splits.

## Build

```bat
.venv\Scripts\python.exe -m tetrio.tools.build_top_players_expert_corpus ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --output-dir data\tetrio\expert ^
  --threads 20
```

Outputs:

```text
data/tetrio/expert/
├─ top_players_s1_train.parquet
├─ top_players_s1_val.parquet
├─ top_players_s1_test.parquet
├─ top_players_s1_ambiguous_hold.parquet
├─ top_players_s1_unmatched.parquet
└─ top_players_s1_expert_manifest.json
```

## Validated model-state contract

Only these are currently approved as pre-action state:

- `board_before`
- `active_piece`
- `hold_piece`
- `preview_queue`

Approved action labels:

- `placed_piece`
- `final_x`
- `final_y`
- `final_rotation`
- `use_hold` only when `hold_label_valid=true`

Do NOT silently use these as model-state inputs yet:

- combo
- B2B
- incoming/immediate garbage
- attack
- cleared lines
- T-spin label
- win result
- rating/Glicko

The latter fields are retained as metadata/outcomes for later validation,
filtering and auxiliary objectives.

## Next gate

Before GPU pretraining, sample the generated corpus and verify that each expert
final placement can be matched to a candidate produced by the current TETR.IO
reachability engine. That gate ensures the imitation target and the actual V9
action space are the same.
