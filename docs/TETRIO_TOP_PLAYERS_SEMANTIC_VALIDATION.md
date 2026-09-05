# TETR.IO Top-Player Corpus Semantic Validation

The historical top-player corpus has been converted to Parquet successfully:

- rows: 7,716,524
- games: 76,692
- CSV: ~1.389 GiB
- Parquet: ~138 MiB

Before imitation learning, placement semantics must be validated against game
continuity rather than inferred from column names.

## Observed representation hypothesis

The first corpus rows strongly support:

- `playfield` is the board before the row's `placed` action.
- The string is flattened in 10-cell rows from the floor upward.
- `N` represents empty.
- trailing empty cells and rows are omitted.
- `r`: N/E/S/W correspond to 0/90/180/270 clockwise orientations.
- dataset `x/y` appear to be SRS-origin-style coordinates.
- the current shared-engine conversion hypothesis is:
  - `PieceState.x = x - 1`
  - `PieceState.y = 38 - y`

The validator tests this across sampled complete games.

## Unit tests

```bat
.venv\Scripts\python.exe -m unittest tetrio.tests.test_top_players_dataset_semantics -v
```

The two first observed transitions are included as fixed regression examples.

## Infer coordinate convention

Run a deterministic sample of 250 games and score nearby coordinate/orientation
hypotheses:

```bat
.venv\Scripts\python.exe -m tetrio.tools.validate_top_players_semantics ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --games 250 ^
  --infer-coordinates
```

The expected best hypothesis is currently:

```text
x_offset = -1
y_base   = 38
rotation = cw
```

Do not make this a training contract until the corpus confirms it.

## Larger validation

After the 250-game pass:

```bat
.venv\Scripts\python.exe -m tetrio.tools.validate_top_players_semantics ^
  --input data\tetrio\processed\top_players_s1.parquet ^
  --games 5000 ^
  --x-offset -1 ^
  --y-base 38 ^
  --rotation-map cw
```

The validator reports:

- placement legality
- reported-vs-simulated line-clear agreement
- exact next-playfield continuity
- a conservative "quiet" continuity rate where both adjacent rows report zero
  immediate/incoming garbage
- breakdown by piece and rotation
- mismatch examples for diagnosis

## Important interpretation

This dataset is historical expert-play data. A placement/board continuity PASS
can validate expert state/action semantics, but it must not be used as proof of
current TETR.IO Season 2 garbage/B2B/Surge rules.

Do not train until:
1. coordinate/orientation mapping is validated;
2. board continuity is understood;
3. any remaining mismatch classes are explained;
4. post-action fields are excluded from model inputs.
