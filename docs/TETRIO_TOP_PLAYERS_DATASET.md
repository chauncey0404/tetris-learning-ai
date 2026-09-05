# TETR.IO Top-Player Dataset Ingest

This tool profiles and converts the Kaggle placement-level TETR.IO dataset
whose observed columns are:

`game_id, subframe, won, playfield, x, y, r, placed, hold, next, cleared,
garbage_cleared, attack, t_spin, btb, combo, immediate_garbage,
incoming_garbage, rating, glicko, glicko_rd`

## Local data layout

Keep raw and large processed data outside Git:

```text
data/
└─ tetrio/
   ├─ raw/
   │  ├─ data.csv
   │  └─ data.csv.xz
   └─ processed/
      ├─ top_players_s1.parquet
      ├─ top_players_s1_profile.json
      └─ top_players_s1_parquet_manifest.json
```

Recommended `.gitignore` rules:

```gitignore
data/tetrio/raw/
data/tetrio/processed/*.parquet
```

The JSON profile/manifest can remain trackable if desired.

## Dependency

DuckDB is used because its CSV scanner and Parquet writer are multi-threaded
and can efficiently use the host CPU without loading the full 1.4+ GB CSV
through Python objects.

```bat
.venv\Scripts\python.exe -m pip install duckdb
```

## Profile only

```bat
.venv\Scripts\python.exe -m tetrio.tools.prepare_top_players_dataset ^
  --input data\tetrio\raw\data.csv ^
  --output-dir data\tetrio\processed ^
  --threads 20 ^
  --profile-only
```

## Profile and convert to Parquet

```bat
.venv\Scripts\python.exe -m tetrio.tools.prepare_top_players_dataset ^
  --input data\tetrio\raw\data.csv ^
  --output-dir data\tetrio\processed ^
  --threads 20
```

The conversion validates that the Parquet row count exactly matches the CSV.

## Data-leakage rule

Do not automatically feed every CSV column into a neural network.

Until row semantics are independently verified:

- likely pre-action/context fields: `playfield`, `hold`, `next`,
  `incoming_garbage`, `immediate_garbage`, `btb`, `combo`
- candidate/action labels to investigate: `placed`, `x`, `y`, `r`
- post-action/future information that must not silently become model input:
  `won`, `cleared`, `garbage_cleared`, `attack`, `t_spin`
- `rating`, `glicko`, `glicko_rd` are player-skill metadata and are better
  treated as filtering/weighting metadata than game-state inputs unless a
  later experiment explicitly proves otherwise.

The next research step after profiling is continuity/semantic validation,
not training.
