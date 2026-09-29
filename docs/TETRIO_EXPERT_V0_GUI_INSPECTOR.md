# Expert-v0 Graphical Held-out Inspector — V3.4 Style

This viewer is the TETR.IO Expert-v0 equivalent of the project's older
`watch_models.py` V3.4 viewer. It keeps the Expert-v0 checkpoint/cache contract
but restores the old visual language:

- Guideline tetromino colors: I cyan, J blue, L orange, O yellow, S green,
  T purple, Z red;
- V3.4-style glossy block rendering;
- colored ACTIVE / HOLD / NEXT chips;
- clickable rounded hotkey/control chips at the bottom;
- 1–9 speed presets and +/- speed controls.

## Why the first inspector was gray

The validated Expert-v0 cache intentionally packs board **occupancy** only. It
is sufficient for model evaluation, but it discards the historical tetromino ID
that the older single-player viewer received directly from the live Gym board.

This version optionally reads the held-out source parquet **only for rendering**.
If the source retained `playfield` or piece-ID board data, exact historical
piece colors are restored. Model inference, rankings, metrics, and cache
semantics are unchanged. If source identity is unavailable, existing locked
cells remain neutral gray while the inspected candidate placement is colored.

Color sources are tried in this order:

```text
data\tetrio\expert\top_players_s1_test.parquet
data\tetrio\processed\top_players_s1.parquet
```

The second source is the processed historical corpus and is used as a fallback
when the expert split only retained binary occupancy.

Disable the optional color-source read with:

```text
--no-source-colors
```

## Run

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_expert_v0_gui ^
  --checkpoint models\tetrio_expert_v0_full.pt ^
  --cache data\tetrio\expert_v0\test_full_fast_s8192 ^
  --filter disagreement ^
  --max-cases 300 ^
  --device cuda
```

## Controls

```text
Space        Play / Pause autoplay
Left/Right   Previous / next case
PgUp/PgDn    -10 / +10 cases
Home/End     First / last case
1..9         Speed presets
- / +        Slower / faster
D            Details
S            Screenshot
Esc / Q      Quit
```

The controls are also clickable in the V3.4-style bottom bar.
