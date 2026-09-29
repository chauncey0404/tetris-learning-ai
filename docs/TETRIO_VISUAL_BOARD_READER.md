# TETR.IO Visual Board Reader V0

This stage runs after the Dynamic Layout Detector / phase-lock gate.

It converts each detected 10×20 board into a conservative static visual matrix:

```text
. = EMPTY
# = visible colored MINO
g = GHOST_CANDIDATE
n = NEUTRAL_CANDIDATE
? = UNKNOWN
```

Important: `#` is not yet the final locked board. A currently falling piece is
also colored, so it also appears as `#`. The next Temporal Tracker stage will
separate active/falling cells from the locked stack by comparing frames.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_vision_board ^
  --input data\tetrio\vision\screenshots ^
  --self-username MAYSHOWGUNMORE77 ^
  --output-dir artifacts\tetrio\vision\board_debug
```

Outputs:

```text
*_board.png
board_report.json
```

Overlay colors:

```text
green   = visible colored mino (#)
cyan    = ghost candidate (g)
magenta = neutral/special candidate (n)
red     = unknown (?)
```

V0 gate:

1. every clearly colored mino is green;
2. ordinary empty cells are not green;
3. ghost outlines are not promoted to green;
4. uncertain animation/special cells stay diagnostic rather than guessed.

Do not train on this output yet.


## Colored fill-ratio guard

V0 originally classified a cell from its 90th-percentile brightness/chroma.
That correctly found minos, but a large yellow countdown digit (`3/2/1`) can
also make those percentiles high.

The reader now also measures how much of the cell interior is actually filled
with saturated color:

```text
genuine colored minos in supplied screenshots: >= ~0.93
countdown false positives:                    ~0.10 .. 0.43
MINO promotion threshold:                     >= 0.70
```

Bright/saturated cells below the fill threshold become:

```text
t = TRANSIENT_OVERLAY_CANDIDATE
```

They are never treated as board occupancy.
