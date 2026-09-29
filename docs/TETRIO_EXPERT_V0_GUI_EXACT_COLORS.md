# Expert-v0 GUI Exact Historical Colors

This patch fixes the source-color reconstruction in the V3.4-style Expert-v0
held-out inspector.

## Root cause

The historical `playfield` is a floor-up flat cell string. `N` means empty;
tetromino letters preserve piece identity; other occupied source codes represent
non-tetromino cells such as garbage.

The previous GUI decoder incorrectly filtered the string down to only
`NIJLOSTZ`. When a garbage code occurred, that cell was deleted from the
sequence and every later colored cell shifted to the wrong board position.

The model/cache was never affected: Expert-v0 uses validated binary occupancy.
This was a display-only bug.

## New rendering contract

- Cyan = I
- Blue = J
- Orange = L
- Yellow = O
- Green = S
- Purple = T
- Red = Z
- Dark gray with a center mark = exact non-tetromino source cell / garbage
- Light silver = occupied in binary cache but historical identity unavailable
- Black = empty

The decoder now preserves every serialized cell position. The known historical
J/L extractor naming swap is also canonicalized before selecting display color.

The candidate/result boards propagate the exact historical IDs through line
clears and paint the newly placed tetromino with its canonical piece color.

The console now reports:

```text
Color source : playfield exact=... piece_cells=... garbage_cells=... unknown_cells=...
```

For a fully recovered sample, `unknown_cells` should normally be zero.

This remains visualization-only and does not change evaluation metrics,
candidate ranking, checkpoint weights, or training data.
