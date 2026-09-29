# Visual HOLD / NEXT Reader

HOLD is gray in the supplied TETR.IO layouts, so the reader does not try to
identify HOLD by color. It extracts the four-square silhouette and matches that
geometry to I/O/T/S/Z/J/L.

NEXT is colored. Color is used only to separate the five preview objects from
the dark panel. Identity is still determined from the four-cell geometry.

Run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_vision_pieces ^
  --input data\tetrio\vision\screenshots ^
  --self-username MAYSHOWGUNMORE77 ^
  --output-dir artifacts\tetrio\vision\piece_debug
```

Example:

```text
HOLD=Z NEXT=['T', 'I', 'L', 'J', 'S'] complete=True
```

An empty HOLD is returned as `None`.

This gives the model `hold_piece` and `preview_queue`. The falling active piece
is intentionally left for the temporal tracker.
