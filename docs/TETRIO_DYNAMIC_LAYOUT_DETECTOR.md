# TETR.IO Dynamic Layout Detector

This is the first stage of the visual state pipeline.

It does **not** read the board contents yet and it does not control TETR.IO.

## V0 scope

```text
full screenshot
    ↓
geometry-first 10×20 playfield detection
    ↓
username strip OCR
    ↓
SELF / OPPONENT / UNKNOWN
    ↓
debug overlay + JSON report
```

The playfield detector does not depend on a fixed screen location. It uses the
10-column / 20-row grid geometry and therefore supports changing browser size
and the supplied single-player / Quick Play / 1v1 layouts.

Tiny Quick Play spectator boards are intentionally ignored in V0. Only the
full-size active board and full-size 1v1 opponent board are targets.

In a single-full-size-board layout, that board is safely SELF.

In a multi-full-size-board layout, the resolver fails closed unless the OCR
username matches `--self-username` strongly enough. It will report UNKNOWN
rather than guess.

## Dependencies

Inside the project's existing venv:

```bat
.venv\Scripts\python.exe -m pip install opencv-python rapidocr-onnxruntime
```

`numpy` is already an OpenCV dependency.

## Put screenshots in one folder

```bat
mkdir data\tetrio\vision\screenshots
```

Then copy screenshots there.

## Run board-only first

This verifies geometry without introducing OCR:

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_vision_layout ^
  --input data\tetrio\vision\screenshots ^
  --no-ocr ^
  --output-dir artifacts\tetrio\vision\layout_debug_no_ocr
```

Expected:
* single-player / Quick Play active view: one full-size board;
* TETRA LEAGUE 1v1: two full-size boards.

## Run with username role resolution

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_vision_layout ^
  --input data\tetrio\vision\screenshots ^
  --self-username MAYSHOWGUNMORE77 ^
  --output-dir artifacts\tetrio\vision\layout_debug
```

Outputs:
* `*_layout.png`: playfield boxes and role labels;
* `*_name_N.png`: exact OCR crop used for each board;
* `layout_report.json`: bboxes, scores, OCR and SELF-match scores.

## Gate before board-state reading

All supplied layouts must satisfy:

```text
single player:
  exactly one full-size playfield
  role = SELF

Quick Play:
  exactly one full-size active playfield
  tiny spectator boards ignored
  role = SELF

TETRA LEAGUE 1v1:
  exactly two full-size playfields
  MAYSHOWGUNMORE77 = SELF
  other full-size board = OPPONENT
```

If username OCR fails, inspect the generated `*_name_N.png` crop first. Do not
weaken the role threshold just to force a pass.

Only after this gate passes should `BoardReader` be added.
