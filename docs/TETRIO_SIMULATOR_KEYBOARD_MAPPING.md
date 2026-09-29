# Simulator keyboard mapping

```text
MOVE_LEFT    -> A
MOVE_RIGHT   -> D
SOFT_DROP    -> W
HARD_DROP    -> S
ROTATE_CCW   -> Left Arrow
ROTATE_CW    -> Right Arrow
ROTATE_180   -> Up Arrow
HOLD         -> Left Shift
```

The controller uses Windows `SendInput` scan codes. No extra keyboard package
is required.

Test:

```bat
.venv\Scripts\python.exe -m unittest tetrio.tests.test_keyboard_mapping -v
```

Dry run:

```bat
.venv\Scripts\python.exe -m tetrio.tools.test_keyboard_mapping
```

Live test:

```bat
.venv\Scripts\python.exe -m tetrio.tools.test_keyboard_mapping --live
```

If the simulator misses taps:

```bat
.venv\Scripts\python.exe -m tetrio.tools.test_keyboard_mapping ^
  --live ^
  --tap-ms 40 ^
  --gap-ms 220
```
