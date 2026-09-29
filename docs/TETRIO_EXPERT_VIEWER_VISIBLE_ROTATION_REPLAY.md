# Expert v0/v1 Viewer — Visible Rotation + History Replay

## Bug fixed

TETR.IO entry is in hidden rows. The previous V3.4 reconstruction performed
rotation during:

```text
spawn y≈17 -> y≈20
```

so the user saw the piece only after rotation was already complete.

The visual animation now has four explicit phases:

```text
1. hidden entry -> visible top row, spawn orientation
2. visible rotation
3. visible horizontal adjustment
4. accelerating fall -> exact selected landing
```

For readability:

```text
r1: 0 -> 1
r2: 0 -> 1 -> 2
r3: 0 -> 3
```

Thus a T going to r3 visibly makes one CCW quarter-turn, and r2 visibly makes
two quarter-turns.

This remains display-only and never changes the model-selected final landing.

## History Next now replays animation

Previously:

```text
Prev -> previous committed board
Next -> instantly jump to next committed board
```

Now:

```text
Prev -> previous committed board
Next -> replay that historical placement's visible rotation/fall
        -> then reveal next committed board
```

Pressing Next a second time during the replay immediately finishes it.

The history replay does not rewind or modify RNG/model state.

Status text distinguishes:

```text
REPLAY FALLING/ROTATING
FALLING/ROTATING
STEP FALLING/ROTATING
```
