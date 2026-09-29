# Expert-v0/v1 Viewer — V3.4 Falling Animation + Prev/Next History

This patch restores the viewer behavior that existed in the old
`watch_models_v3_4.py` lineage.

## Falling animation

The policy still chooses only the final legal landing. The viewer reconstructs
a purely visual sequence:

```text
TETR.IO entry
  -> rotate / move horizontally near the top
  -> accelerating vertical fall
  -> exact selected landing
  -> commit the real rollout state
```

Animation never changes the selected candidate, reachability, queue, Hold, or
model score.

Duration uses the same V3.4 scaling:

```text
max(0.12, min(1.15, 2.20 / speed))
```

During a Hold move, HOLD/NEXT are displayed as the post-Hold / pre-lock visual
state. After lock, Active/Hold/Next advance exactly once.

Use `--no-fall-animation` to disable the visual reconstruction.

## Prev / Next history

Every committed state is saved as a lightweight display-only frame.

```text
Left Arrow / Prev
    review previous committed state

Right Arrow / Next
    if reviewing history: move one state forward
    if already at live edge: animate one new model placement
    if the current manual animation is already running: finish/commit it
```

History review never rewinds the model or RNG. Pressing Play while reviewing
returns to the latest live state before autoplay resumes.

Controls:

```text
Space       Play / Pause
Left        Previous committed state
Right       Next history state / one-piece animated step
R           Reset same seed
N           Next seed
1..9        Speed presets
+ / -       Fine speed adjustment
D           Detail
S           Screenshot
Esc / Q     Quit
```

This patch also fixes the V1 panel/window title so it says Expert v1 rather than
inheriting the v0 label.
