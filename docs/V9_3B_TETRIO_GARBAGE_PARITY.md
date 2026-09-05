# V9.3B — TETR.IO Garbage Replay Parity Harness

This stage does **not** invent missing TETR.IO ranked garbage constants.
It creates a strict evidence path from current `.ttrm` replays (or controlled
captures) to the V9.3A garbage transport reference.

## Why this exists

TETR.IO multiplayer replays are saved as `.ttrm` files. Community replay
 tooling treats these files as JSON and exposes multiplayer rounds through
`data[round].replays`. The input/event payload still needs engine simulation
or direct observation to derive higher-level garbage facts, so V9 does not
silently assume undocumented event names or fields.

## Tools

### 1. Inventory a real replay

```bat
.venv\Scripts\python.exe -m tetrio.tools.inspect_ttrm "C:\path\match.ttrm" ^
  --output "artifacts	etrio\parity\match_inventory.json"
```

The inspector reports:

- top-level keys
- round/replay/frame structure
- recurring `type` / `event` / `name` / `action` values
- JSON paths containing garbage-related key names

It never interprets a matching field as a semantic fact by name alone.

### 2. Normalize observed garbage facts

Create a small JSON oracle trace from a replay/capture whose events have been
verified. Example:

```json
{
  "events": [
    {
      "kind": "send",
      "player": "A",
      "target": "B",
      "frame": 100,
      "packets": [4],
      "holes": [3],
      "expected": {"pending": 4, "active": 0, "active_frames": [120]}
    },
    {
      "kind": "cancel",
      "player": "B",
      "frame": 110,
      "lines": 2,
      "expected": {"cancelled": 2, "pending": 2, "active": 0}
    },
    {
      "kind": "advance",
      "player": "B",
      "frame": 120,
      "expected": {"pending": 2, "active": 2}
    },
    {
      "kind": "tank",
      "player": "B",
      "frame": 120,
      "expected": {"inserted": 2, "pending": 0, "bottom_garbage_holes": [3, 3]}
    }
  ]
}
```

### 3. Run strict parity

```bat
.venv\Scripts\python.exe -m tetrio.tools.validate_garbage_parity ^
  "artifacts	etrio\parity\match_trace.json"
```

Any mismatch returns exit code 1 and identifies the exact event/field.

## V9.3B freeze gate

Do not freeze current ranked garbage constants until current Tetra League
replay/capture evidence covers at least:

1. send -> queue arrival
2. pre-activation cancellation
3. activation timing
4. tank/insertion timing
5. packet boundary behavior
6. garbage cap behavior
7. hole changes between attacks
8. within-attack messiness, if any
9. blocking / clutch ordering
10. top-out ordering

The harness intentionally remains fail-closed for active packets whose hole
layout has not been observed.
