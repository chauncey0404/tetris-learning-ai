# V9.0 Battle Movement Engine

Status: initial ruleset-aware movement implementation.

## Scope

V9.0 is intentionally isolated from the validated V8 single-player pipeline.
It does not change `PlacementAction(rotation, x, use_hold)`, V8 checkpoints,
state243/candidate215, Teacher, replay, or DDQN code.

V9 adds:

- game-specific movement rulesets;
- native SRS rotation boxes;
- standard SRS 90-degree kicks;
- TETR.IO-oriented SRS+ I-piece 90-degree kicks;
- TETR.IO native 180-degree kick handling;
- exact movement path metadata;
- BFS reachability;
- hard-drop landing enumeration;
- preservation of last rotation / kick index for later spin classification.

## Coordinate contract

- board storage: row-major NumPy array;
- x increases right;
- y increases down;
- kick tables are converted from conventional SRS `+y = up` at the rotation-system boundary;
- piece state x/y is the top-left corner of the native rotation box;
- I uses a 4x4 box, O 2x2, JLSTZ 3x3;
- canonical rotation numbers remain V8-compatible: 0, 1=CW90, 2=180, 3=CW270.

## Ruleset separation

`GUIDELINE_SRS`:

- 10x40 internal board;
- 20 visible rows;
- standard SRS 90-degree kicks;
- no native 180 action.

`TETRIO_DEFAULT`:

- 10x40 internal board;
- 20 visible rows;
- SRS+;
- native 180 action enabled.

TETR.IO room options can change kick tables and 180 permission, so the current
factory is deliberately named a *default movement* profile rather than a
universal TETR.IO rules object.

## Important non-goals in V9.0

Not implemented yet:

- T-Spin Full/Mini classification;
- All-Mini+/T-Spins+ selectable spin policy;
- lock delay / move reset limits;
- gravity timing / DAS / ARR / SDF;
- hold/queue/bag battle state;
- attack tables;
- combo/B2B/surge;
- garbage queue/cancel;
- top-out/lockout policy;
- self-play or neural-network changes.

Those belong to later V9 phases after movement parity is established.

## Verification

Run from project root:

```bat
.venv\Scripts\python.exe -m unittest tests.test_v9_rotation_systems tests.test_v9_reachability -v
```

No full V8 retraining or qualification run is required because V9.0 adds a new
package and does not modify V8 runtime files.
