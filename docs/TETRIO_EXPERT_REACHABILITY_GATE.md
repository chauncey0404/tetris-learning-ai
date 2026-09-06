# TETR.IO Expert Placement ↔ Reachability Gate

This is the final action-space gate before expert imitation pretraining.

The historical corpus has already passed:

- 7,716,524 source placements
- piece/orientation mapping validation
- 100% placement legality on a 5,000-game sample
- 100% line-clear agreement
- 99.9997% quiet board continuity
- 99.9996% hold/preview classification
- leakage-safe game-level train/val/test split

This gate asks a different question:

> Is the expert's final canonical placement actually produced by the current
> TETR.IO SRS+/180 reachability engine from the same pre-action board?

## Hold handling

The expert corpus stores the validated pre-action state and `hold_mode`.

The candidate piece is derived as:

```text
no_hold:
    active_piece

hold_swap:
    hold_piece

hold_empty:
    preview_queue[0]
```

This derived piece must equal `placed_piece` before reachability is evaluated.

## Candidate match

The gate starts the selected piece at:

```python
TETRIO_MOVEMENT.spawn_state(piece)
```

and calls the existing reference engine:

```python
enumerate_reachable_placements(
    board,
    start,
    TETRIO_MOVEMENT,
)
```

The primary contract is exact canonical landing geometry:

```text
(piece, x, y, rotation)
```

An occupied-cell-only match is reported separately and does not count as PASS,
because the training action label uses canonical `PieceState` geometry.

## Fast gate

Use the TEST split so this validation remains separate from model fitting:

```bat
.venv\Scripts\python.exe -m unittest ^
  tetrio.tests.test_expert_reachability_gate ^
  -v

.venv\Scripts\python.exe -m tetrio.tools.validate_expert_reachability ^
  --input data\tetrio\expert\top_players_s1_test.parquet ^
  --per-group 20 ^
  --workers 16
```

The sample is deterministic and stratified by:

```text
placed_piece × hold_mode × final_rotation
```

so rare hold-empty and orientation groups receive coverage.

## Larger gate

Only after the fast gate is clean:

```bat
.venv\Scripts\python.exe -m tetrio.tools.validate_expert_reachability ^
  --input data\tetrio\expert\top_players_s1_test.parquet ^
  --per-group 100 ^
  --workers 16
```

## Spin-labelled rows

The tool also reports whether an exact geometry has a candidate path whose
pre-hard-drop state ends in rotation, and whether that rotation-ending path has
zero hard-drop distance.

That is a diagnostic only. Exact TETR.IO spin-label parity still belongs to a
separate spin-path gate using the game's spin classifier; it is not silently
treated as proven by geometry reachability.

## PASS contract

The current strict gate requires:

```text
hold semantics = 100%
exact canonical geometry = 100%
```

Anything else is a diagnostic failure and must be explained before pretraining.
