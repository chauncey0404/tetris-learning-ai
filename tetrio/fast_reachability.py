from __future__ import annotations

from collections import deque

import numpy as np

from tetrio.reachability import tetrio_spawn_state
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.tetrominoes import native_matrix
from tetris_ai.core.types import MoveAction, PieceState


# Expert-v0 only needs the unique reachable landing geometries. The reference
# engine deliberately keeps last-action / kick metadata in its search identity
# for spin classification. That is correct but much more expensive.
#
# Movement legality itself depends only on (piece, x, y, rotation, board).
# Therefore a geometry-only visited set is exact for the *set of reachable
# landing geometries*. This backend must not be used as a spin-path oracle.
#
# Inner-loop collision checks use 40 integer row bitmasks and precomputed
# occupied-cell offsets/kicks, avoiding numpy conversion, dataclasses, native
# matrix creation and RotationTrace construction per attempted move.

_WIDTH = int(TETRIO_MOVEMENT.width)
_HEIGHT = int(TETRIO_MOVEMENT.height)

if (_HEIGHT, _WIDTH) != (40, 10):
    raise RuntimeError(
        "TETR.IO fast reachability v0 is specialized for the validated 40x10 board"
    )

_PIECES = ("I", "O", "T", "S", "Z", "J", "L")
_ROTATIONS = range(4)

_CELL_OFFSETS: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {}
for _piece in _PIECES:
    for _rotation in _ROTATIONS:
        _shape = native_matrix(_piece, _rotation)
        _ys, _xs = np.nonzero(_shape)
        _CELL_OFFSETS[(_piece, _rotation)] = tuple(
            (int(x), int(y)) for y, x in zip(_ys, _xs)
        )

# Preserve the ruleset's configured movement action set, but precompute every
# rotation target + ordered kick table once.
_ACTIONS = tuple(TETRIO_MOVEMENT.movement_actions)
_ROTATION_TABLE: dict[
    tuple[str, int, MoveAction],
    tuple[int, tuple[tuple[int, int], ...]],
] = {}
for _piece in _PIECES:
    for _rotation in _ROTATIONS:
        for _action in _ACTIONS:
            if not _action.is_rotation:
                continue
            _target, _tests = TETRIO_MOVEMENT.rotation_system.kick_tests(
                _piece,
                _rotation,
                _action,
            )
            _ROTATION_TABLE[(_piece, _rotation, _action)] = (
                int(_target) % 4,
                tuple((int(dx), int(dy)) for dx, dy in _tests),
            )

# Tiny immutable lookup, shared by all calls.
_BIT_WEIGHTS = (1 << np.arange(_WIDTH, dtype=np.uint16))


def board_to_row_masks(board: np.ndarray) -> tuple[int, ...]:
    """Convert a validated 40x10 board to ten-bit integer row masks."""
    arr = np.asarray(board)
    if arr.shape != (_HEIGHT, _WIDTH):
        raise ValueError(
            f"Expected TETR.IO board shape {(_HEIGHT, _WIDTH)}, got {arr.shape}"
        )
    binary = (arr != 0).astype(np.uint16, copy=False)
    masks = binary @ _BIT_WEIGHTS
    return tuple(int(v) for v in masks)


def _can_place(
    row_masks: tuple[int, ...],
    piece: str,
    rotation: int,
    x: int,
    y: int,
) -> bool:
    for dx, dy in _CELL_OFFSETS[(piece, rotation)]:
        xx = x + dx
        yy = y + dy
        if xx < 0 or xx >= _WIDTH or yy >= _HEIGHT:
            return False
        if yy >= 0 and (row_masks[yy] & (1 << xx)):
            return False
    return True


def _try_rotate(
    row_masks: tuple[int, ...],
    piece: str,
    rotation: int,
    x: int,
    y: int,
    action: MoveAction,
) -> tuple[int, int, int] | None:
    target, tests = _ROTATION_TABLE[(piece, rotation, action)]
    for dx, dy in tests:
        nx = x + dx
        ny = y + dy
        if _can_place(row_masks, piece, target, nx, ny):
            return nx, ny, target
    return None


def _hard_drop_y(
    row_masks: tuple[int, ...],
    piece: str,
    rotation: int,
    x: int,
    y: int,
) -> int:
    yy = y
    while _can_place(row_masks, piece, rotation, x, yy + 1):
        yy += 1
    return yy


def enumerate_tetrio_reachable_geometries_fast(
    board: np.ndarray,
    piece: str,
    *,
    max_states: int = 10_000,
) -> list[PieceState]:
    """Return every unique reachable TETR.IO hard-drop landing geometry.

    Contract:
    - same unique landing geometry set as the path-sensitive reference engine;
    - TETR.IO production entry state (including the validated one-row raise);
    - SRS+ CW/CCW/180 kick order from the production rotation system;
    - deterministic ordering by (rotation, x, y).

    This intentionally does *not* preserve path / last-rotation metadata and
    must not be used for T-spin/mini classification.
    """
    if piece not in _PIECES:
        raise ValueError(f"Unknown piece: {piece!r}")

    row_masks = board_to_row_masks(board)
    start = tetrio_spawn_state(piece)
    sx = int(start.x)
    sy = int(start.y)
    sr = int(start.rotation) % 4

    if not _can_place(row_masks, piece, sr, sx, sy):
        return []

    queue: deque[tuple[int, int, int]] = deque([(sx, sy, sr)])
    visited: set[tuple[int, int, int]] = {(sx, sy, sr)}
    landing_keys: set[tuple[int, int, int]] = set()

    while queue:
        x, y, rotation = queue.popleft()
        if len(visited) > max_states:
            raise RuntimeError(
                f"Fast reachability exceeded max_states={max_states}; "
                "check geometry-state cycle handling"
            )

        landing_y = _hard_drop_y(row_masks, piece, rotation, x, y)
        landing_keys.add((rotation, x, landing_y))

        for action in _ACTIONS:
            if action == MoveAction.LEFT:
                nxt = (x - 1, y, rotation)
                if not _can_place(row_masks, piece, rotation, nxt[0], nxt[1]):
                    continue
            elif action == MoveAction.RIGHT:
                nxt = (x + 1, y, rotation)
                if not _can_place(row_masks, piece, rotation, nxt[0], nxt[1]):
                    continue
            elif action == MoveAction.DOWN:
                nxt = (x, y + 1, rotation)
                if not _can_place(row_masks, piece, rotation, nxt[0], nxt[1]):
                    continue
            elif action.is_rotation:
                rotated = _try_rotate(
                    row_masks,
                    piece,
                    rotation,
                    x,
                    y,
                    action,
                )
                if rotated is None:
                    continue
                nxt = rotated
            else:
                # HARD_DROP is terminal and should not be in movement_actions.
                continue

            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append(nxt)

    return [
        PieceState(piece=piece, x=x, y=y, rotation=rotation)
        for rotation, x, y in sorted(landing_keys)
    ]


def fast_unique_geometry_keys(
    board: np.ndarray,
    piece: str,
    *,
    max_states: int = 10_000,
) -> set[tuple[str, int, int, int]]:
    """Convenience set form for parity validation."""
    return {
        state.geometry_key()
        for state in enumerate_tetrio_reachable_geometries_fast(
            board,
            piece,
            max_states=max_states,
        )
    }
