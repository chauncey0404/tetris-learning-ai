from __future__ import annotations

from dataclasses import replace

import numpy as np

from tetris_ai.battle.pieces import matrix, occupied_cells
from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.types import (
    MoveAction,
    PieceState,
    RotationAttempt,
    RotationTrace,
)


def normalize_board(board: np.ndarray, ruleset: MovementRuleset) -> np.ndarray:
    arr = np.asarray(board, dtype=np.uint8)
    expected = (ruleset.height, ruleset.width)
    if arr.shape != expected:
        raise ValueError(f"Expected board shape {expected}, got {arr.shape}")
    return (arr != 0).astype(np.uint8, copy=False)


def can_place(board: np.ndarray, state: PieceState, ruleset: MovementRuleset) -> bool:
    arr = normalize_board(board, ruleset)
    for x, y in occupied_cells(state):
        if x < 0 or x >= ruleset.width:
            return False
        if y >= ruleset.height:
            return False
        if y < 0:
            continue
        if arr[y, x]:
            return False
    return True


def _non_rotation_state(state: PieceState, *, x: int, y: int, action: MoveAction) -> PieceState:
    return PieceState(
        piece=state.piece,
        x=x,
        y=y,
        rotation=state.rotation,
        last_action=action,
        last_rotation=None,
    )


def try_shift(
    board: np.ndarray,
    state: PieceState,
    action: MoveAction,
    ruleset: MovementRuleset,
) -> PieceState | None:
    if action == MoveAction.LEFT:
        candidate = _non_rotation_state(state, x=state.x - 1, y=state.y, action=action)
    elif action == MoveAction.RIGHT:
        candidate = _non_rotation_state(state, x=state.x + 1, y=state.y, action=action)
    elif action == MoveAction.DOWN:
        candidate = _non_rotation_state(state, x=state.x, y=state.y + 1, action=action)
    else:
        raise ValueError(f"Not a shift/down action: {action}")
    return candidate if can_place(board, candidate, ruleset) else None


def try_rotate(
    board: np.ndarray,
    state: PieceState,
    action: MoveAction,
    ruleset: MovementRuleset,
) -> RotationAttempt:
    if not action.is_rotation:
        raise ValueError(f"Not a rotation action: {action}")
    if action == MoveAction.ROTATE_180 and not ruleset.allow_180:
        return RotationAttempt(success=False, state=state)

    target, tests = ruleset.rotation_system.kick_tests(
        state.piece,
        state.rotation,
        action,
    )

    for kick_index, (dx, dy) in enumerate(tests):
        trace = RotationTrace(
            action=action,
            from_rotation=int(state.rotation) % 4,
            to_rotation=int(target) % 4,
            kick_index=kick_index,
            kick_dx=dx,
            kick_dy=dy,
        )
        candidate = PieceState(
            piece=state.piece,
            x=state.x + dx,
            y=state.y + dy,
            rotation=target,
            last_action=action,
            last_rotation=trace,
        )
        if can_place(board, candidate, ruleset):
            return RotationAttempt(
                success=True,
                state=candidate,
                kick_index=kick_index,
                kick_dx=dx,
                kick_dy=dy,
            )

    return RotationAttempt(success=False, state=state)


def apply_action(
    board: np.ndarray,
    state: PieceState,
    action: MoveAction,
    ruleset: MovementRuleset,
) -> PieceState | None:
    if action in (MoveAction.LEFT, MoveAction.RIGHT, MoveAction.DOWN):
        return try_shift(board, state, action, ruleset)
    if action.is_rotation:
        attempt = try_rotate(board, state, action, ruleset)
        return attempt.state if attempt.success else None
    raise ValueError("HARD_DROP is terminal; call hard_drop() instead")


def hard_drop(
    board: np.ndarray,
    state: PieceState,
    ruleset: MovementRuleset,
) -> tuple[PieceState, int]:
    if not can_place(board, state, ruleset):
        raise ValueError("Cannot hard-drop from an invalid/colliding state")

    y = state.y
    while True:
        candidate = replace(state, y=y + 1)
        if not can_place(board, candidate, ruleset):
            break
        y += 1

    # Preserve pre-drop last-action/rotation metadata. Spin classification in
    # V9.1 can then decide how hard-drop distance interacts with its rules.
    return replace(state, y=y), y - state.y


def is_grounded(board: np.ndarray, state: PieceState, ruleset: MovementRuleset) -> bool:
    return not can_place(board, replace(state, y=state.y + 1), ruleset)


def lock_piece(
    board: np.ndarray,
    state: PieceState,
    ruleset: MovementRuleset,
) -> np.ndarray:
    arr = normalize_board(board, ruleset).copy()
    if not can_place(arr, state, ruleset):
        raise ValueError("Cannot lock a colliding state")
    for x, y in occupied_cells(state):
        if y < 0:
            # Exact top-out policy is ruleset-specific and belongs to the later
            # battle/lockout layer. Movement keeps the board operation safe.
            continue
        arr[y, x] = 1
    return arr


def clear_lines(board: np.ndarray, ruleset: MovementRuleset) -> tuple[np.ndarray, int]:
    arr = normalize_board(board, ruleset)
    full = np.all(arr != 0, axis=1)
    count = int(np.count_nonzero(full))
    if count == 0:
        return arr.copy(), 0
    kept = arr[~full]
    zeros = np.zeros((count, ruleset.width), dtype=np.uint8)
    return np.vstack([zeros, kept]), count
