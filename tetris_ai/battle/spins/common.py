from __future__ import annotations

from dataclasses import replace

import numpy as np

from tetris_ai.battle.movement import can_place, normalize_board
from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.types import MoveAction, PieceState, ReachablePlacement, RotationTrace


_T_CORNER_OFFSETS = ((-1, -1), (+1, -1), (-1, +1), (+1, +1))
_T_FRONT_BY_ROTATION = {
    0: ((-1, -1), (+1, -1)),
    1: ((+1, -1), (+1, +1)),
    2: ((-1, +1), (+1, +1)),
    3: ((-1, -1), (-1, +1)),
}


def final_rotation_trace(placement: ReachablePlacement) -> RotationTrace | None:
    state = placement.pre_drop_state
    if placement.drop_distance != 0:
        return None
    if state.last_action is None or not state.last_action.is_rotation:
        return None
    return state.last_rotation


def occupied_or_wall(board: np.ndarray, x: int, y: int, ruleset: MovementRuleset) -> bool:
    if x < 0 or x >= ruleset.width or y < 0 or y >= ruleset.height:
        return True
    return bool(board[y, x])


def t_corner_counts(
    board_before_lock: np.ndarray,
    state: PieceState,
    ruleset: MovementRuleset,
) -> tuple[int, int]:
    board = normalize_board(board_before_lock, ruleset)
    pivot_x = state.x + 1
    pivot_y = state.y + 1
    all_count = sum(
        occupied_or_wall(board, pivot_x + dx, pivot_y + dy, ruleset)
        for dx, dy in _T_CORNER_OFFSETS
    )
    front_count = sum(
        occupied_or_wall(board, pivot_x + dx, pivot_y + dy, ruleset)
        for dx, dy in _T_FRONT_BY_ROTATION[int(state.rotation) % 4]
    )
    return int(all_count), int(front_count)


def used_fifth_90_kick(trace: RotationTrace | None) -> bool:
    return bool(
        trace is not None
        and trace.action in (MoveAction.CW, MoveAction.CCW)
        and trace.kick_index == 4
    )


def is_immobile(
    board_before_lock: np.ndarray,
    state: PieceState,
    ruleset: MovementRuleset,
) -> bool:
    board = normalize_board(board_before_lock, ruleset)
    for dx, dy in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
        if can_place(board, replace(state, x=state.x + dx, y=state.y + dy), ruleset):
            return False
    return True
