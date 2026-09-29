from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RetargetResult:
    safe: bool
    reason: str
    current_states: tuple[Any, ...]
    movement_path: tuple[str, ...] | None
    target_cells: tuple[tuple[int, int], ...]
    search_nodes: int = 0
    search_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "safe": bool(self.safe),
            "reason": self.reason,
            "current_states": [
                {
                    "piece": s.piece,
                    "x": int(s.x),
                    "y": int(s.y),
                    "rotation": int(s.rotation) % 4,
                }
                for s in self.current_states
            ],
            "movement_path": None if self.movement_path is None else list(self.movement_path),
            "target_cells": [list(x) for x in self.target_cells],
            "search_nodes": int(self.search_nodes),
            "search_ms": float(self.search_ms),
        }


@dataclass(frozen=True)
class RetargetRequest:
    board: tuple[tuple[int, ...], ...]
    source_active_piece: str
    active_piece: str
    active_rotation: int
    active_x: int
    active_y: int
    target_piece: str
    target_x: int
    target_y: int
    target_rotation: int

    @property
    def active_signature(self) -> tuple[str, int, int, int]:
        return (
            self.active_piece,
            int(self.active_rotation),
            int(self.active_x),
            int(self.active_y),
        )

    def board_array(self) -> np.ndarray:
        return np.asarray(self.board, dtype=np.uint8)


def _canonical_rotations_for_visual(active) -> list[int]:
    from tetris_ai.core.tetrominoes import trimmed_matrix, unique_trimmed_rotations

    piece = str(active.piece)
    unique = unique_trimmed_rotations(piece)
    idx = int(active.rotation)
    if idx < 0 or idx >= len(unique):
        return []
    visual_shape = unique[idx][1]
    return [
        rot
        for rot in range(4)
        if np.array_equal(trimmed_matrix(piece, rot), visual_shape)
    ]


def visual_active_state_candidates(active, board40: np.ndarray) -> tuple[Any, ...]:
    """Map a tracker trimmed-shape observation to all compatible SRS states.

    Several SRS rotation states can have identical trimmed pixels.  Keep every
    compatible state and only accept a retarget path if the same key sequence is
    valid for all of them.
    """
    from tetrio.ruleset import TETRIO_MOVEMENT
    from tetris_ai.core.movement import can_place
    from tetris_ai.core.tetrominoes import native_matrix
    from tetris_ai.core.types import PieceState

    board = np.asarray(board40, dtype=np.uint8).reshape(40, 10)
    piece = str(active.piece)
    states = []
    seen = set()
    for rotation in _canonical_rotations_for_visual(active):
        matrix = native_matrix(piece, rotation)
        ys, xs = np.nonzero(matrix)
        if len(xs) != 4:
            continue
        native_min_x = int(xs.min())
        native_min_y = int(ys.min())
        state = PieceState(
            piece=piece,
            x=int(active.x) - native_min_x,
            # Visual row 0 is project row 20 on the 40-row movement board.
            y=20 + int(active.y) - native_min_y,
            rotation=int(rotation),
        )
        key = state.geometry_key()
        if key in seen:
            continue
        seen.add(key)
        if can_place(board, state, TETRIO_MOVEMENT):
            states.append(state)
    return tuple(states)


def _occupied_key(state) -> tuple[tuple[int, int], ...]:
    from tetris_ai.core.tetrominoes import occupied_cells
    return tuple(sorted((int(x), int(y)) for x, y in occupied_cells(state)))


def _search_key(states: tuple[Any, ...]) -> tuple:
    return tuple(
        s.search_key() if hasattr(s, "search_key") else s.geometry_key()
        for s in states
    )


def _rotation_action_distance(rotation: int, target_rotation: int) -> int:
    delta = (int(target_rotation) - int(rotation)) % 4
    if delta == 0:
        return 0
    # TETR.IO movement supports 180, so every non-zero rotation delta is at
    # least one input away.  This is an ordering heuristic only; kick legality
    # is still decided by the authoritative movement engine.
    return 1


def _target_heuristic(states: tuple[Any, ...], target_state) -> int:
    """Cheap admissibility-agnostic ordering hint for target-directed search.

    The search remains complete because this value only orders the frontier;
    it never prunes a legal state.  Horizontal/rotation alignment is the only
    thing HARD_DROP cannot change, so prioritizing it sharply reduces the
    number of states examined before the exact target can be tested.
    """
    target_x = int(target_state.x)
    target_rotation = int(target_state.rotation) % 4
    return max(
        2 * abs(int(state.x) - target_x)
        + 2 * _rotation_action_distance(int(state.rotation) % 4, target_rotation)
        for state in states
    )


def _aligned_for_target_drop(states: tuple[Any, ...], target_state) -> bool:
    """HARD_DROP preserves x/rotation, so test it only when both already match."""
    tx = int(target_state.x)
    tr = int(target_state.rotation) % 4
    ty = int(target_state.y)
    return all(
        int(state.x) == tx
        and int(state.rotation) % 4 == tr
        and int(state.y) <= ty
        for state in states
    )


def _common_target_path(
    board: np.ndarray,
    states: tuple[Any, ...],
    target_state,
    *,
    max_states: int,
) -> tuple[tuple[str, ...] | None, int, str]:
    """Target-directed synchronized best-first search.

    Every node represents the same input history applied to every SRS state
    compatible with the visual silhouette.  Unlike the previous BFS, this
    implementation does *not* run HARD_DROP at every node: x and rotation must
    already equal the selected policy target because HARD_DROP cannot change
    either.  On the live gate this removes the dominant per-node collision
    scanning cost while preserving the exact movement rules and fail-closed
    rotation-alias contract.
    """
    from tetrio.ruleset import TETRIO_MOVEMENT
    from tetris_ai.core.movement import apply_action, hard_drop
    from tetris_ai.core.types import MoveAction

    target_key = target_state.geometry_key()
    initial = tuple(states)
    serial = itertools.count()
    # (priority, path_len, serial, states, path)
    frontier: list[tuple[int, int, int, tuple[Any, ...], tuple[str, ...]]] = []
    heapq.heappush(
        frontier,
        (_target_heuristic(initial, target_state), 0, next(serial), initial, tuple()),
    )
    best_depth = {_search_key(initial): 0}
    expanded = 0

    while frontier:
        _priority, depth, _serial, current, path = heapq.heappop(frontier)
        key0 = _search_key(current)
        if depth != best_depth.get(key0):
            continue

        expanded += 1
        if expanded > int(max_states):
            return None, expanded, "retarget_search_exceeded_max_states"

        # HARD_DROP is expensive because it repeatedly checks collision while
        # descending.  It can only reach target_key when x/rotation already
        # match the target, so avoid calling it for every exploratory node.
        if _aligned_for_target_drop(current, target_state):
            all_target = True
            for state in current:
                landing, _distance = hard_drop(board, state, TETRIO_MOVEMENT)
                if landing.geometry_key() != target_key:
                    all_target = False
                    break
            if all_target:
                return path + (MoveAction.HARD_DROP.value,), expanded, "safe_common_path"

        for action in TETRIO_MOVEMENT.movement_actions:
            next_states = []
            valid = True
            for state in current:
                nxt = apply_action(board, state, action, TETRIO_MOVEMENT)
                if nxt is None:
                    valid = False
                    break
                next_states.append(nxt)
            if not valid:
                continue

            group = tuple(next_states)
            key = _search_key(group)
            next_depth = depth + 1
            old_depth = best_depth.get(key)
            if old_depth is not None and old_depth <= next_depth:
                continue
            best_depth[key] = next_depth
            heuristic = _target_heuristic(group, target_state)
            heapq.heappush(
                frontier,
                (
                    next_depth + heuristic,
                    next_depth,
                    next(serial),
                    group,
                    path + (action.value,),
                ),
            )

    if len(states) > 1:
        return None, expanded, "rotation_alias_has_no_common_safe_path"
    return None, expanded, "target_unreachable_from_possible_visual_state"

def retarget_to_landing(
    board40: np.ndarray,
    active,
    target_state,
    *,
    max_states: int = 50_000,
) -> RetargetResult:
    """Find one exact action path valid for every visually possible SRS state."""
    board = np.asarray(board40, dtype=np.uint8).reshape(40, 10)
    target_cells = _occupied_key(target_state)
    states = visual_active_state_candidates(active, board)
    if not states:
        return RetargetResult(False, "visual_state_unmappable", (), None, target_cells)

    started = time.perf_counter()
    path, nodes, reason = _common_target_path(
        board,
        states,
        target_state,
        max_states=int(max_states),
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    return RetargetResult(
        path is not None,
        reason,
        states,
        path,
        target_cells,
        search_nodes=nodes,
        search_ms=elapsed,
    )


def _fresh_board40(fresh_temporal) -> np.ndarray | None:
    locked = getattr(fresh_temporal, "locked_board", None)
    if locked is None:
        return None
    from tetrio.ruleset import TETRIO_MOVEMENT
    visible = (np.asarray(locked) != 0).astype(np.uint8)
    return (
        np.asarray(TETRIO_MOVEMENT.lift_visible_board(visible)) != 0
    ).astype(np.uint8)


def build_retarget_request(source_observation, decision, fresh_temporal):
    """Cheap fail-closed validation before a CPU retarget worker is launched."""
    if bool(decision.chosen.use_hold):
        return None, RetargetResult(
            False,
            "hold_must_be_executed_and_reobserved_before_retarget",
            (), None, (),
        )
    active = getattr(fresh_temporal, "active", None)
    if active is None:
        return None, RetargetResult(False, "fresh_active_unresolved", (), None, ())
    if str(active.piece) != str(source_observation.active_piece):
        return None, RetargetResult(False, "active_generation_changed", (), None, ())
    fresh_board = _fresh_board40(fresh_temporal)
    if fresh_board is None:
        return None, RetargetResult(False, "fresh_locked_board_unresolved", (), None, ())
    source_board = np.asarray(source_observation.board_array(), dtype=np.uint8)
    if not np.array_equal(fresh_board, source_board):
        return None, RetargetResult(False, "locked_board_changed_decision_stale", (), None, ())

    target = decision.chosen.state
    request = RetargetRequest(
        board=tuple(tuple(int(v) for v in row) for row in source_board),
        source_active_piece=str(source_observation.active_piece),
        active_piece=str(active.piece),
        active_rotation=int(active.rotation),
        active_x=int(active.x),
        active_y=int(active.y),
        target_piece=str(target.piece),
        target_x=int(target.x),
        target_y=int(target.y),
        target_rotation=int(target.rotation) % 4,
    )
    return request, None


def run_retarget_request(request: RetargetRequest, *, max_states: int = 50_000) -> RetargetResult:
    """Pickle-safe CPU worker entrypoint for current-state target search."""
    from types import SimpleNamespace
    from tetris_ai.core.types import PieceState

    active = SimpleNamespace(
        piece=request.active_piece,
        rotation=int(request.active_rotation),
        x=int(request.active_x),
        y=int(request.active_y),
    )
    target = PieceState(
        piece=request.target_piece,
        x=int(request.target_x),
        y=int(request.target_y),
        rotation=int(request.target_rotation) % 4,
    )
    return retarget_to_landing(
        request.board_array(),
        active,
        target,
        max_states=int(max_states),
    )


def retarget_request_stale_reason(request: RetargetRequest, fresh_temporal) -> str | None:
    """Validate that a completed path still starts from the currently seen state."""
    active = getattr(fresh_temporal, "active", None)
    if active is None:
        return "fresh_active_unresolved"
    if str(active.piece) != request.source_active_piece:
        return "active_generation_changed"
    fresh_board = _fresh_board40(fresh_temporal)
    if fresh_board is None:
        return "fresh_locked_board_unresolved"
    if not np.array_equal(fresh_board, request.board_array()):
        return "locked_board_changed_decision_stale"
    signature = (
        str(active.piece),
        int(active.rotation),
        int(active.x),
        int(active.y),
    )
    if signature != request.active_signature:
        return "active_visual_state_advanced"
    return None


def retarget_live_decision(source_observation, decision, fresh_temporal) -> RetargetResult:
    """Synchronous compatibility wrapper used by tests and diagnostics."""
    request, failure = build_retarget_request(
        source_observation,
        decision,
        fresh_temporal,
    )
    if failure is not None:
        return failure
    assert request is not None
    return run_retarget_request(request)

