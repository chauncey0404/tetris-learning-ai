from __future__ import annotations

from collections import deque

import numpy as np

from tetris_ai.battle.movement import apply_action, can_place, hard_drop
from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.types import MoveAction, PieceState, ReachablePlacement


def enumerate_reachable_placements(
    board: np.ndarray,
    start: PieceState,
    ruleset: MovementRuleset,
    *,
    max_states: int = 50_000,
) -> list[ReachablePlacement]:
    """BFS all path-sensitive hard-drop placements from one falling piece.

    Search identity deliberately keeps final-action / rotation-kick metadata.
    Two paths that reach the same geometry can remain distinct if one ends in a
    rotation and the other ends in translation; V9.1 spin classification needs
    that distinction.
    """

    if not can_place(board, start, ruleset):
        return []

    queue: deque[tuple[PieceState, tuple[MoveAction, ...]]] = deque([(start, tuple())])
    visited = {start.search_key()}
    placements: dict[tuple, ReachablePlacement] = {}

    while queue:
        state, path = queue.popleft()
        if len(visited) > max_states:
            raise RuntimeError(
                f"Reachability exceeded max_states={max_states}; "
                "check ruleset/state-key cycle handling"
            )

        landing, distance = hard_drop(board, state, ruleset)
        placement = ReachablePlacement(
            pre_drop_state=state,
            landing_state=landing,
            path=path + (MoveAction.HARD_DROP,),
            drop_distance=distance,
        )
        key = (
            landing.geometry_key(),
            placement.spin_signature,
        )
        previous = placements.get(key)
        if previous is None or len(placement.path) < len(previous.path):
            placements[key] = placement

        for action in ruleset.movement_actions:
            next_state = apply_action(board, state, action, ruleset)
            if next_state is None:
                continue
            key2 = next_state.search_key()
            if key2 in visited:
                continue
            visited.add(key2)
            queue.append((next_state, path + (action,)))

    def order_key(item: ReachablePlacement):
        s = item.landing_state
        return (
            s.rotation,
            s.x,
            s.y,
            len(item.path),
            tuple(a.value for a in item.path),
        )

    return sorted(placements.values(), key=order_key)


def unique_landing_geometries(
    placements: list[ReachablePlacement],
) -> set[tuple[str, int, int, int]]:
    return {p.landing_state.geometry_key() for p in placements}
