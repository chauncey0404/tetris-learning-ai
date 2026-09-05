from __future__ import annotations

import numpy as np

from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.spins.base import SpinKind, SpinResult, SpinSystem
from tetris_ai.battle.spins.common import (
    final_rotation_trace,
    is_immobile,
    t_corner_counts,
    used_fifth_90_kick,
)
from tetris_ai.battle.types import ReachablePlacement


class TetrioAllMiniPlusSpinSystem(SpinSystem):
    """TETR.IO All-Mini+ spin classification.

    This is separate from movement so TETR.IO profiles can vary spin settings
    without changing the shared reachability engine.
    """

    name = "TETR.IO All-Mini+"

    def classify(
        self,
        board_before_lock: np.ndarray,
        placement: ReachablePlacement,
        movement_ruleset: MovementRuleset,
        *,
        lines_cleared: int = 0,
    ) -> SpinResult:
        state = placement.landing_state
        trace = final_rotation_trace(placement)
        if trace is None:
            return SpinResult(
                SpinKind.NONE,
                state.piece,
                lines_cleared,
                reason="final maneuver was not a rotation at lock position",
            )

        immobile = is_immobile(board_before_lock, state, movement_ruleset)

        if state.piece != "T":
            return SpinResult(
                SpinKind.MINI if immobile else SpinKind.NONE,
                state.piece,
                lines_cleared,
                immobile=immobile,
                rotation=trace,
                reason="immobile non-T Mini" if immobile else "non-T piece is mobile",
            )

        corners, front = t_corner_counts(board_before_lock, state, movement_ruleset)
        if corners >= 3:
            full = front >= 2 or used_fifth_90_kick(trace)
            return SpinResult(
                SpinKind.FULL if full else SpinKind.MINI,
                state.piece,
                lines_cleared,
                corner_count=corners,
                front_corner_count=front,
                immobile=immobile,
                rotation=trace,
                reason="T 3-corner full" if full else "T 3-corner mini",
            )

        return SpinResult(
            SpinKind.MINI if immobile else SpinKind.NONE,
            state.piece,
            lines_cleared,
            corner_count=corners,
            front_corner_count=front,
            immobile=immobile,
            rotation=trace,
            reason="All-Mini+ immobile T fallback" if immobile else "T is mobile",
        )
