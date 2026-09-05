from __future__ import annotations

import numpy as np

from tetris_ai.core.rules.base import MovementRuleset
from tetris_ai.core.spins.base import SpinKind, SpinResult, SpinSystem
from tetris_ai.core.spins.common import final_rotation_trace, t_corner_counts, used_fifth_90_kick
from tetris_ai.core.types import ReachablePlacement


class GuidelineTSpinSystem(SpinSystem):
    name = "Guideline T-only 3-corner"

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
        if state.piece != "T" or trace is None:
            return SpinResult(
                SpinKind.NONE,
                state.piece,
                lines_cleared,
                rotation=trace,
                reason="not a final-rotation T placement",
            )

        corners, front = t_corner_counts(board_before_lock, state, movement_ruleset)
        if corners < 3:
            return SpinResult(
                SpinKind.NONE,
                state.piece,
                lines_cleared,
                corner_count=corners,
                front_corner_count=front,
                rotation=trace,
                reason="T failed 3-corner rule",
            )

        full = front >= 2 or used_fifth_90_kick(trace)
        return SpinResult(
            SpinKind.FULL if full else SpinKind.MINI,
            state.piece,
            lines_cleared,
            corner_count=corners,
            front_corner_count=front,
            rotation=trace,
            reason="T 3-corner full" if full else "T 3-corner mini",
        )
