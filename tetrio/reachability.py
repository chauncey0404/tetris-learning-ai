from __future__ import annotations

from dataclasses import replace

import numpy as np

from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.reachability import enumerate_reachable_placements
from tetris_ai.core.types import PieceState, ReachablePlacement


# Historical top-player corpus parity:
# - baseline generic entry missed 4 / 1,459 sampled expert placements because
#   the generic start collided with the stack;
# - shifting the entry one row upward recovered all 4;
# - a larger deterministic stratified gate then matched 6,537 / 6,537 expert
#   placements exactly.
#
# Keep this TETR.IO-specific. Do not alter the shared Guideline/core spawn.
TETRIO_ENTRY_RAISE_ROWS = 1


def tetrio_spawn_state(piece: str) -> PieceState:
    """Return the TETR.IO entry state in project 40-row board coordinates."""
    generic = TETRIO_MOVEMENT.spawn_state(piece)
    return replace(generic, y=generic.y - TETRIO_ENTRY_RAISE_ROWS)


def enumerate_tetrio_reachable_placements(
    board: np.ndarray,
    piece: str,
    *,
    max_states: int = 50_000,
) -> list[ReachablePlacement]:
    """Enumerate TETR.IO SRS+/180 hard-drop candidates from TETR.IO entry."""
    return enumerate_reachable_placements(
        board,
        tetrio_spawn_state(piece),
        TETRIO_MOVEMENT,
        max_states=max_states,
    )
