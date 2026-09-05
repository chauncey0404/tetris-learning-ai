"""Shared V9 battle/movement engine.

This package contains game-agnostic movement/search primitives. Concrete game
rules such as TETR.IO live in top-level sibling packages (for example tetrio/).
"""

from tetris_ai.battle.rules import GUIDELINE_SRS
from tetris_ai.battle.types import MoveAction, PieceState, ReachablePlacement, Rotation

__all__ = [
    "GUIDELINE_SRS",
    "MoveAction",
    "PieceState",
    "ReachablePlacement",
    "Rotation",
]
