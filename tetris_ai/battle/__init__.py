"""V9 battle/movement engine.

This package is intentionally separate from the validated V8 single-player
PlacementAction pipeline. V8 remains frozen; V9 adds path-aware movement and
ruleset-specific rotation behavior.
"""

from tetris_ai.battle.rules import GUIDELINE_SRS, TETRIO_DEFAULT
from tetris_ai.battle.types import MoveAction, PieceState, ReachablePlacement, Rotation

__all__ = [
    "GUIDELINE_SRS",
    "TETRIO_DEFAULT",
    "MoveAction",
    "PieceState",
    "ReachablePlacement",
    "Rotation",
]
