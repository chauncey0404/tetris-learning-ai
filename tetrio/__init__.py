"""TETR.IO-specific rules, tests, tools and experiments.

Shared Tetris/battle machinery belongs in tetris_ai; TETR.IO-specific behavior
belongs in this top-level package.
"""

from tetrio.rotation import TetrioSRSPlusRotationSystem
from tetrio.ruleset import TETRIO_DEFAULT, TETRIO_MOVEMENT, TETRIO_MULTIPLAYER
from tetrio.spins import TetrioAllMiniPlusSpinSystem

__all__ = [
    "TetrioSRSPlusRotationSystem",
    "TetrioAllMiniPlusSpinSystem",
    "TETRIO_DEFAULT",
    "TETRIO_MOVEMENT",
    "TETRIO_MULTIPLAYER",
]
