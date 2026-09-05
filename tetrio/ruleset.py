from __future__ import annotations

from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.rules.game import GameRuleset
from tetrio.rotation import TetrioSRSPlusRotationSystem
from tetrio.spins import TetrioAllMiniPlusSpinSystem


TETRIO_MOVEMENT = MovementRuleset(
    name="TETR.IO SRS+ movement",
    rotation_system=TetrioSRSPlusRotationSystem(),
    width=10,
    height=40,
    visible_height=20,
    allow_180=True,
)

# Current project target. Attack/combo/B2B/garbage semantics are added in V9.2.
TETRIO_MULTIPLAYER = GameRuleset(
    game_id="tetrio",
    profile_id="current_multiplayer",
    name="TETR.IO current multiplayer",
    movement=TETRIO_MOVEMENT,
    spins=TetrioAllMiniPlusSpinSystem(),
)

# Backward-friendly movement alias for V9.0-era callers.
TETRIO_DEFAULT = TETRIO_MOVEMENT
