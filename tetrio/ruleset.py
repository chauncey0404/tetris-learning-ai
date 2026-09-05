from __future__ import annotations

from tetris_ai.core.rules.base import MovementRuleset
from tetris_ai.core.rules.game import GameRuleset
from tetrio.rotation import TetrioSRSPlusRotationSystem
from tetrio.spins import TetrioAllMiniPlusSpinSystem, TetrioAllMiniSpinSystem


TETRIO_MOVEMENT = MovementRuleset(
    name="TETR.IO SRS+ movement",
    rotation_system=TetrioSRSPlusRotationSystem(),
    width=10,
    height=40,
    visible_height=20,
    allow_180=True,
)

# Ranked Season 2 target. Official Season 2 notes specify All-Mini.
TETRIO_TETRA_LEAGUE = GameRuleset(
    game_id="tetrio",
    profile_id="tetra_league_season2",
    name="TETR.IO Tetra League Season 2",
    movement=TETRIO_MOVEMENT,
    spins=TetrioAllMiniSpinSystem(),
)

# Optional profile for TETR.IO modes/custom settings that enable All-Mini+.
TETRIO_ALL_MINI_PLUS = GameRuleset(
    game_id="tetrio",
    profile_id="all_mini_plus",
    name="TETR.IO All-Mini+ profile",
    movement=TETRIO_MOVEMENT,
    spins=TetrioAllMiniPlusSpinSystem(),
)

# Backward-compatible project alias. V9 battle research now targets ranked TL.
TETRIO_MULTIPLAYER = TETRIO_TETRA_LEAGUE
TETRIO_DEFAULT = TETRIO_MOVEMENT
