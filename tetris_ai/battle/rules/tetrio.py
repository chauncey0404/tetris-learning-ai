from __future__ import annotations

from tetris_ai.battle.rotation.tetrio_srs_plus import TetrioSRSPlusRotationSystem
from tetris_ai.battle.rules.base import MovementRuleset


def tetrio_default_movement_ruleset() -> MovementRuleset:
    """TETR.IO-oriented default movement profile for V9 research.

    TETR.IO exposes kickset and 180 permission as room/game options. This
    factory represents the project's current target: default SRS+ with native
    180 enabled. Future custom-room profiles should be separate factories.
    """

    return MovementRuleset(
        name="TETR.IO default movement (SRS+ + 180)",
        rotation_system=TetrioSRSPlusRotationSystem(),
        width=10,
        height=40,
        visible_height=20,
        allow_180=True,
    )


TETRIO_DEFAULT = tetrio_default_movement_ruleset()
