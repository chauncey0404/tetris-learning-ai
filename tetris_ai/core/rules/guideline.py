from __future__ import annotations

from tetris_ai.core.rotation.srs import SRSRotationSystem
from tetris_ai.core.rules.base import MovementRuleset


def guideline_srs_ruleset() -> MovementRuleset:
    """Conservative Guideline-style movement profile.

    Native 180 rotation is not part of standard SRS, so it is disabled here.
    """

    return MovementRuleset(
        name="Guideline SRS",
        rotation_system=SRSRotationSystem(),
        width=10,
        height=40,
        visible_height=20,
        allow_180=False,
    )


GUIDELINE_SRS = guideline_srs_ruleset()
