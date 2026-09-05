from __future__ import annotations

from dataclasses import dataclass

from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.spins.base import SpinSystem


@dataclass(frozen=True)
class GameRuleset:
    """Composes movement + spin semantics for a named game/profile.

    Attack/combo/B2B/garbage components will be added in V9.2 without moving
    game-specific code back into the shared engine.
    """

    game_id: str
    profile_id: str
    name: str
    movement: MovementRuleset
    spins: SpinSystem
