from __future__ import annotations

from dataclasses import dataclass

from tetris_ai.core.rules.game import GameRuleset


@dataclass(frozen=True, slots=True)
class TetrioBattleProfile:
    profile_id: str
    name: str
    game_rules: GameRuleset
    opener_phase_pieces: int = 14
    surge_start: int = 4
    all_clear_attack: int = 5
    rounding: str = "down"


# Filled from tetrio.ruleset after movement/spin systems are constructed.
