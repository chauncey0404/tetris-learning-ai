from __future__ import annotations

import unittest

from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.reachability import enumerate_reachable_placements
from tetris_ai.core.types import PieceState


class ExpertReachabilityGateTests(unittest.TestCase):
    def test_empty_board_i_horizontal_expert_geometry_is_reachable(self):
        board = TETRIO_MOVEMENT.empty_board()
        start = TETRIO_MOVEMENT.spawn_state("I")
        placements = enumerate_reachable_placements(
            board, start, TETRIO_MOVEMENT
        )
        keys = {p.landing_state.geometry_key() for p in placements}

        # Historical corpus first observed placement:
        # dataset I x=4,y=0,N -> canonical PieceState(I,3,38,0)
        self.assertIn(PieceState("I", 3, 38, 0).geometry_key(), keys)

    def test_tetrio_reachability_contains_180_landings(self):
        board = TETRIO_MOVEMENT.empty_board()
        start = TETRIO_MOVEMENT.spawn_state("T")
        placements = enumerate_reachable_placements(
            board, start, TETRIO_MOVEMENT
        )
        self.assertTrue(
            any(p.landing_state.rotation == 2 for p in placements)
        )


if __name__ == "__main__":
    unittest.main()
