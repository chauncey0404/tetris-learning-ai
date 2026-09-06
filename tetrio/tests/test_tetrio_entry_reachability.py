from __future__ import annotations

import unittest

from tetrio.reachability import (
    TETRIO_ENTRY_RAISE_ROWS,
    enumerate_tetrio_reachable_placements,
    tetrio_spawn_state,
)
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.types import PieceState


class TetrioEntryReachabilityTests(unittest.TestCase):
    def test_entry_is_one_row_above_generic_movement_spawn(self):
        self.assertEqual(TETRIO_ENTRY_RAISE_ROWS, 1)
        for piece in "IOTSZJL":
            with self.subTest(piece=piece):
                generic = TETRIO_MOVEMENT.spawn_state(piece)
                entry = tetrio_spawn_state(piece)
                self.assertEqual(entry.piece, generic.piece)
                self.assertEqual(entry.x, generic.x)
                self.assertEqual(entry.rotation, generic.rotation)
                self.assertEqual(entry.y, generic.y - 1)

    def test_empty_board_first_observed_i_geometry_is_reachable(self):
        board = TETRIO_MOVEMENT.empty_board()
        placements = enumerate_tetrio_reachable_placements(board, "I")
        keys = {p.landing_state.geometry_key() for p in placements}
        self.assertIn(PieceState("I", 3, 38, 0).geometry_key(), keys)

    def test_all_pieces_have_candidates_on_empty_board(self):
        board = TETRIO_MOVEMENT.empty_board()
        for piece in "IOTSZJL":
            with self.subTest(piece=piece):
                self.assertTrue(
                    enumerate_tetrio_reachable_placements(board, piece)
                )


if __name__ == "__main__":
    unittest.main()
