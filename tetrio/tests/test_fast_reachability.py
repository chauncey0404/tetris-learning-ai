from __future__ import annotations

import unittest

import numpy as np

from tetrio.fast_reachability import fast_unique_geometry_keys
from tetrio.reachability import enumerate_tetrio_reachable_placements
from tetrio.ruleset import TETRIO_MOVEMENT


def reference_keys(board, piece: str):
    return {
        p.landing_state.geometry_key()
        for p in enumerate_tetrio_reachable_placements(board, piece)
    }


class FastReachabilityParityTests(unittest.TestCase):
    def assert_piece_parity(self, board, piece: str):
        self.assertEqual(
            fast_unique_geometry_keys(board, piece),
            reference_keys(board, piece),
        )

    def test_empty_board_all_pieces_match_reference(self):
        board = TETRIO_MOVEMENT.empty_board()
        for piece in "IOTSZJL":
            with self.subTest(piece=piece):
                self.assert_piece_parity(board, piece)

    def test_irregular_stack_all_pieces_match_reference(self):
        board = TETRIO_MOVEMENT.empty_board()
        # Deterministic non-full skyline with holes/tuck opportunities.
        heights = (3, 5, 2, 6, 4, 7, 3, 5, 1, 4)
        for x, height in enumerate(heights):
            board[40 - height :, x] = 1
        board[38, 1] = 0
        board[37, 4] = 0
        board[39, 7] = 0

        for piece in "IOTSZJL":
            with self.subTest(piece=piece):
                self.assert_piece_parity(board, piece)

    def test_fast_backend_returns_unique_geometry_keys(self):
        board = TETRIO_MOVEMENT.empty_board()
        keys = fast_unique_geometry_keys(board, "T")
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(any(k[3] == 2 for k in keys))


if __name__ == "__main__":
    unittest.main()
