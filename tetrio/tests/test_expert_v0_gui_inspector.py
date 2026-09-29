from __future__ import annotations

import unittest
import numpy as np

from tetrio.tools.inspect_expert_v0_gui import (
    GARBAGE_BLOCK_ID,
    UNKNOWN_BLOCK_ID,
    VIS_PIECE_ID,
    _propagate_colors_after_clear,
    decode_playfield_ids,
    matches_filter,
)


class ExpertV0GuiInspectorTests(unittest.TestCase):
    def test_disagreement_filter(self):
        self.assertTrue(matches_filter("disagreement", 2, 1, 4))
        self.assertFalse(matches_filter("disagreement", 1, 1, 1))

    def test_top3_miss_filter(self):
        self.assertTrue(matches_filter("top3_miss", 0, 4, 4))
        self.assertFalse(matches_filter("top3_miss", 0, 2, 3))

    def test_all_filter(self):
        self.assertTrue(matches_filter("all", 0, 0, 1))

    def test_playfield_floor_up_color_decode(self):
        board = decode_playfield_ids("INNNNNNNNN")
        self.assertIsNotNone(board)
        assert board is not None
        self.assertEqual(int(board[39, 0]), VIS_PIECE_ID["I"])
        self.assertEqual(int(board[39, 1]), 0)

    def test_literal_g_is_garbage_and_preserves_position(self):
        board = decode_playfield_ids("GNNNNNNNNNI")
        self.assertIsNotNone(board)
        assert board is not None
        self.assertEqual(int(board[39, 0]), GARBAGE_BLOCK_ID)
        self.assertEqual(int(board[38, 0]), VIS_PIECE_ID["I"])
        self.assertEqual(int(board[39, 1]), 0)

    def test_unexpected_nonpiece_code_is_unknown_not_garbage(self):
        board = decode_playfield_ids("X")
        self.assertIsNotNone(board)
        assert board is not None
        self.assertEqual(int(board[39, 0]), UNKNOWN_BLOCK_ID)
        self.assertNotEqual(int(board[39, 0]), GARBAGE_BLOCK_ID)

    def test_source_j_l_are_canonicalized_for_display(self):
        board = decode_playfield_ids("JL")
        self.assertIsNotNone(board)
        assert board is not None
        self.assertEqual(int(board[39, 0]), VIS_PIECE_ID["L"])
        self.assertEqual(int(board[39, 1]), VIS_PIECE_ID["J"])

    def test_no_clear_preserves_old_color_and_colors_new_piece(self):
        before_ids = np.zeros((40, 10), dtype=np.uint8)
        before_ids[39, 0] = GARBAGE_BLOCK_ID
        after = np.zeros((40, 10), dtype=np.uint8)
        after[39, 0] = 1
        after[39, 1] = 1
        out = _propagate_colors_after_clear(
            before_ids,
            after,
            placed_piece="O",
            lines=0,
        )
        self.assertEqual(int(out[39, 0]), GARBAGE_BLOCK_ID)
        self.assertEqual(int(out[39, 1]), VIS_PIECE_ID["O"])


if __name__ == "__main__":
    unittest.main()
