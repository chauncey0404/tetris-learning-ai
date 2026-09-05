from __future__ import annotations

import unittest

import numpy as np

from tetrio.datasets.top_players_s1 import (
    CoordinateConvention,
    decode_playfield,
    encode_binary_playfield,
    row_to_piece_state,
    simulate_dataset_placement,
)


class TopPlayersDatasetSemanticsTests(unittest.TestCase):
    def test_empty_playfield(self):
        board = decode_playfield("")
        self.assertEqual(board.shape, (40, 10))
        self.assertEqual(int(board.sum()), 0)

    def test_first_observed_i_example_decodes_bottom_up(self):
        board = decode_playfield("NNNIIII")
        self.assertEqual(int(board.sum()), 4)
        self.assertTrue(np.all(board[39, 3:7] == 1))

    def test_observed_first_i_placement_reconstructs_next_playfield(self):
        before = decode_playfield("")
        sim = simulate_dataset_placement(
            before,
            piece="I",
            x=4,
            y=0,
            rotation_code="N",
            convention=CoordinateConvention(x_offset=-1, y_base=38, rotation_map_name="cw"),
        )
        self.assertTrue(sim.legal)
        self.assertEqual(sim.lines_cleared, 0)
        self.assertTrue(
            np.array_equal(sim.board_after_clear, decode_playfield("NNNIIII"))
        )

    def test_second_observed_z_placement_reconstructs_third_playfield(self):
        before = decode_playfield("NNNIIII")
        sim = simulate_dataset_placement(
            before,
            piece="Z",
            x=4,
            y=1,
            rotation_code="N",
            convention=CoordinateConvention(x_offset=-1, y_base=38, rotation_map_name="cw"),
        )
        expected = decode_playfield("NNNIIIINNNNNNNZZNNNNNNNZZ")
        self.assertTrue(sim.legal)
        self.assertEqual(sim.lines_cleared, 0)
        self.assertTrue(np.array_equal(sim.board_after_clear, expected))

    def test_rotation_codes_map_clockwise(self):
        expected = {"N": 0, "E": 1, "S": 2, "W": 3}
        for code, rotation in expected.items():
            with self.subTest(code=code):
                state = row_to_piece_state(
                    piece="T",
                    x=4,
                    y=0,
                    rotation_code=code,
                )
                self.assertEqual(state.rotation, rotation)

    def test_binary_encode_round_trip(self):
        board = decode_playfield("NNNIIIINNNNNNNZZNNNNNNNZZ")
        encoded = encode_binary_playfield(board, filled_code="X")
        decoded = decode_playfield(encoded, empty_code="N")
        self.assertTrue(np.array_equal(board, decoded))


if __name__ == "__main__":
    unittest.main()
