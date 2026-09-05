from __future__ import annotations

import unittest

import numpy as np

from tetris_ai.battle.movement import try_rotate
from tetris_ai.battle.pieces import matrix
from tetris_ai.battle.rotation.srs import SRSRotationSystem
from tetris_ai.battle.rotation.tetrio_srs_plus import TetrioSRSPlusRotationSystem
from tetris_ai.battle.rules.guideline import GUIDELINE_SRS
from tetris_ai.battle.rules.tetrio import TETRIO_DEFAULT
from tetris_ai.battle.types import MoveAction, PieceState


class V9PieceGeometryTests(unittest.TestCase):
    def test_v8_native_spawn_geometry_is_preserved(self):
        expected_i = np.asarray(
            [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
        expected_t = np.asarray(
            [[0, 1, 0], [1, 1, 1], [0, 0, 0]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(matrix("I", 0), expected_i)
        np.testing.assert_array_equal(matrix("T", 0), expected_t)

    def test_four_clockwise_rotations_return_to_spawn(self):
        for piece in ("I", "O", "T", "S", "Z", "J", "L"):
            np.testing.assert_array_equal(matrix(piece, 0), matrix(piece, 4))


class V9KickTableTests(unittest.TestCase):
    def test_standard_srs_jlstz_board_coordinate_conversion(self):
        system = SRSRotationSystem()
        target, kicks = system.kick_tests("T", 0, MoveAction.CW)
        self.assertEqual(target, 1)
        self.assertEqual(
            kicks,
            ((0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)),
        )

    def test_tetrio_srs_plus_i_0_to_r_is_distinct_from_standard_srs(self):
        standard = SRSRotationSystem()
        tetrio = TetrioSRSPlusRotationSystem()
        _, standard_kicks = standard.kick_tests("I", 0, MoveAction.CW)
        target, tetrio_kicks = tetrio.kick_tests("I", 0, MoveAction.CW)
        self.assertEqual(target, 1)
        self.assertEqual(
            tetrio_kicks,
            ((0, 0), (1, 0), (-2, 0), (-2, 1), (1, -2)),
        )
        self.assertNotEqual(standard_kicks, tetrio_kicks)

    def test_tetrio_non_i_180_has_six_ordered_tests(self):
        system = TetrioSRSPlusRotationSystem()
        target, kicks = system.kick_tests("T", 0, MoveAction.ROTATE_180)
        self.assertEqual(target, 2)
        self.assertEqual(
            kicks,
            ((0, 0), (0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0)),
        )
        self.assertEqual(len(kicks), 6)

    def test_tetrio_i_180_is_native_not_two_90s(self):
        system = TetrioSRSPlusRotationSystem()
        target, kicks = system.kick_tests("I", 0, MoveAction.ROTATE_180)
        self.assertEqual(target, 2)
        self.assertEqual(kicks, ((0, 0), (0, -1)))

    def test_guideline_profile_rejects_180(self):
        board = GUIDELINE_SRS.empty_board()
        state = GUIDELINE_SRS.spawn_state("T")
        attempt = try_rotate(board, state, MoveAction.ROTATE_180, GUIDELINE_SRS)
        self.assertFalse(attempt.success)

    def test_floor_kick_is_applied_in_board_coordinates(self):
        board = TETRIO_DEFAULT.empty_board()
        # Horizontal T occupies y=38,39. Basic CW would occupy through y=40,
        # so SRS test #3 (-1,-1 in board coordinates) kicks it up one row.
        state = PieceState(piece="T", x=3, y=38, rotation=0)
        attempt = try_rotate(board, state, MoveAction.CW, TETRIO_DEFAULT)
        self.assertTrue(attempt.success)
        self.assertEqual(attempt.kick_index, 2)
        self.assertEqual((attempt.kick_dx, attempt.kick_dy), (-1, -1))
        self.assertEqual((attempt.state.x, attempt.state.y, attempt.state.rotation), (2, 37, 1))


if __name__ == "__main__":
    unittest.main()
