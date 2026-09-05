from __future__ import annotations

import unittest

import numpy as np

from tetris_ai.battle.reachability import enumerate_reachable_placements, unique_landing_geometries
from tetris_ai.battle.rules.guideline import GUIDELINE_SRS
from tetris_ai.battle.rules.tetrio import TETRIO_DEFAULT
from tetris_ai.battle.types import MoveAction


class V9RulesetTests(unittest.TestCase):
    def test_tetrio_internal_board_is_10x40_with_20_visible_rows(self):
        self.assertEqual((TETRIO_DEFAULT.height, TETRIO_DEFAULT.width), (40, 10))
        self.assertEqual(TETRIO_DEFAULT.visible_height, 20)
        self.assertEqual(TETRIO_DEFAULT.hidden_rows, 20)

    def test_legacy_20x10_board_lifts_into_bottom_visible_half(self):
        visible = np.zeros((20, 10), dtype=np.uint8)
        visible[-1, 4] = 1
        board = TETRIO_DEFAULT.lift_visible_board(visible)
        self.assertEqual(board.shape, (40, 10))
        self.assertEqual(int(board[39, 4]), 1)
        self.assertEqual(int(board[:20].sum()), 0)

    def test_spawn_geometry_matches_modern_srs_rows(self):
        t = TETRIO_DEFAULT.spawn_state("T")
        i = TETRIO_DEFAULT.spawn_state("I")
        self.assertEqual((t.x, t.y), (3, 18))
        self.assertEqual((i.x, i.y), (3, 18))


class V9ReachabilityTests(unittest.TestCase):
    def test_empty_board_reachability_is_deterministic(self):
        board = TETRIO_DEFAULT.empty_board()
        start = TETRIO_DEFAULT.spawn_state("T")
        a = enumerate_reachable_placements(board, start, TETRIO_DEFAULT)
        b = enumerate_reachable_placements(board, start, TETRIO_DEFAULT)
        sig_a = [(p.landing_state.geometry_key(), p.spin_signature, p.path) for p in a]
        sig_b = [(p.landing_state.geometry_key(), p.spin_signature, p.path) for p in b]
        self.assertEqual(sig_a, sig_b)

    def test_tetrio_search_exposes_native_180_paths(self):
        board = TETRIO_DEFAULT.empty_board()
        start = TETRIO_DEFAULT.spawn_state("T")
        placements = enumerate_reachable_placements(board, start, TETRIO_DEFAULT)
        self.assertTrue(any(MoveAction.ROTATE_180 in p.path for p in placements))
        geoms = unique_landing_geometries(placements)
        self.assertTrue(any(g[3] == 2 for g in geoms))

    def test_guideline_search_has_no_180_input(self):
        board = GUIDELINE_SRS.empty_board()
        start = GUIDELINE_SRS.spawn_state("T")
        placements = enumerate_reachable_placements(board, start, GUIDELINE_SRS)
        self.assertFalse(any(MoveAction.ROTATE_180 in p.path for p in placements))

    def test_path_sensitive_same_geometry_can_survive_as_distinct_candidate(self):
        board = TETRIO_DEFAULT.empty_board()
        start = TETRIO_DEFAULT.spawn_state("T")
        placements = enumerate_reachable_placements(board, start, TETRIO_DEFAULT)
        grouped = {}
        for p in placements:
            grouped.setdefault(p.landing_state.geometry_key(), set()).add(p.spin_signature)
        self.assertTrue(any(len(signatures) > 1 for signatures in grouped.values()))


if __name__ == "__main__":
    unittest.main()
