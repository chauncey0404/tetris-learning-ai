from __future__ import annotations
import unittest
import numpy as np
from tetris_ai.battle.reachability import enumerate_reachable_placements
from tetris_ai.battle.rules.guideline import GUIDELINE_SRS
from tetris_ai.battle.types import MoveAction

class SharedBattleReachabilityTests(unittest.TestCase):
    def test_guideline_search_is_deterministic(self):
        board = GUIDELINE_SRS.empty_board(); start = GUIDELINE_SRS.spawn_state("T")
        a = enumerate_reachable_placements(board, start, GUIDELINE_SRS)
        b = enumerate_reachable_placements(board, start, GUIDELINE_SRS)
        self.assertEqual([(p.landing_state.geometry_key(), p.spin_signature, p.path) for p in a], [(p.landing_state.geometry_key(), p.spin_signature, p.path) for p in b])
    def test_guideline_search_has_no_180(self):
        board = GUIDELINE_SRS.empty_board(); start = GUIDELINE_SRS.spawn_state("T")
        placements = enumerate_reachable_placements(board, start, GUIDELINE_SRS)
        self.assertFalse(any(MoveAction.ROTATE_180 in p.path for p in placements))
    def test_legacy_visible_board_bridge(self):
        visible = np.zeros((20,10), dtype=np.uint8); visible[-1,4]=1
        board = GUIDELINE_SRS.lift_visible_board(visible)
        self.assertEqual(board.shape,(40,10)); self.assertEqual(int(board[39,4]),1)

if __name__ == "__main__": unittest.main()
