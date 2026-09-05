from __future__ import annotations
import unittest
from tetrio import TETRIO_MOVEMENT
from tetris_ai.core.reachability import enumerate_reachable_placements, unique_landing_geometries
from tetris_ai.core.types import MoveAction

class TetrioReachabilityTests(unittest.TestCase):
    def test_board_contract(self):
        self.assertEqual((TETRIO_MOVEMENT.height,TETRIO_MOVEMENT.width),(40,10)); self.assertEqual(TETRIO_MOVEMENT.visible_height,20)
    def test_search_exposes_native_180_paths(self):
        board=TETRIO_MOVEMENT.empty_board(); start=TETRIO_MOVEMENT.spawn_state("T")
        placements=enumerate_reachable_placements(board,start,TETRIO_MOVEMENT)
        self.assertTrue(any(MoveAction.ROTATE_180 in p.path for p in placements)); self.assertTrue(any(g[3]==2 for g in unique_landing_geometries(placements)))
    def test_path_sensitive_same_geometry_can_remain_distinct(self):
        board=TETRIO_MOVEMENT.empty_board(); start=TETRIO_MOVEMENT.spawn_state("T")
        placements=enumerate_reachable_placements(board,start,TETRIO_MOVEMENT); grouped={}
        for p in placements: grouped.setdefault(p.landing_state.geometry_key(),set()).add(p.spin_signature)
        self.assertTrue(any(len(v)>1 for v in grouped.values()))

if __name__ == "__main__": unittest.main()
