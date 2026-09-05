from __future__ import annotations
import unittest
from tetrio import TETRIO_MOVEMENT, TetrioSRSPlusRotationSystem
from tetris_ai.battle.movement import try_rotate
from tetris_ai.battle.rotation.srs import SRSRotationSystem
from tetris_ai.battle.types import MoveAction, PieceState

class TetrioRotationTests(unittest.TestCase):
    def test_srs_plus_i_0_to_r_differs_from_standard_srs(self):
        standard=SRSRotationSystem(); tetrio=TetrioSRSPlusRotationSystem()
        _, a=standard.kick_tests("I",0,MoveAction.CW); target,b=tetrio.kick_tests("I",0,MoveAction.CW)
        self.assertEqual(target,1); self.assertEqual(b,((0,0),(1,0),(-2,0),(-2,1),(1,-2))); self.assertNotEqual(a,b)
    def test_non_i_180_has_six_ordered_tests(self):
        target,kicks=TetrioSRSPlusRotationSystem().kick_tests("T",0,MoveAction.ROTATE_180)
        self.assertEqual(target,2); self.assertEqual(kicks,((0,0),(0,-1),(1,-1),(-1,-1),(1,0),(-1,0)))
    def test_i_180_is_native(self):
        target,kicks=TetrioSRSPlusRotationSystem().kick_tests("I",0,MoveAction.ROTATE_180)
        self.assertEqual(target,2); self.assertEqual(kicks,((0,0),(0,-1)))
    def test_floor_kick_executes(self):
        board=TETRIO_MOVEMENT.empty_board(); state=PieceState(piece="T",x=3,y=38,rotation=0)
        attempt=try_rotate(board,state,MoveAction.CW,TETRIO_MOVEMENT)
        self.assertTrue(attempt.success); self.assertEqual(attempt.kick_index,2); self.assertEqual((attempt.state.x,attempt.state.y,attempt.state.rotation),(2,37,1))

if __name__ == "__main__": unittest.main()
