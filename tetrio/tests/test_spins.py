from __future__ import annotations
import unittest
from tetrio import TETRIO_MULTIPLAYER
from tetris_ai.battle.spins import SpinKind
from tetris_ai.battle.types import MoveAction, PieceState, ReachablePlacement, RotationTrace

def rotated_state(piece, *, x=3, y=36, rotation=0, action=MoveAction.CW, kick_index=0):
    if action == MoveAction.CW: fr=(rotation-1)%4
    elif action == MoveAction.CCW: fr=(rotation+1)%4
    else: fr=(rotation+2)%4
    return PieceState(piece=piece,x=x,y=y,rotation=rotation,last_action=action,last_rotation=RotationTrace(action=action,from_rotation=fr,to_rotation=rotation,kick_index=kick_index,kick_dx=0,kick_dy=0))

def placement(state, drop_distance=0):
    landing=PieceState(piece=state.piece,x=state.x,y=state.y+drop_distance,rotation=state.rotation,last_action=state.last_action,last_rotation=state.last_rotation)
    return ReachablePlacement(pre_drop_state=state,landing_state=landing,path=(state.last_action,MoveAction.HARD_DROP),drop_distance=drop_distance)

class TetrioSpinTests(unittest.TestCase):
    def _t_board(self,*offs):
        board=TETRIO_MULTIPLAYER.movement.empty_board(); px,py=4,37
        for dx,dy in offs: board[py+dy,px+dx]=1
        return board
    def test_profile_uses_all_mini_plus_and_180(self):
        self.assertEqual(TETRIO_MULTIPLAYER.spins.name,"TETR.IO All-Mini+"); self.assertTrue(TETRIO_MULTIPLAYER.movement.allow_180)
    def test_t_three_corner_full(self):
        state=rotated_state("T"); board=self._t_board((-1,-1),(1,-1),(-1,1))
        self.assertEqual(TETRIO_MULTIPLAYER.spins.classify(board,placement(state),TETRIO_MULTIPLAYER.movement).kind,SpinKind.FULL)
    def test_positive_hard_drop_distance_invalidates_spin(self):
        state=rotated_state("T",y=35); board=TETRIO_MULTIPLAYER.movement.empty_board()
        self.assertEqual(TETRIO_MULTIPLAYER.spins.classify(board,placement(state,1),TETRIO_MULTIPLAYER.movement).kind,SpinKind.NONE)
    def test_180_kick_index_four_not_fifth_90_exception(self):
        state=rotated_state("T",action=MoveAction.ROTATE_180,kick_index=4); board=self._t_board((-1,-1),(-1,1),(1,1))
        self.assertEqual(TETRIO_MULTIPLAYER.spins.classify(board,placement(state),TETRIO_MULTIPLAYER.movement).kind,SpinKind.MINI)
    def test_classifier_does_not_mutate_board(self):
        state=rotated_state("T"); board=self._t_board((-1,-1),(1,-1),(-1,1)); before=board.copy()
        TETRIO_MULTIPLAYER.spins.classify(board,placement(state),TETRIO_MULTIPLAYER.movement); self.assertTrue((board==before).all())

if __name__ == "__main__": unittest.main()
