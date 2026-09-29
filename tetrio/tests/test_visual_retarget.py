from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tetrio.vision.retarget import (
    build_retarget_request,
    retarget_live_decision,
    retarget_request_stale_reason,
    retarget_to_landing,
)


class S:
    def __init__(self, piece="I", x=3, y=20, rotation=0, last=None):
        self.piece = piece
        self.x = x
        self.y = y
        self.rotation = rotation
        self.last = last

    def geometry_key(self):
        return (self.piece, self.x, self.y, self.rotation % 4)

    def search_key(self):
        return self.geometry_key() + (self.last,)


class Act:
    def __init__(self, value):
        self.value = value


CW = Act("cw")
RIGHT = Act("right")
LEFT = Act("left")


def _mods(*, allow_normalize=True, hard_drop_counter=None):
    movement_rules = SimpleNamespace(
        movement_actions=((CW, RIGHT, LEFT) if allow_normalize else (RIGHT, LEFT))
    )
    rules = types.ModuleType("tetrio.ruleset")
    rules.TETRIO_MOVEMENT = movement_rules

    movement = types.ModuleType("tetris_ai.core.movement")

    def apply_action(board, state, action, ruleset):
        if action.value == "cw":
            return S(state.piece, state.x, state.y, 0, "cw")
        if action.value == "right":
            if state.x >= 6:
                return None
            return S(state.piece, state.x + 1, state.y, state.rotation, "right")
        if action.value == "left":
            if state.x <= 0:
                return None
            return S(state.piece, state.x - 1, state.y, state.rotation, "left")
        return None

    def hard_drop(board, state, ruleset):
        if hard_drop_counter is not None:
            hard_drop_counter[0] += 1
        return S(state.piece, state.x, 35, state.rotation, state.last), 15

    movement.apply_action = apply_action
    movement.hard_drop = hard_drop

    types_mod = types.ModuleType("tetris_ai.core.types")
    types_mod.MoveAction = SimpleNamespace(HARD_DROP=Act("hard_drop"))

    tet = types.ModuleType("tetris_ai.core.tetrominoes")
    tet.occupied_cells = lambda state: (
        (state.x, state.y),
        (state.x + 1, state.y),
        (state.x + 2, state.y),
        (state.x + 3, state.y),
    )

    tetris = types.ModuleType("tetris_ai")
    tetris.__path__ = []
    core = types.ModuleType("tetris_ai.core")
    core.__path__ = []
    return {
        "tetrio.ruleset": rules,
        "tetris_ai": tetris,
        "tetris_ai.core": core,
        "tetris_ai.core.movement": movement,
        "tetris_ai.core.types": types_mod,
        "tetris_ai.core.tetrominoes": tet,
    }


class RetargetTests(unittest.TestCase):

    def test_target_search_defers_hard_drop_until_x_rotation_align(self):
        calls = [0]
        with patch.dict(sys.modules, _mods(allow_normalize=True, hard_drop_counter=calls)), patch(
            "tetrio.vision.retarget.visual_active_state_candidates",
            return_value=(S(x=0, rotation=0),),
        ):
            out = retarget_to_landing(
                np.zeros((40, 10), np.uint8),
                SimpleNamespace(piece="I"),
                S("I", 5, 35, 0),
            )
        self.assertTrue(out.safe)
        self.assertLessEqual(calls[0], 2)
        self.assertGreater(out.search_nodes, calls[0])

    def test_completed_request_is_stale_after_visual_active_advances(self):
        rules = types.ModuleType("tetrio.ruleset")

        class M:
            @staticmethod
            def lift_visible_board(v):
                out = np.zeros((40, 10), np.uint8)
                out[-20:] = v
                return out

        rules.TETRIO_MOVEMENT = M()
        source_board = np.zeros((40, 10), np.uint8)
        source = SimpleNamespace(active_piece="T", board_array=lambda: source_board)
        decision = SimpleNamespace(
            chosen=SimpleNamespace(
                use_hold=False,
                state=S("T", 3, 35, 0),
            )
        )
        locked = np.zeros((20, 10), np.uint8)
        fresh = SimpleNamespace(
            active=SimpleNamespace(piece="T", rotation=0, x=3, y=2),
            locked_board=locked,
        )
        with patch.dict(sys.modules, {"tetrio.ruleset": rules}):
            request, failure = build_retarget_request(source, decision, fresh)
            self.assertIsNone(failure)
            advanced = SimpleNamespace(
                active=SimpleNamespace(piece="T", rotation=0, x=3, y=3),
                locked_board=locked,
            )
            reason = retarget_request_stale_reason(request, advanced)
        self.assertEqual(reason, "active_visual_state_advanced")

    def test_target_directed_common_path_across_rotation_aliases_is_safe(self):
        with patch.dict(sys.modules, _mods(allow_normalize=True)), patch(
            "tetrio.vision.retarget.visual_active_state_candidates",
            return_value=(S(rotation=0), S(rotation=2)),
        ):
            out = retarget_to_landing(
                np.zeros((40, 10), np.uint8),
                SimpleNamespace(piece="I"),
                S("I", 5, 35, 0),
            )
        self.assertTrue(out.safe)
        self.assertEqual(out.movement_path[-1], "hard_drop")
        self.assertIn("cw", out.movement_path)
        self.assertLess(out.search_nodes, 50)

    def test_no_common_path_across_aliases_fails_closed(self):
        with patch.dict(sys.modules, _mods(allow_normalize=False)), patch(
            "tetrio.vision.retarget.visual_active_state_candidates",
            return_value=(S(rotation=0), S(rotation=2)),
        ):
            out = retarget_to_landing(
                np.zeros((40, 10), np.uint8),
                SimpleNamespace(piece="I"),
                S("I", 5, 35, 0),
                max_states=200,
            )
        self.assertFalse(out.safe)
        self.assertEqual(out.reason, "rotation_alias_has_no_common_safe_path")

    def test_target_requires_exact_policy_rotation_not_only_same_cells(self):
        with patch.dict(sys.modules, _mods(allow_normalize=False)), patch(
            "tetrio.vision.retarget.visual_active_state_candidates",
            return_value=(S(x=5, rotation=2),),
        ):
            out = retarget_to_landing(
                np.zeros((40, 10), np.uint8),
                SimpleNamespace(piece="I"),
                S("I", 5, 35, 0),
                max_states=100,
            )
        self.assertFalse(out.safe)
        self.assertEqual(out.reason, "target_unreachable_from_possible_visual_state")

    def test_hold_decision_requires_reobserve_before_retarget(self):
        decision = SimpleNamespace(chosen=SimpleNamespace(use_hold=True, state=S()))
        src = SimpleNamespace(active_piece="T")
        out = retarget_live_decision(src, decision, SimpleNamespace())
        self.assertFalse(out.safe)
        self.assertEqual(
            out.reason,
            "hold_must_be_executed_and_reobserved_before_retarget",
        )

    def test_locked_board_change_marks_decision_stale(self):
        rules = types.ModuleType("tetrio.ruleset")

        class M:
            @staticmethod
            def lift_visible_board(v):
                out = np.zeros((40, 10), np.uint8)
                out[-20:] = v
                return out

        rules.TETRIO_MOVEMENT = M()
        source_board = np.zeros((40, 10), np.uint8)
        source = SimpleNamespace(active_piece="T", board_array=lambda: source_board)
        decision = SimpleNamespace(chosen=SimpleNamespace(use_hold=False, state=S("T")))
        locked = np.zeros((20, 10), np.uint8)
        locked[19, 0] = 1
        fresh = SimpleNamespace(active=SimpleNamespace(piece="T"), locked_board=locked)
        with patch.dict(sys.modules, {"tetrio.ruleset": rules}):
            out = retarget_live_decision(source, decision, fresh)
        self.assertFalse(out.safe)
        self.assertEqual(out.reason, "locked_board_changed_decision_stale")


if __name__ == "__main__":
    unittest.main()
