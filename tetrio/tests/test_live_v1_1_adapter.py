from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

from tetrio.live_v1_1 import LiveV11Policy, branch_plans
from tetrio.vision.observation import (
    ModelObservation,
    ObservationNotReady,
    build_model_observation,
)


class _MovementStub:
    @staticmethod
    def lift_visible_board(visible):
        visible = np.asarray(visible, dtype=np.uint8)
        out = np.zeros((40, 10), dtype=np.uint8)
        out[-20:, :] = visible
        return out


def _install_ruleset_stub_if_missing():
    try:
        import tetrio.ruleset  # noqa: F401
    except Exception:
        module = types.ModuleType("tetrio.ruleset")
        module.TETRIO_MOVEMENT = _MovementStub()
        sys.modules["tetrio.ruleset"] = module


def _temporal(*, stable=True, hold=None, preview=("I", "O", "S", "Z", "L")):
    board = np.zeros((20, 10), dtype=np.uint8)
    board[19, :4] = 1
    return SimpleNamespace(
        frame_index=123,
        stable_pre_action=stable,
        reason="stable_pre_action" if stable else "active_unresolved",
        locked_board=tuple(tuple(int(v) for v in row) for row in board),
        active=SimpleNamespace(piece="T", confidence=0.97),
        hold_piece=hold,
        preview_queue=preview,
        board_confidence=0.96,
        hold_confidence=0.95,
        preview_confidence=0.98,
        unknown_count=0,
    )


class ObservationAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_ruleset_stub_if_missing()

    def test_rejects_non_ready_snapshot(self):
        with self.assertRaises(ObservationNotReady):
            build_model_observation(_temporal(stable=False))

    def test_lifts_visible_board_to_project_40x10_contract(self):
        obs = build_model_observation(_temporal())
        board = obs.board_array()
        self.assertEqual(board.shape, (40, 10))
        self.assertEqual(int(board[:20].sum()), 0)
        self.assertEqual(int(board[20:].sum()), 4)
        self.assertEqual(obs.active_piece, "T")
        self.assertEqual(obs.preview_queue, ("I", "O", "S", "Z", "L"))

    def test_fingerprint_ignores_visual_active_coordinates(self):
        a = _temporal()
        b = _temporal()
        a.active.x = 3
        a.active.y = 4
        b.active.x = 7
        b.active.y = 12
        self.assertEqual(
            build_model_observation(a).fingerprint,
            build_model_observation(b).fingerprint,
        )

    def test_hold_empty_branch_uses_next0_and_consumes_two(self):
        obs = build_model_observation(_temporal(hold=None))
        no_hold, hold = branch_plans(obs)
        self.assertEqual(no_hold.selected_piece, "T")
        self.assertFalse(no_hold.use_hold)
        self.assertEqual(no_hold.next_active, "I")
        self.assertEqual(hold.mode, "hold_empty")
        self.assertEqual(hold.selected_piece, "I")
        self.assertEqual(hold.hold_after, "T")
        self.assertEqual(hold.next_active, "O")
        self.assertEqual(hold.consume_count, 2)

    def test_hold_swap_branch_uses_current_hold(self):
        obs = build_model_observation(_temporal(hold="J"))
        _, hold = branch_plans(obs)
        self.assertEqual(hold.mode, "hold_swap")
        self.assertEqual(hold.selected_piece, "J")
        self.assertEqual(hold.hold_after, "T")
        self.assertEqual(hold.next_active, "I")
        self.assertEqual(hold.consume_count, 1)


    def test_policy_decide_orchestration_with_stubs(self):
        import torch
        from unittest.mock import patch

        class FakeState:
            def __init__(self, piece, x, y=20, rotation=0):
                self.piece = piece
                self.x = x
                self.y = y
                self.rotation = rotation

            def geometry_key(self):
                return (self.piece, self.x, self.y, self.rotation)

        class FakeAction:
            def __init__(self, value):
                self.value = value

        class FakePlacement:
            def __init__(self, state):
                self.landing_state = state
                self.path = (FakeAction("left"), FakeAction("hard_drop"))

        class FakeScorer:
            def encode_state(self, state):
                return torch.zeros((1, 2), dtype=torch.float32)

            def score_from_state_latent(self, latent, candidates):
                # dense candidate stub stores x in col0 and use_hold in col1.
                return candidates[:, :, 0] + 0.25 * candidates[:, :, 1]

        class FakeModel:
            def __init__(self):
                self.scorer = FakeScorer()

            def eval(self):
                return self

            def final_scores(self, *, base_scores, raw_features, candidate_use_hold, mask):
                residual = 0.10 * candidate_use_hold.to(base_scores.dtype)
                return base_scores + residual, residual

        checkpoint_mod = types.ModuleType("tetrio.network.checkpoint")
        checkpoint_mod.load_expert_v1_1 = lambda path, device: (
            FakeModel(),
            {"format": "tetrio_expert_v1_1", "epoch": 500000},
        )
        network_pkg = types.ModuleType("tetrio.network")
        network_pkg.__path__ = []

        encoding_mod = types.ModuleType("tetrio.network.encoding")
        encoding_mod.pack_board = lambda board: np.zeros((50,), dtype=np.uint8)
        encoding_mod.piece_id = lambda piece: 7 if piece is None else "IOTSZJL".index(piece)
        encoding_mod.dense_state_batch = lambda board, active, hold, preview: np.zeros((1, 8), dtype=np.float32)
        def dense_candidate_batch(board, piece, rotations, xs, ys, use_hold, lines):
            return np.stack((np.asarray(xs, dtype=np.float32), np.asarray(use_hold, dtype=np.float32)), axis=1)
        encoding_mod.dense_candidate_batch = dense_candidate_batch

        fast_mod = types.ModuleType("tetrio.fast_reachability")
        fast_mod.enumerate_tetrio_reachable_geometries_fast = (
            lambda board, piece, max_states: [FakeState(piece, 0), FakeState(piece, 1)]
        )

        movement_mod = types.ModuleType("tetris_ai.core.movement")
        movement_mod.lock_piece = lambda board, state, ruleset: np.asarray(board, dtype=np.uint8).copy()
        movement_mod.clear_lines = lambda board, ruleset: (np.asarray(board, dtype=np.uint8).copy(), 0)
        tetris_ai_pkg = types.ModuleType("tetris_ai")
        tetris_ai_pkg.__path__ = []
        core_pkg = types.ModuleType("tetris_ai.core")
        core_pkg.__path__ = []

        reach_mod = types.ModuleType("tetrio.reachability")
        reach_mod.enumerate_tetrio_reachable_placements = (
            lambda board, piece, max_states: [
                FakePlacement(FakeState(piece, 0)),
                FakePlacement(FakeState(piece, 1)),
            ]
        )

        future_pkg = types.ModuleType("tetrio.future")
        future_pkg.__path__ = []
        future_mod = types.ModuleType("tetrio.future.lookahead")
        class FakeFutureCandidateInput:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        class FakeFutureFeatureConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        future_mod.FutureCandidateInput = FakeFutureCandidateInput
        future_mod.FutureFeatureConfig = FakeFutureFeatureConfig
        future_mod.build_row_future_features = (
            lambda board_before, active, hold, preview, candidates, config: np.zeros((len(candidates), 1), dtype=np.float32)
        )

        tools_pkg = types.ModuleType("tetrio.tools")
        tools_pkg.__path__ = []
        shortlist_mod = types.ModuleType("tetrio.tools.build_expert_v1_1_future_cache")
        shortlist_mod.select_inference_shortlist = (
            lambda scores, use_hold, top_overall, top_per_branch: list(range(len(scores)))
        )

        rule_mod = types.ModuleType("tetrio.ruleset")
        rule_mod.TETRIO_MOVEMENT = _MovementStub()

        stubs = {
            "tetrio.network": network_pkg,
            "tetrio.network.checkpoint": checkpoint_mod,
            "tetrio.network.encoding": encoding_mod,
            "tetrio.fast_reachability": fast_mod,
            "tetrio.reachability": reach_mod,
            "tetrio.future": future_pkg,
            "tetrio.future.lookahead": future_mod,
            "tetrio.tools": tools_pkg,
            "tetrio.tools.build_expert_v1_1_future_cache": shortlist_mod,
            "tetrio.ruleset": rule_mod,
            "tetris_ai": tetris_ai_pkg,
            "tetris_ai.core": core_pkg,
            "tetris_ai.core.movement": movement_mod,
        }

        obs = ModelObservation(
            frame_index=1,
            board=tuple(tuple(0 for _ in range(10)) for _ in range(40)),
            active_piece="T",
            hold_piece=None,
            preview_queue=("I", "O", "S", "Z", "L"),
            board_confidence=1.0,
            active_confidence=1.0,
            hold_confidence=1.0,
            preview_confidence=1.0,
            fingerprint="stub",
        )

        with patch.dict(sys.modules, stubs, clear=False):
            policy = LiveV11Policy("fake.pt", device="cpu")
            decision = policy.decide(obs)

        self.assertEqual(len(decision.candidates), 4)
        self.assertTrue(decision.chosen.use_hold)
        self.assertEqual(decision.chosen.state.x, 1)
        self.assertEqual(decision.movement_path[-1], "hard_drop")
        self.assertEqual(decision.branch.mode, "hold_empty")
        self.assertAlmostEqual(decision.chosen.final_score, decision.chosen.base_score + 0.10, places=6)

    def test_policy_can_defer_exact_path_for_live_retarget(self):
        import torch
        from unittest.mock import patch

        class FakeState:
            def __init__(self, piece, x, y=20, rotation=0):
                self.piece=piece; self.x=x; self.y=y; self.rotation=rotation
            def geometry_key(self): return (self.piece,self.x,self.y,self.rotation)
        class FakeScorer:
            def encode_state(self, state): return torch.zeros((1,2),dtype=torch.float32)
            def score_from_state_latent(self, latent, candidates): return candidates[:,:,0]
        class FakeModel:
            scorer=FakeScorer()
            def eval(self): return self
            def final_scores(self, *, base_scores, raw_features, candidate_use_hold, mask):
                return base_scores, torch.zeros_like(base_scores)

        checkpoint_mod=types.ModuleType("tetrio.network.checkpoint")
        checkpoint_mod.load_expert_v1_1=lambda path,device:(FakeModel(),{"format":"tetrio_expert_v1_1"})
        network_pkg=types.ModuleType("tetrio.network"); network_pkg.__path__=[]
        encoding_mod=types.ModuleType("tetrio.network.encoding")
        encoding_mod.pack_board=lambda b: np.zeros((50,),np.uint8)
        encoding_mod.piece_id=lambda p: 0
        encoding_mod.dense_state_batch=lambda *a: np.zeros((1,8),np.float32)
        encoding_mod.dense_candidate_batch=lambda board,piece,rot,x,y,use_hold,lines: np.stack([np.asarray(x,np.float32),np.asarray(use_hold,np.float32)],axis=1)
        fast_mod=types.ModuleType("tetrio.fast_reachability")
        fast_mod.enumerate_tetrio_reachable_geometries_fast=lambda board,piece,max_states:[FakeState(piece,0)]
        movement_mod=types.ModuleType("tetris_ai.core.movement")
        movement_mod.lock_piece=lambda board,state,ruleset: np.asarray(board,np.uint8).copy()
        movement_mod.clear_lines=lambda board,ruleset:(np.asarray(board,np.uint8).copy(),0)
        future_pkg=types.ModuleType("tetrio.future"); future_pkg.__path__=[]
        future_mod=types.ModuleType("tetrio.future.lookahead")
        class FCI:
            def __init__(self,**kw): self.__dict__.update(kw)
        class FFC:
            def __init__(self,**kw): self.__dict__.update(kw)
        future_mod.FutureCandidateInput=FCI; future_mod.FutureFeatureConfig=FFC
        future_mod.build_row_future_features=lambda **kw: np.zeros((len(kw["candidates"]),1),np.float32)
        tools_pkg=types.ModuleType("tetrio.tools"); tools_pkg.__path__=[]
        shortlist_mod=types.ModuleType("tetrio.tools.build_expert_v1_1_future_cache")
        shortlist_mod.select_inference_shortlist=lambda scores,use_hold,top_overall,top_per_branch:list(range(len(scores)))
        rule_mod=types.ModuleType("tetrio.ruleset"); rule_mod.TETRIO_MOVEMENT=_MovementStub()
        tetris_ai_pkg=types.ModuleType("tetris_ai"); tetris_ai_pkg.__path__=[]
        core_pkg=types.ModuleType("tetris_ai.core"); core_pkg.__path__=[]
        stubs={
            "tetrio.network":network_pkg,"tetrio.network.checkpoint":checkpoint_mod,
            "tetrio.network.encoding":encoding_mod,"tetrio.fast_reachability":fast_mod,
            "tetrio.future":future_pkg,"tetrio.future.lookahead":future_mod,
            "tetrio.tools":tools_pkg,"tetrio.tools.build_expert_v1_1_future_cache":shortlist_mod,
            "tetrio.ruleset":rule_mod,"tetris_ai":tetris_ai_pkg,"tetris_ai.core":core_pkg,
            "tetris_ai.core.movement":movement_mod,
        }
        obs=ModelObservation(1,tuple(tuple(0 for _ in range(10)) for _ in range(40)),"T",None,("I","O","S","Z","L"),1,1,1,1,"defer")
        with patch.dict(sys.modules,stubs,clear=False), patch(
            "tetrio.live_v1_1._exact_path_for_geometry",
            side_effect=AssertionError("exact path must be deferred"),
        ):
            policy=LiveV11Policy("fake.pt",device="cpu")
            decision=policy.decide(obs,resolve_exact_path=False)
        self.assertEqual(decision.movement_path,())
        self.assertIn("exact_path_deferred",decision.timings_ms)

    def test_invalid_preview_is_rejected(self):
        with self.assertRaises(ObservationNotReady):
            build_model_observation(_temporal(preview=("I", "O", "S")))


if __name__ == "__main__":
    unittest.main()
