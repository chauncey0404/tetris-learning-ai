from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tetrio.live_pipeline import (
    PreparedCandidate,
    PredictedNextTemplate,
    SpeculationSeed,
    SpeculativePreparation,
    build_speculative_preparation,
    decide_prepared,
    decisions_equivalent,
    observation_matches_template,
    observation_generation_key,
    predict_next_template,
    template_match_reason,
)
from tetrio.live_v1_1 import BranchPlan, LiveCandidate, LiveDecision, LiveV11PolicyConfig
from tetrio.vision.observation import ModelObservation


class FakeState:
    def __init__(self, piece, x, y=30, rotation=0):
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
    def __init__(self, state, path=("hard_drop",)):
        self.landing_state = state
        self.path = tuple(FakeAction(x) for x in path)


def _obs(active="T", hold="J", preview=("I", "O", "S", "Z", "L"), board=None, fp="src"):
    if board is None:
        board = np.zeros((40, 10), dtype=np.uint8)
    return ModelObservation(
        frame_index=1,
        board=tuple(tuple(int(v) for v in row) for row in board),
        active_piece=active,
        hold_piece=hold,
        preview_queue=tuple(preview),
        board_confidence=1.0,
        active_confidence=1.0,
        hold_confidence=1.0,
        preview_confidence=1.0,
        fingerprint=fp,
    )


def _state_transition_stub():
    module = types.ModuleType("tetrio.future.state_transition")

    def advance_after_lock(*, active, hold, preview, use_hold, placed_piece=None):
        preview = tuple(preview)
        if not use_hold:
            return SimpleNamespace(active=preview[0], hold=hold, preview=preview[1:])
        if hold is None:
            return SimpleNamespace(active=preview[1], hold=active, preview=preview[2:])
        return SimpleNamespace(active=preview[0], hold=active, preview=preview[1:])

    module.advance_after_lock = advance_after_lock
    return module


class PipelineContractTests(unittest.TestCase):

    def test_generation_key_ignores_board_fingerprint_jitter(self):
        a = _obs(board=np.zeros((40, 10), dtype=np.uint8), fp="a")
        board = np.zeros((40, 10), dtype=np.uint8)
        board[39, 0] = 1
        b = _obs(board=board, fp="b")
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertEqual(observation_generation_key(a), observation_generation_key(b))

    def test_generation_key_changes_when_queue_or_hold_changes(self):
        a = _obs(active="T", hold="J", preview=("I", "O", "S", "Z", "L"))
        shifted = _obs(active="I", hold="J", preview=("O", "S", "Z", "L", "T"))
        held = _obs(active="J", hold="T", preview=("I", "O", "S", "Z", "L"))
        self.assertNotEqual(observation_generation_key(a), observation_generation_key(shifted))
        self.assertNotEqual(observation_generation_key(a), observation_generation_key(held))

    def test_predict_next_template_no_hold_keeps_four_known_preview(self):
        seed = SpeculationSeed(
            source_fingerprint="a",
            source_active="T",
            source_hold="J",
            source_preview=("I", "O", "S", "Z", "L"),
            board_after=tuple(tuple(0 for _ in range(10)) for _ in range(40)),
            placed_piece="T",
            use_hold=False,
            branch_mode="no_hold",
        )
        with patch.dict(sys.modules, {"tetrio.future.state_transition": _state_transition_stub()}):
            t = predict_next_template(seed)
        self.assertEqual(t.active_piece, "I")
        self.assertEqual(t.hold_piece, "J")
        self.assertEqual(t.preview_prefix, ("O", "S", "Z", "L"))
        self.assertEqual(t.missing_preview, 1)

    def test_predict_next_template_hold_empty_keeps_three_known_preview(self):
        seed = SpeculationSeed(
            source_fingerprint="a",
            source_active="T",
            source_hold=None,
            source_preview=("I", "O", "S", "Z", "L"),
            board_after=tuple(tuple(0 for _ in range(10)) for _ in range(40)),
            placed_piece="I",
            use_hold=True,
            branch_mode="hold_empty",
        )
        with patch.dict(sys.modules, {"tetrio.future.state_transition": _state_transition_stub()}):
            t = predict_next_template(seed)
        self.assertEqual(t.active_piece, "O")
        self.assertEqual(t.hold_piece, "T")
        self.assertEqual(t.preview_prefix, ("S", "Z", "L"))
        self.assertEqual(t.missing_preview, 2)

    def test_template_match_rejects_tail_safe_but_prefix_mismatch(self):
        board = np.zeros((40, 10), dtype=np.uint8)
        template = PredictedNextTemplate(
            source_fingerprint="src",
            board=tuple(tuple(0 for _ in range(10)) for _ in range(40)),
            active_piece="I",
            hold_piece="J",
            preview_prefix=("O", "S", "Z", "L"),
            missing_preview=1,
            source_branch_mode="no_hold",
        )
        good = _obs(active="I", hold="J", preview=("O", "S", "Z", "L", "T"), board=board)
        bad = _obs(active="I", hold="J", preview=("O", "T", "Z", "L", "S"), board=board)
        self.assertTrue(observation_matches_template(good, template))
        self.assertEqual(template_match_reason(bad, template), "preview_prefix_mismatch")

    def test_build_speculative_preparation_precomputes_features_but_defers_paths(self):
        seed = SpeculationSeed(
            source_fingerprint="src",
            source_active="T",
            source_hold="J",
            source_preview=("I", "O", "S", "Z", "L"),
            board_after=tuple(tuple(0 for _ in range(10)) for _ in range(40)),
            placed_piece="T",
            use_hold=False,
            branch_mode="no_hold",
        )

        future_pkg = types.ModuleType("tetrio.future")
        future_pkg.__path__ = []
        look = types.ModuleType("tetrio.future.lookahead")
        class FCI:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class FFC:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        look.FutureCandidateInput = FCI
        look.FutureFeatureConfig = FFC
        look.build_row_future_features = lambda **kw: np.arange(len(kw["candidates"])*2, dtype=np.float32).reshape(len(kw["candidates"]),2)

        rules = types.ModuleType("tetrio.ruleset")
        rules.TETRIO_MOVEMENT = object()
        movement = types.ModuleType("tetris_ai.core.movement")
        movement.lock_piece = lambda board, state, ruleset: np.asarray(board, dtype=np.uint8).copy()
        movement.clear_lines = lambda board, ruleset: (np.asarray(board, dtype=np.uint8).copy(), 0)
        tetris_ai = types.ModuleType("tetris_ai"); tetris_ai.__path__=[]
        core = types.ModuleType("tetris_ai.core"); core.__path__=[]

        with patch.dict(sys.modules, {
            "tetrio.future": future_pkg,
            "tetrio.future.lookahead": look,
            "tetrio.future.state_transition": _state_transition_stub(),
            "tetrio.ruleset": rules,
            "tetris_ai": tetris_ai,
            "tetris_ai.core": core,
            "tetris_ai.core.movement": movement,
        }), patch("tetrio.live_pipeline._landing_states", side_effect=lambda board,piece,config:[FakeState(piece,0),FakeState(piece,1)]), patch("tetrio.live_pipeline._reference_best_placements", side_effect=lambda board,piece,max_states:[FakePlacement(FakeState(piece,0),("left","hard_drop")),FakePlacement(FakeState(piece,1),("right","hard_drop"))]):
            prepared = build_speculative_preparation(seed, LiveV11PolicyConfig())
        self.assertEqual(len(prepared.candidates), 4)
        self.assertEqual(sum(x.spawn_path is not None for x in prepared.candidates), 0)
        self.assertEqual(prepared.candidates[0].future_features.shape, (2,))

    def test_decisions_equivalent_checks_geometry_hold_path_and_score(self):
        state = FakeState("T", 3)
        cand = LiveCandidate(state, np.zeros((40,10),np.uint8), 0, False, "no_hold", 1.0, 0.1, 1.1)
        plan = BranchPlan("T",False,"J","I",1,"no_hold")
        kwargs = dict(
            observation_fingerprint="x", active_piece="T", hold_piece="J",
            preview_queue=("I","O","S","Z","L"), candidates=(cand,), shortlist_indices=(0,),
            chosen_index=0, branch=plan, movement_path=("hard_drop",), timings_ms={"total":1.0}
        )
        a=LiveDecision(**kwargs); b=LiveDecision(**kwargs)
        self.assertTrue(decisions_equivalent(a,b))

    def test_decide_prepared_uses_precomputed_features_and_path(self):
        class FakeScorer:
            def encode_state(self, state): return torch.zeros((1,2),dtype=torch.float32)
            def score_from_state_latent(self, latent, candidates):
                return candidates[:,:,0] + 0.2*candidates[:,:,1]
        class FakeModel:
            scorer=FakeScorer()
            def final_scores(self, *, base_scores, raw_features, candidate_use_hold, mask):
                residual=0.1*candidate_use_hold.to(base_scores.dtype)
                return base_scores+residual,residual
        class FakePolicy:
            torch=torch; device=torch.device("cpu"); amp_dtype=torch.float32; model=FakeModel()
            config=LiveV11PolicyConfig()
            def _state_latent(self, observation): return self.model.scorer.encode_state(None)

        encoding = types.ModuleType("tetrio.network.encoding")
        encoding.pack_board=lambda b: np.zeros((50,),np.uint8)
        encoding.piece_id=lambda p: 0
        def dense_candidate_batch(board,piece,rot,x,y,use_hold,lines):
            return np.stack([np.asarray(x,np.float32),np.asarray(use_hold,np.float32)],axis=1)
        encoding.dense_candidate_batch=dense_candidate_batch
        network=types.ModuleType("tetrio.network"); network.__path__=[]
        tools=types.ModuleType("tetrio.tools"); tools.__path__=[]
        shortlist=types.ModuleType("tetrio.tools.build_expert_v1_1_future_cache")
        shortlist.select_inference_shortlist=lambda scores,use_hold,top_overall,top_per_branch:list(range(len(scores)))

        board=tuple(tuple(0 for _ in range(10)) for _ in range(40))
        template=PredictedNextTemplate("src",board,"I","J",("O","S","Z","L"),1,"no_hold")
        p0=PreparedCandidate(FakeState("I",0),np.zeros((40,10),np.uint8),0,False,"no_hold",np.zeros(2,np.float32),("left","hard_drop"))
        p1=PreparedCandidate(FakeState("J",1),np.zeros((40,10),np.uint8),0,True,"hold_swap",np.ones(2,np.float32),("right","hard_drop"))
        plans=(BranchPlan("I",False,"J","O",1,"no_hold"),BranchPlan("J",True,"I","O",1,"hold_swap"))
        prepared=SpeculativePreparation(template,(p0,p1),plans,{"total":1.0})
        obs=_obs(active="I",hold="J",preview=("O","S","Z","L","T"),fp="actual")
        with patch.dict(sys.modules, {
            "tetrio.network":network,
            "tetrio.network.encoding":encoding,
            "tetrio.tools":tools,
            "tetrio.tools.build_expert_v1_1_future_cache":shortlist,
        }):
            out=decide_prepared(FakePolicy(),obs,prepared)
        self.assertTrue(out.chosen.use_hold)
        self.assertEqual(out.movement_path,("right","hard_drop"))
        self.assertEqual(out.branch.mode,"hold_swap")

        with patch.dict(sys.modules, {
            "tetrio.network":network,
            "tetrio.network.encoding":encoding,
            "tetrio.tools":tools,
            "tetrio.tools.build_expert_v1_1_future_cache":shortlist,
        }):
            deferred=decide_prepared(
                FakePolicy(),obs,prepared,resolve_exact_path=False
            )
        self.assertEqual(deferred.movement_path,())
        self.assertIn("exact_path_deferred",deferred.timings_ms)


if __name__ == "__main__":
    unittest.main()
