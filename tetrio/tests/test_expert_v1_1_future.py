from __future__ import annotations

import unittest

import numpy as np
import torch

from tetrio.future.features import FEATURE_SIZE, normalize_future_features_numpy
from tetrio.future.state_transition import (
    advance_after_lock,
    available_next_branch_pieces,
)
from tetrio.network.model_v1_1 import TetrioExpertV11Network
from tetrio.tools.build_expert_v1_1_future_cache import select_inference_shortlist


class ExpertV11FutureTests(unittest.TestCase):
    def test_no_hold_transition(self):
        s = advance_after_lock(
            active="J",
            hold="S",
            preview=("T", "I", "O", "L", "Z"),
            use_hold=False,
            placed_piece="J",
        )
        self.assertEqual(s.active, "T")
        self.assertEqual(s.hold, "S")
        self.assertEqual(s.preview, ("I", "O", "L", "Z"))

    def test_hold_swap_transition(self):
        s = advance_after_lock(
            active="T",
            hold="S",
            preview=("I", "O", "L", "Z", "J"),
            use_hold=True,
            placed_piece="S",
        )
        self.assertEqual(s.active, "I")
        self.assertEqual(s.hold, "T")
        self.assertEqual(s.preview, ("O", "L", "Z", "J"))

    def test_hold_empty_transition(self):
        s = advance_after_lock(
            active="Z",
            hold=None,
            preview=("J", "T", "I", "O", "L"),
            use_hold=True,
            placed_piece="J",
        )
        self.assertEqual(s.active, "T")
        self.assertEqual(s.hold, "Z")
        self.assertEqual(s.preview, ("I", "O", "L"))

    def test_next_branches_are_exact_from_known_state(self):
        s = advance_after_lock(
            active="J",
            hold="S",
            preview=("T", "I", "O", "L", "Z"),
            use_hold=False,
            placed_piece="J",
        )
        self.assertEqual(
            available_next_branch_pieces(s),
            ((False, "T"), (True, "S")),
        )

    def test_shortlist_keeps_both_branches(self):
        scores = np.asarray([9, 8, 7, 6, 5, 4], dtype=np.float32)
        hold = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.uint8)
        sel = select_inference_shortlist(
            scores,
            hold,
            top_overall=2,
            top_per_branch=1,
        )
        self.assertIn(0, sel)
        self.assertIn(3, sel)

    def test_feature_normalization_contract(self):
        x = np.zeros((2, FEATURE_SIZE), dtype=np.float32)
        y = normalize_future_features_numpy(x)
        self.assertEqual(y.shape, x.shape)

    def test_v11_zero_initialized_reranker_is_exact_v1(self):
        model = TetrioExpertV11Network()
        base = torch.tensor([[3.0, 2.0, 1.0]])
        feat = torch.zeros((1, 3, FEATURE_SIZE))
        hold = torch.tensor([[False, True, False]])
        mask = torch.tensor([[True, True, True]])
        final, residual = model.final_scores(
            base_scores=base,
            raw_features=feat,
            candidate_use_hold=hold,
            mask=mask,
        )
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual)))
        self.assertTrue(torch.allclose(final, base))


if __name__ == "__main__":
    unittest.main()
