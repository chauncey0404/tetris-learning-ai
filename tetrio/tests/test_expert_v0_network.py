from __future__ import annotations

import unittest

import numpy as np
import torch

from tetris_ai.learning.listwise import (
    candidate_ranking_metrics,
    masked_candidate_cross_entropy,
)
from tetrio.network.encoding import (
    CANDIDATE_SIZE,
    PACKED_BOARD_BYTES,
    PREVIEW_DEPTH,
    STATE_SIZE,
    dense_candidate_batch,
    dense_state_batch,
)
from tetrio.network.model import TetrioExpertV0Network


class TetrioExpertV0Tests(unittest.TestCase):
    def test_feature_contract_sizes(self):
        self.assertEqual(STATE_SIZE, 449)
        self.assertEqual(CANDIDATE_SIZE, 415)
        self.assertEqual(PACKED_BOARD_BYTES, 50)
        self.assertEqual(PREVIEW_DEPTH, 5)

    def test_dense_batch_encoders(self):
        packed = np.zeros((2, 50), dtype=np.uint8)
        state = dense_state_batch(
            packed,
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([7, 2], dtype=np.uint8),
            np.asarray([[1,2,3,4,5],[0,1,2,3,4]], dtype=np.uint8),
        )
        self.assertEqual(state.shape, (2, STATE_SIZE))

        cand = dense_candidate_batch(
            packed,
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([0, 3], dtype=np.uint8),
            np.asarray([3, 4], dtype=np.int8),
            np.asarray([38, 37], dtype=np.int8),
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([0, 2], dtype=np.uint8),
        )
        self.assertEqual(cand.shape, (2, CANDIDATE_SIZE))

    def test_network_and_masked_listwise_loss(self):
        model = TetrioExpertV0Network()
        state = torch.randn(3, STATE_SIZE)
        candidates = torch.randn(3, 5, CANDIDATE_SIZE)
        mask = torch.tensor(
            [
                [1,1,1,0,0],
                [1,1,1,1,0],
                [1,1,1,1,1],
            ],
            dtype=torch.bool,
        )
        target = torch.tensor([2, 1, 4], dtype=torch.long)
        scores, hold = model(state=state, candidates=candidates)
        self.assertEqual(tuple(scores.shape), (3, 5))
        self.assertEqual(tuple(hold.shape), (3,))

        loss = masked_candidate_cross_entropy(scores, target, mask, label_smoothing=0.05)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

        metrics = candidate_ranking_metrics(scores.detach(), target, mask)
        self.assertIn("top1", metrics)
        self.assertIn("top3", metrics)
        self.assertIn("mrr", metrics)

    def test_masked_target_is_rejected(self):
        scores = torch.zeros(1, 2)
        target = torch.tensor([1])
        mask = torch.tensor([[True, False]])
        with self.assertRaises(ValueError):
            masked_candidate_cross_entropy(scores, target, mask)


if __name__ == "__main__":
    unittest.main()
