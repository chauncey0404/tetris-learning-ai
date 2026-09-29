from __future__ import annotations

import unittest

import numpy as np
import torch

from tetrio.network.cache_v1 import compact_batches_from_pair
from tetrio.network.model_v1 import TetrioExpertV1Network
from tetrio.tools.build_expert_v1_counterfactual_cache import (
    other_branch_piece,
)
from tetrio.tools.train_expert_v1 import (
    structure_margin_loss,
)


class ExpertV1Tests(unittest.TestCase):
    def test_counterfactual_piece_when_expert_held(self):
        piece, use_hold = other_branch_piece(
            active_id=0,  # I
            hold_id=1,    # O
            preview_ids=np.asarray([2, 3, 4, 5, 6], dtype=np.uint8),
            expert_use_hold=1,
        )
        self.assertEqual(piece, "I")
        self.assertEqual(use_hold, 0)

    def test_counterfactual_piece_hold_swap(self):
        piece, use_hold = other_branch_piece(
            active_id=0,
            hold_id=1,
            preview_ids=np.asarray([2, 3, 4, 5, 6], dtype=np.uint8),
            expert_use_hold=0,
        )
        self.assertEqual(piece, "O")
        self.assertEqual(use_hold, 1)

    def test_counterfactual_piece_hold_empty_uses_preview0(self):
        piece, use_hold = other_branch_piece(
            active_id=0,
            hold_id=7,
            preview_ids=np.asarray([2, 3, 4, 5, 6], dtype=np.uint8),
            expert_use_hold=0,
        )
        self.assertEqual(piece, "T")
        self.assertEqual(use_hold, 1)

    def test_v1_model_has_no_hold_head(self):
        model = TetrioExpertV1Network()
        self.assertFalse(hasattr(model, "hold_head"))

    def test_structure_loss_downstream_contract(self):
        scores = torch.tensor([[2.0, 1.5, 3.0]])
        mask = torch.tensor([[True, True, True]])
        target = torch.tensor([0])
        holes = torch.tensor([[0, 1, 2]])
        before = torch.tensor([0])

        loss, expert_risky, min_holes = structure_margin_loss(
            scores=scores,
            mask=mask,
            target=target,
            holes=holes,
            holes_before=before,
            margin=0.25,
        )
        self.assertGreater(float(loss), 0.0)
        self.assertFalse(bool(expert_risky[0]))
        self.assertEqual(int(min_holes[0]), 0)

    def test_risky_expert_is_detected_but_not_structurally_overridden(self):
        scores = torch.tensor([[2.0, 1.5, 1.0]])
        mask = torch.tensor([[True, True, True]])
        target = torch.tensor([0])
        holes = torch.tensor([[2, 0, 1]])
        before = torch.tensor([0])

        loss, expert_risky, _ = structure_margin_loss(
            scores=scores,
            mask=mask,
            target=target,
            holes=holes,
            holes_before=before,
            margin=0.25,
        )
        self.assertTrue(bool(expert_risky[0]))
        # No margin auxiliary is applied when the expert label itself is risky;
        # the row is handled by CE downweighting instead.
        self.assertAlmostEqual(float(loss), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
