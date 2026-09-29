from __future__ import annotations

import unittest

import torch

from tetrio.future.features import FEATURE_SIZE, feature_index
from tetrio.stateful.safety import (
    change_diagnostics,
    classify_expert_vs_baseline,
    relation_row_weights,
    unsafe_expert_preservation_loss,
)


H = feature_index("holes_after")
HEIGHT = feature_index("max_height_after")
NEXT_H = feature_index("next_min_holes")
NEXT_HEIGHT = feature_index("next_min_height")
LINES = feature_index("lines_now")
DEAD = feature_index("next_dead_end")


def set_candidate(
    f: torch.Tensor,
    row: int,
    cand: int,
    *,
    holes: float,
    height: float,
    next_holes: float,
    next_height: float,
    lines: float,
) -> None:
    f[row, cand, H] = holes
    f[row, cand, HEIGHT] = height
    f[row, cand, NEXT_H] = next_holes
    f[row, cand, NEXT_HEIGHT] = next_height
    f[row, cand, LINES] = lines
    f[row, cand, DEAD] = 0.0


class StatefulSafetyTests(unittest.TestCase):
    def make_fixture(self):
        # Four rows:
        # 0 agreement
        # 1 expert safely dominates baseline
        # 2 baseline safely dominates expert
        # 3 ambiguous trade-off
        scores = torch.tensor(
            [
                [2.0, 1.0],
                [2.0, 1.0],
                [2.0, 1.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        features = torch.zeros(
            4,
            2,
            FEATURE_SIZE,
            dtype=torch.float32,
        )
        mask = torch.ones(4, 2, dtype=torch.bool)

        # row 0: equivalent structure; expert==baseline.
        set_candidate(
            features, 0, 0,
            holes=0, height=4, next_holes=0, next_height=4, lines=0,
        )
        set_candidate(
            features, 0, 1,
            holes=1, height=5, next_holes=1, next_height=5, lines=0,
        )

        # row 1: expert candidate 1 is strictly no-worse / better.
        set_candidate(
            features, 1, 0,
            holes=1, height=5, next_holes=1, next_height=5, lines=0,
        )
        set_candidate(
            features, 1, 1,
            holes=0, height=4, next_holes=0, next_height=4, lines=1,
        )

        # row 2: baseline candidate 0 dominates expert candidate 1.
        set_candidate(
            features, 2, 0,
            holes=0, height=4, next_holes=0, next_height=4, lines=1,
        )
        set_candidate(
            features, 2, 1,
            holes=2, height=7, next_holes=2, next_height=7, lines=0,
        )

        # row 3: trade holes for height => neither dominates.
        set_candidate(
            features, 3, 0,
            holes=0, height=7, next_holes=0, next_height=7, lines=0,
        )
        set_candidate(
            features, 3, 1,
            holes=1, height=4, next_holes=1, next_height=4, lines=0,
        )

        expert = torch.tensor([0, 1, 1, 1], dtype=torch.long)
        recall = torch.ones(4, dtype=torch.bool)
        return scores, features, mask, expert, recall

    def test_relation_partition(self):
        scores, features, mask, expert, recall = self.make_fixture()
        rel = classify_expert_vs_baseline(
            baseline_scores=scores,
            raw_features=features,
            inference_mask=mask,
            expert_local=expert,
            expert_in_shortlist=recall,
        )
        self.assertEqual(rel.agreement.tolist(), [True, False, False, False])
        self.assertEqual(rel.safe_expert.tolist(), [False, True, False, False])
        self.assertEqual(rel.unsafe_expert.tolist(), [False, False, True, False])
        self.assertEqual(rel.ambiguous.tolist(), [False, False, False, True])

    def test_row_weights_follow_contract(self):
        scores, features, mask, expert, recall = self.make_fixture()
        rel = classify_expert_vs_baseline(
            baseline_scores=scores,
            raw_features=features,
            inference_mask=mask,
            expert_local=expert,
            expert_in_shortlist=recall,
        )
        w = relation_row_weights(
            rel,
            agreement_weight=0.10,
            safe_weight=1.0,
            unsafe_weight=0.0,
            ambiguous_weight=0.20,
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            w,
            torch.tensor([0.10, 1.0, 0.0, 0.20]),
        )

    def test_unsafe_pair_loss_preserves_baseline(self):
        scores, features, mask, expert, recall = self.make_fixture()
        rel = classify_expert_vs_baseline(
            baseline_scores=scores,
            raw_features=features,
            inference_mask=mask,
            expert_local=expert,
            expert_in_shortlist=recall,
        )

        bad_final = scores.clone()
        bad_final[2, 0] = 0.0
        bad_final[2, 1] = 1.0
        loss, count = unsafe_expert_preservation_loss(
            final_scores=bad_final,
            expert_local=expert,
            relation=rel,
            margin=0.10,
        )
        self.assertEqual(count, 1)
        self.assertGreater(float(loss), 1.0)

        good_final = scores.clone()
        good_final[2, 0] = 1.0
        good_final[2, 1] = 0.0
        loss, count = unsafe_expert_preservation_loss(
            final_scores=good_final,
            expert_local=expert,
            relation=rel,
            margin=0.10,
        )
        self.assertEqual(count, 1)
        self.assertEqual(float(loss), 0.0)

    def test_change_diagnostics_classify_safe_and_unsafe(self):
        scores, features, mask, expert, recall = self.make_fixture()
        baseline = scores.argmax(dim=1)
        true = torch.tensor([0, 1, 1, 1], dtype=torch.long)
        d = change_diagnostics(
            baseline_pred=baseline,
            true_pred=true,
            expert_local=expert,
            expert_in_shortlist=recall,
            raw_features=features,
            inference_mask=mask,
        )
        self.assertEqual(d["changed_rows"], 3)
        self.assertEqual(d["safe_change_rows"], 1)
        self.assertEqual(d["unsafe_change_rows"], 1)
        self.assertEqual(d["ambiguous_change_rows"], 1)
        self.assertEqual(d["changed_to_expert_rows"], 3)


if __name__ == "__main__":
    unittest.main()
