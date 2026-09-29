from __future__ import annotations

import unittest

import torch

from tetrio.future.dominance import safe_dominance_mask
from tetrio.future.features import FEATURE_SIZE, feature_index


class ExpertV11DominanceTests(unittest.TestCase):
    def test_clean_candidate_dominates_hole_candidate(self):
        f = torch.zeros((1, 2, FEATURE_SIZE))
        f[0, 0, feature_index("holes_after")] = 0
        f[0, 1, feature_index("holes_after")] = 2
        f[0, 0, feature_index("max_height_after")] = 4
        f[0, 1, feature_index("max_height_after")] = 5
        f[0, 0, feature_index("next_min_holes")] = 0
        f[0, 1, feature_index("next_min_holes")] = 1
        f[0, 0, feature_index("next_min_height")] = 4
        f[0, 1, feature_index("next_min_height")] = 5

        mask = torch.tensor([[True, True]])
        dom = safe_dominance_mask(f, mask)
        self.assertTrue(bool(dom[0, 0, 1]))
        self.assertFalse(bool(dom[0, 1, 0]))

    def test_tactical_candidate_is_not_dominated_by_non_tactical_equal_structure(self):
        f = torch.zeros((1, 2, FEATURE_SIZE))
        # candidate 1 has a Full T-spin; candidate 0 must not dominate it merely
        # because everything structural is equal.
        f[0, 1, feature_index("current_tspin_full")] = 1
        mask = torch.tensor([[True, True]])
        dom = safe_dominance_mask(f, mask)
        self.assertFalse(bool(dom[0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
