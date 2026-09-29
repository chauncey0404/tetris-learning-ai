from __future__ import annotations

import unittest

import numpy as np
import torch

from tetrio.future.features import (
    FEATURE_SIZE,
    MODEL_NEUTRAL_FEATURES,
    feature_index,
    normalize_future_features_numpy,
    normalize_future_features_torch,
)


class ExpertV11FastBulkTests(unittest.TestCase):
    def test_exact_only_columns_are_neutralized_numpy(self):
        x = np.ones((2, FEATURE_SIZE), dtype=np.float32)
        y = normalize_future_features_numpy(x)
        for name in MODEL_NEUTRAL_FEATURES:
            self.assertEqual(float(y[0, feature_index(name)]), 0.0)

    def test_exact_only_columns_are_neutralized_torch(self):
        x = torch.ones((2, FEATURE_SIZE), dtype=torch.float32)
        y = normalize_future_features_torch(x)
        for name in MODEL_NEUTRAL_FEATURES:
            self.assertEqual(float(y[0, feature_index(name)]), 0.0)


if __name__ == "__main__":
    unittest.main()
