from __future__ import annotations

import unittest
from pathlib import Path

from tetrio.future.lookahead import FutureFeatureConfig
from tetrio.rollout.batched import BatchedRolloutConfig


class FutureOptimizationBenchmarkContractTests(unittest.TestCase):
    def test_execution_toggles_exist(self):
        f = FutureFeatureConfig()
        self.assertTrue(f.use_search_cache)
        self.assertTrue(f.use_feature_memo)

        r = BatchedRolloutConfig()
        self.assertTrue(r.adaptive_future_scheduling)
        self.assertTrue(r.future_search_cache)
        self.assertTrue(r.future_feature_memo)
        self.assertGreaterEqual(r.future_max_chunks_per_row, 1)

    def test_modes_present(self):
        text = Path("tetrio/tools/benchmark_future_search.py").read_text(
            encoding="utf-8"
        )
        for name in ("baseline", "cache_memo", "chunk2", "chunk4"):
            self.assertIn(f'"{name}"', text)
        self.assertIn("POLICY DIVERGENCE", text)


if __name__ == "__main__":
    unittest.main()
