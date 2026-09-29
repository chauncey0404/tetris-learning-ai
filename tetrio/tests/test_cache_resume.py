from __future__ import annotations

import unittest
from pathlib import Path


class CacheResumeContractTests(unittest.TestCase):
    def test_counterfactual_builder_has_cross_directory_reuse(self):
        text = Path(
            "tetrio/tools/build_expert_v1_counterfactual_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--reuse-dir", text)
        self.assertIn("_existing_sidecar_stats", text)
        self.assertIn("expected_rows=take", text)

    def test_future_builder_has_cross_directory_reuse(self):
        text = Path(
            "tetrio/tools/build_expert_v1_1_future_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--reuse-dir", text)
        self.assertIn("_try_reuse_future_shard", text)
        self.assertIn("expected_game_id", text)


if __name__ == "__main__":
    unittest.main()
