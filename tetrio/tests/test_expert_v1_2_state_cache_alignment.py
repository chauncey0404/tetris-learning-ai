from __future__ import annotations

import unittest
from pathlib import Path


class ExpertV12StateCacheAlignmentContractTests(unittest.TestCase):
    def test_adapter_uses_unordered_exact_lookup(self):
        text = Path(
            "tetrio/tools/build_expert_v1_2_state_cache.py"
        ).read_text(encoding="utf-8")
        self.assertIn("exact unordered key lookup", text)
        self.assertIn("target_positions", text)
        self.assertNotIn("ordered subsequence", text)
        self.assertIn("Refusing approximate alignment", text)


if __name__ == "__main__":
    unittest.main()
