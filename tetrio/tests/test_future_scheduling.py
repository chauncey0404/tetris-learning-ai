from __future__ import annotations

import unittest
from pathlib import Path


class FutureSchedulingContractTests(unittest.TestCase):
    def test_adaptive_scheduler_reassembles_in_original_order(self):
        text = Path("tetrio/rollout/batched.py").read_text(encoding="utf-8")
        self.assertIn("def _parallel_future_features(", text)
        self.assertIn("adaptive future scheduling reassembly mismatch", text)
        self.assertIn("np.concatenate(", text)

    def test_strict_neural_path_is_unchanged(self):
        text = Path("tetrio/rollout/batched.py").read_text(encoding="utf-8")
        self.assertIn("def _score_state_exact(", text)
        self.assertIn("score_from_state_latent(", text)
        self.assertNotIn("def _score_all_candidates(", text)


if __name__ == "__main__":
    unittest.main()
