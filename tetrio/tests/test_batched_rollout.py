from __future__ import annotations

import unittest

from tetrio.rollout.batched import (
    BatchedRolloutConfig,
    RolloutState,
)


class BatchedRolloutTests(unittest.TestCase):
    def test_rollout_state_has_deterministic_preview(self):
        a = RolloutState(seed=9051, max_pieces=100)
        b = RolloutState(seed=9051, max_pieces=100)
        self.assertEqual(a.active, b.active)
        self.assertEqual(a.preview(), b.preview())

    def test_no_hold_plan_matches_queue_semantics(self):
        s = RolloutState(seed=9051, max_pieces=100)
        before = s.preview()
        plan = s.plan_branch(False)
        self.assertEqual(plan.selected_piece, s.active)
        self.assertEqual(plan.next_active, before[0])
        self.assertEqual(plan.consume_count, 1)
        self.assertEqual(plan.mode, "no_hold")

    def test_hold_empty_plan_matches_queue_semantics(self):
        s = RolloutState(seed=9051, max_pieces=100)
        active = s.active
        before = s.preview()
        plan = s.plan_branch(True)
        self.assertEqual(plan.selected_piece, before[0])
        self.assertEqual(plan.hold_after, active)
        self.assertEqual(plan.next_active, before[1])
        self.assertEqual(plan.consume_count, 2)
        self.assertEqual(plan.mode, "hold_empty")

    def test_config_defaults_are_parallel(self):
        c = BatchedRolloutConfig()
        self.assertGreaterEqual(c.workers, 1)
        self.assertGreaterEqual(c.state_batch, 1)


    def test_strict_path_preserves_legacy_neural_shapes(self):
        from pathlib import Path

        source = Path("tetrio/rollout/batched.py").read_text(encoding="utf-8")
        self.assertIn("def _score_state_exact(", source)
        self.assertIn("score_from_state_latent(", source)
        self.assertNotIn("def _score_all_candidates(", source)



if __name__ == "__main__":
    unittest.main()
