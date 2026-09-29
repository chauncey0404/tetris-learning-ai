from __future__ import annotations

import unittest

from tetrio.tools.audit_expert_battle_state import classify_timing


class BattleStateAuditContractTests(unittest.TestCase):
    def test_post_action_timing(self):
        self.assertEqual(
            classify_timing(
                before_score=0.20,
                after_score=0.9995,
                threshold=0.999,
                margin=0.10,
            ),
            "post_action",
        )

    def test_pre_action_timing(self):
        self.assertEqual(
            classify_timing(
                before_score=0.9995,
                after_score=0.20,
                threshold=0.999,
                margin=0.10,
            ),
            "pre_action",
        )

    def test_ambiguous_timing_fails_closed(self):
        self.assertEqual(
            classify_timing(
                before_score=0.9995,
                after_score=0.9994,
                threshold=0.999,
                margin=0.10,
            ),
            "unresolved",
        )


if __name__ == "__main__":
    unittest.main()
