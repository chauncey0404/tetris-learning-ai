from __future__ import annotations

import unittest
from pathlib import Path


class ExpertV11RecoveryWeightHotfixTests(unittest.TestCase):
    def test_trainer_applies_recovery_weight(self):
        path = Path("tetrio/tools/train_expert_v1_1.py")
        text = path.read_text(encoding="utf-8")
        self.assertIn("float(recovery_weight) * pair", text)
        self.assertIn("recovery_weight=args.recovery_weight", text)


if __name__ == "__main__":
    unittest.main()
