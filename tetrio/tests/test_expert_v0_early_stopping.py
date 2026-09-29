from __future__ import annotations

import unittest

from tetris_ai.learning.early_stopping import EarlyStoppingTracker


class EarlyStoppingTrackerTests(unittest.TestCase):
    def test_max_mode_stops_after_patience_without_meaningful_gain(self):
        tracker = EarlyStoppingTracker(
            patience=3,
            min_delta=0.001,
            mode="max",
        )

        self.assertFalse(tracker.update(0.6000, 1))
        self.assertFalse(tracker.update(0.6005, 2))
        self.assertFalse(tracker.update(0.6008, 3))
        self.assertTrue(tracker.update(0.6007, 4))

        self.assertEqual(tracker.anchor_value, 0.6000)
        self.assertEqual(tracker.bad_epochs, 3)

    def test_meaningful_gain_resets_patience(self):
        tracker = EarlyStoppingTracker(
            patience=2,
            min_delta=0.001,
            mode="max",
        )

        self.assertFalse(tracker.update(0.6000, 1))
        self.assertFalse(tracker.update(0.6005, 2))
        self.assertFalse(tracker.update(0.6012, 3))
        self.assertEqual(tracker.bad_epochs, 0)
        self.assertEqual(tracker.anchor_value, 0.6012)

        self.assertFalse(tracker.update(0.6015, 4))
        self.assertTrue(tracker.update(0.6014, 5))

    def test_patience_zero_disables_stopping(self):
        tracker = EarlyStoppingTracker(
            patience=0,
            min_delta=0.001,
            mode="max",
        )
        for epoch in range(1, 20):
            self.assertFalse(tracker.update(0.5, epoch))

    def test_min_mode_is_supported(self):
        tracker = EarlyStoppingTracker(
            patience=2,
            min_delta=0.01,
            mode="min",
        )
        self.assertFalse(tracker.update(1.00, 1))
        self.assertFalse(tracker.update(0.995, 2))
        self.assertFalse(tracker.update(0.98, 3))
        self.assertFalse(tracker.update(0.979, 4))
        self.assertTrue(tracker.update(0.978, 5))


if __name__ == "__main__":
    unittest.main()
