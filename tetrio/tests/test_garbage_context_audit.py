from __future__ import annotations

import unittest

from tetrio.tools.audit_garbage_context import (
    nearest_signed_lags,
)


class GarbageContextAuditTests(unittest.TestCase):
    def test_nearest_signed_lags_preserves_direction(self):
        source = [100, 200, 300]
        target = [120, 180, 340]
        self.assertEqual(
            nearest_signed_lags(source, target, 60),
            [20, -20, 40],
        )

    def test_window_filters_far_matches(self):
        self.assertEqual(
            nearest_signed_lags([100], [500], 60),
            [],
        )


if __name__ == "__main__":
    unittest.main()
