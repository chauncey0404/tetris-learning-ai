from __future__ import annotations

import unittest

from tetrio.future.lookahead import TSpinTarget


class ExpertV11SpeedV2HotfixTests(unittest.TestCase):
    def test_tspin_target_symbol_is_available_in_lookahead_module(self):
        target = TSpinTarget()
        self.assertEqual(target.full, 0)
        self.assertEqual(target.mini, 0)


if __name__ == "__main__":
    unittest.main()
