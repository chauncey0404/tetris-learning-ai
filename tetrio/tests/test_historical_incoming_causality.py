from __future__ import annotations
import unittest
from tetrio.tools.audit_historical_incoming_causality import rate, finite

class HistoricalIncomingAuditTests(unittest.TestCase):
    def test_rate(self):
        self.assertEqual(rate(1, 4), 0.25)
        self.assertIsNone(rate(1, 0))
    def test_finite(self):
        self.assertEqual(finite("2"), 2.0)
        self.assertIsNone(finite(None))

if __name__ == "__main__":
    unittest.main()
