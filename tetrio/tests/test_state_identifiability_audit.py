from __future__ import annotations

import unittest

import numpy as np

from tetrio.tools.audit_state_identifiability import (
    deranged_permutation,
)


class StateIdentifiabilityAuditTests(unittest.TestCase):
    def test_derangement_has_no_fixed_points(self):
        for n in (2, 3, 10, 100):
            p = deranged_permutation(n, 12345 + n)
            self.assertEqual(sorted(p.tolist()), list(range(n)))
            self.assertFalse(np.any(p == np.arange(n)))

    def test_derangement_is_deterministic(self):
        a = deranged_permutation(100, 999)
        b = deranged_permutation(100, 999)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
