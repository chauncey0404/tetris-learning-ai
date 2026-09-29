from __future__ import annotations

import unittest

from tetrio.tools.eval_expert_v0 import parse_args


class ExpertV0HeldoutEvalSmokeTests(unittest.TestCase):
    def test_module_imports(self):
        self.assertTrue(callable(parse_args))


if __name__ == "__main__":
    unittest.main()
