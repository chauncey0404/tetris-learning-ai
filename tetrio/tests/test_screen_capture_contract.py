from __future__ import annotations

import os
import unittest

from tetrio.vision.screen_capture import VirtualScreenGeometry


class ScreenCaptureContractTests(unittest.TestCase):
    def test_geometry_dict(self):
        g = VirtualScreenGeometry(x=-100, y=0, width=3840, height=1080)
        self.assertEqual(
            g.to_dict(),
            {"x": -100, "y": 0, "width": 3840, "height": 1080},
        )

    def test_import_is_safe_off_windows(self):
        # Importing the module must remain safe for CI/non-Windows analysis.
        # Actual capture is intentionally Windows-only.
        self.assertIn(os.name, {"nt", "posix", "java"})


if __name__ == "__main__":
    unittest.main()
