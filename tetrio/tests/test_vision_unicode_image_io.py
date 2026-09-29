from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tetrio.tools.inspect_vision_layout import (
    imread_unicode,
    imwrite_unicode,
)


class UnicodeImageIoTests(unittest.TestCase):
    def test_unicode_path_roundtrip(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[4:20, 8:24] = (10, 120, 240)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "螢幕擷取畫面 測試.png"
            self.assertTrue(imwrite_unicode(path, image))
            loaded = imread_unicode(path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, image.shape)
        self.assertTrue(np.array_equal(loaded, image))


if __name__ == "__main__":
    unittest.main()
