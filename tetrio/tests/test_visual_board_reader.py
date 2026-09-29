from __future__ import annotations

import unittest

from tetrio.vision.board import (
    EMPTY,
    MINO,
    classify_cell,
)


class VisualBoardReaderTests(unittest.TestCase):
    def test_default_colored_mino_signature(self):
        label, confidence = classify_cell(
            value_p90=205.0,
            chroma_p90=135.0,
            edge_density=0.18,
        )
        self.assertEqual(label, MINO)
        self.assertGreater(confidence, 0.80)

    def test_dark_board_signature_is_empty(self):
        label, confidence = classify_cell(
            value_p90=32.0,
            chroma_p90=22.0,
            edge_density=0.04,
        )
        self.assertEqual(label, EMPTY)
        self.assertGreater(confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
