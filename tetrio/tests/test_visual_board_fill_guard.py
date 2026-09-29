from __future__ import annotations

import unittest

from tetrio.vision.board import (
    MINO,
    TRANSIENT,
    classify_cell,
)


class VisualBoardFillGuardTests(unittest.TestCase):
    def test_real_mino_like_full_cell_is_mino(self):
        label, confidence = classify_cell(
            value_p90=210.0,
            chroma_p90=138.0,
            edge_density=0.18,
            colored_fill_ratio=0.98,
        )
        self.assertEqual(label, MINO)
        self.assertGreater(confidence, 0.80)

    def test_countdown_like_partial_colored_cell_is_transient(self):
        label, confidence = classify_cell(
            value_p90=128.0,
            chroma_p90=126.0,
            edge_density=0.14,
            colored_fill_ratio=0.33,
        )
        self.assertEqual(label, TRANSIENT)
        self.assertGreater(confidence, 0.70)


if __name__ == "__main__":
    unittest.main()
