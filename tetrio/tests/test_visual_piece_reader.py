from __future__ import annotations
import unittest
import numpy as np
from tetrio.vision.piece_reader import classify_piece_matrix

class PieceGeometryTests(unittest.TestCase):
    def test_shapes(self):
        cases = {
            "I": [[1,1,1,1]],
            "O": [[1,1],[1,1]],
            "T": [[0,1,0],[1,1,1]],
            "S": [[0,1,1],[1,1,0]],
            "Z": [[1,1,0],[0,1,1]],
            "J": [[1,0,0],[1,1,1]],
            "L": [[0,0,1],[1,1,1]],
        }
        for expected, matrix in cases.items():
            piece, score = classify_piece_matrix(
                np.asarray(matrix, dtype=np.uint8)
            )
            self.assertEqual(piece, expected)
            self.assertEqual(score, 1.0)

    def test_rotated_t(self):
        piece, score = classify_piece_matrix(
            np.asarray([[1,0],[1,1],[1,0]], dtype=np.uint8)
        )
        self.assertEqual(piece, "T")
        self.assertEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
