from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tetris_ai.core.tetrominoes import padded_rotation_tensor


@unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX is not installed")
class JaxGeometryParityTests(unittest.TestCase):
    def test_vector_backend_uses_shared_piece_geometry(self):
        from singleplayer.backend.jax.vector_env import PIECE_MATRICES

        actual = np.asarray(PIECE_MATRICES)
        expected = padded_rotation_tensor(dtype=np.int8)
        self.assertEqual(actual.shape, (7, 4, 4, 4))
        self.assertTrue(np.array_equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
