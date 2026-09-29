from __future__ import annotations

import unittest

import numpy as np

from tetrio.future.structural_features import (
    board_structure,
    board_structure_batch,
)


class FutureStructureVectorizationTests(unittest.TestCase):
    def test_batch_matches_scalar_exactly(self):
        rng = np.random.default_rng(20260918)
        boards = (rng.random((64, 40, 10)) < 0.22).astype(np.uint8)

        batch = board_structure_batch(boards)
        scalar = [board_structure(board) for board in boards]

        np.testing.assert_array_equal(
            batch.holes,
            np.asarray([x.holes for x in scalar]),
        )
        np.testing.assert_array_equal(
            batch.max_height,
            np.asarray([x.max_height for x in scalar]),
        )
        np.testing.assert_array_equal(
            batch.aggregate_height,
            np.asarray([x.aggregate_height for x in scalar]),
        )
        np.testing.assert_array_equal(
            batch.bumpiness,
            np.asarray([x.bumpiness for x in scalar]),
        )
        np.testing.assert_array_equal(
            batch.max_well,
            np.asarray([x.max_well for x in scalar]),
        )

    def test_empty_board_matches(self):
        boards = np.zeros((3, 40, 10), dtype=np.uint8)
        batch = board_structure_batch(boards)
        self.assertTrue(np.all(batch.holes == 0))
        self.assertTrue(np.all(batch.max_height == 0))
        self.assertTrue(np.all(batch.bumpiness == 0))


if __name__ == "__main__":
    unittest.main()
