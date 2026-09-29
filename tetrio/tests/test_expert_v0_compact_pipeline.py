from __future__ import annotations

import unittest

import numpy as np
import torch

from tetrio.network.cache import (
    batches_from_shard,
    compact_batches_from_shard,
)
from tetrio.network.encoding import (
    dense_candidate_batch,
    dense_state_batch,
    torch_dense_candidate_batch,
    torch_dense_state_batch,
)


class ExpertV0CompactPipelineTests(unittest.TestCase):
    def test_torch_state_encoding_matches_numpy(self):
        rng = np.random.default_rng(7)
        board = rng.integers(0, 256, size=(4, 50), dtype=np.uint8)
        active = np.asarray([0, 6, 7, 2], dtype=np.uint8)
        hold = np.asarray([7, 1, 4, 0], dtype=np.uint8)
        preview = rng.integers(0, 7, size=(4, 5), dtype=np.uint8)

        expected = dense_state_batch(board, active, hold, preview)
        actual = torch_dense_state_batch(
            torch.from_numpy(board),
            torch.from_numpy(active),
            torch.from_numpy(hold),
            torch.from_numpy(preview),
        ).numpy()

        np.testing.assert_array_equal(actual, expected)

    def test_torch_candidate_encoding_matches_numpy(self):
        rng = np.random.default_rng(11)
        n = 31
        board = rng.integers(0, 256, size=(n, 50), dtype=np.uint8)
        piece = rng.integers(0, 7, size=n, dtype=np.uint8)
        rotation = rng.integers(0, 4, size=n, dtype=np.uint8)
        x = rng.integers(-2, 10, size=n, dtype=np.int8)
        y = rng.integers(-2, 40, size=n, dtype=np.int8)
        hold = rng.integers(0, 2, size=n, dtype=np.uint8)
        lines = rng.integers(0, 5, size=n, dtype=np.uint8)

        expected = dense_candidate_batch(
            board, piece, rotation, x, y, hold, lines
        )
        actual = torch_dense_candidate_batch(
            torch.from_numpy(board),
            torch.from_numpy(piece),
            torch.from_numpy(rotation),
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(hold),
            torch.from_numpy(lines),
        ).numpy()

        np.testing.assert_array_equal(actual, expected)

    def test_compact_and_dense_batch_preserve_candidate_order(self):
        rng = np.random.default_rng(19)
        counts = np.asarray([2, 4, 3], dtype=np.int32)
        offsets = np.asarray([0, 2, 6, 9], dtype=np.int32)
        total = int(counts.sum())

        data = {
            "state_board_packed": rng.integers(0, 256, size=(3, 50), dtype=np.uint8),
            "state_active": np.asarray([0, 1, 2], dtype=np.uint8),
            "state_hold": np.asarray([7, 2, 3], dtype=np.uint8),
            "state_preview": rng.integers(0, 7, size=(3, 5), dtype=np.uint8),
            "candidate_offsets": offsets,
            "candidate_board_packed": rng.integers(0, 256, size=(total, 50), dtype=np.uint8),
            "candidate_piece": rng.integers(0, 7, size=total, dtype=np.uint8),
            "candidate_rotation": rng.integers(0, 4, size=total, dtype=np.uint8),
            "candidate_x": rng.integers(-2, 10, size=total, dtype=np.int8),
            "candidate_y": rng.integers(-2, 40, size=total, dtype=np.int8),
            "candidate_use_hold": rng.integers(0, 2, size=total, dtype=np.uint8),
            "candidate_lines": rng.integers(0, 5, size=total, dtype=np.uint8),
            "expert_index": np.asarray([1, 2, 0], dtype=np.int16),
            "use_hold": np.asarray([0, 1, 0], dtype=np.uint8),
        }

        dense = next(batches_from_shard(data, batch_size=3, rng=None))
        compact = next(compact_batches_from_shard(data, batch_size=3, rng=None))

        flat = torch_dense_candidate_batch(
            torch.from_numpy(compact.candidate_board_packed),
            torch.from_numpy(compact.candidate_piece),
            torch.from_numpy(compact.candidate_rotation),
            torch.from_numpy(compact.candidate_x),
            torch.from_numpy(compact.candidate_y),
            torch.from_numpy(compact.candidate_use_hold),
            torch.from_numpy(compact.candidate_lines),
        ).numpy()

        for n in range(total):
            row = int(compact.candidate_owner[n])
            col = int(compact.candidate_local[n])
            np.testing.assert_array_equal(flat[n], dense.candidates[row, col])


if __name__ == "__main__":
    unittest.main()
