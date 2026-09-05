from __future__ import annotations

import unittest

import numpy as np
import torch

from singleplayer.game.placement import enumerate_placements, get_rotations
from singleplayer.network.q_network import CANDIDATE_SIZE, STATE_SIZE, ObservableSafeQNetwork
from tetris_ai.core.tetrominoes import PIECE_NAMES, native_matrix, padded_rotation_tensor, trimmed_matrix


class ArchitectureParityTests(unittest.TestCase):
    def test_singleplayer_empty_board_i_candidate_count_is_17(self):
        board = np.zeros((20, 10), dtype=np.uint8)
        self.assertEqual(len(enumerate_placements(board, "I")), 17)

    def test_rotation_counts_match_legacy_contract(self):
        expected = {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
        self.assertEqual({p: len(get_rotations(p)) for p in PIECE_NAMES}, expected)

    def test_shared_native_geometry_produces_legacy_spawn_boxes(self):
        expected = {
            "I": np.asarray([[1,1,1,1]], dtype=np.uint8),
            "O": np.asarray([[1,1],[1,1]], dtype=np.uint8),
            "T": np.asarray([[0,1,0],[1,1,1]], dtype=np.uint8),
            "S": np.asarray([[0,1,1],[1,1,0]], dtype=np.uint8),
            "Z": np.asarray([[1,1,0],[0,1,1]], dtype=np.uint8),
            "J": np.asarray([[1,0,0],[1,1,1]], dtype=np.uint8),
            "L": np.asarray([[0,0,1],[1,1,1]], dtype=np.uint8),
        }
        for piece, shape in expected.items():
            self.assertTrue(np.array_equal(trimmed_matrix(piece, 0), shape), piece)

    def test_padded_rotation_tensor_matches_native_matrices(self):
        tensor = padded_rotation_tensor()
        self.assertEqual(tensor.shape, (7, 4, 4, 4))
        for p, piece in enumerate(PIECE_NAMES):
            for r in range(4):
                m = native_matrix(piece, r)
                self.assertTrue(np.array_equal(tensor[p, r, :m.shape[0], :m.shape[1]], m))

    def test_v8_q_network_state_dict_contract(self):
        model = ObservableSafeQNetwork()
        expected_keys = {
            "state_encoder.0.weight", "state_encoder.0.bias",
            "state_encoder.1.weight", "state_encoder.1.bias",
            "state_encoder.3.weight", "state_encoder.3.bias",
            "candidate_encoder.0.weight", "candidate_encoder.0.bias",
            "candidate_encoder.1.weight", "candidate_encoder.1.bias",
            "candidate_encoder.3.weight", "candidate_encoder.3.bias",
            "joint.0.weight", "joint.0.bias", "joint.2.weight", "joint.2.bias",
            "q_head.weight", "q_head.bias",
        }
        self.assertEqual(set(model.state_dict()), expected_keys)
        state = torch.zeros((2, STATE_SIZE))
        candidates = torch.zeros((2, 4, CANDIDATE_SIZE))
        scalars = torch.zeros((2, 4))
        q = model(state=state, candidates=candidates, rewards=scalars, teacher_scores=scalars, teacher_ranks=scalars)
        self.assertEqual(tuple(q.shape), (2, 4))
        self.assertTrue(torch.equal(q, torch.zeros_like(q)))


if __name__ == "__main__":
    unittest.main()
