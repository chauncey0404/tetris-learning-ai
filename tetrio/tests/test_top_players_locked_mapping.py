from __future__ import annotations

import unittest
import numpy as np

from tetrio.datasets.top_players_s1 import (
    decode_playfield,
    resolve_dataset_mapping,
    simulate_mapped_dataset_placement,
)


class TopPlayersLockedMappingTests(unittest.TestCase):
    def test_i_orientation_specific_origins(self):
        expected = {
            "N": ("I",-1,38),
            "E": ("I",-2,38),
            "S": ("I",-2,37),
            "W": ("I",-1,37),
        }
        for r, value in expected.items():
            m = resolve_dataset_mapping("I", r)
            self.assertEqual((m.canonical_piece,m.x_offset,m.y_base), value)

    def test_o_only_n_is_validated(self):
        m = resolve_dataset_mapping("O","N")
        self.assertEqual((m.canonical_piece,m.x_offset,m.y_base), ("O",0,38))
        with self.assertRaises(ValueError):
            resolve_dataset_mapping("O","E")

    def test_dataset_j_l_are_swapped_relative_to_canonical_geometry(self):
        for r in "NESW":
            self.assertEqual(resolve_dataset_mapping("J",r).canonical_piece, "L")
            self.assertEqual(resolve_dataset_mapping("L",r).canonical_piece, "J")

    def test_t_s_z_use_common_origin(self):
        for p in ("T","S","Z"):
            for r in "NESW":
                m = resolve_dataset_mapping(p,r)
                self.assertEqual((m.canonical_piece,m.x_offset,m.y_base), (p,-1,38))

    def test_first_observed_i_transition(self):
        sim = simulate_mapped_dataset_placement(
            decode_playfield(""), piece="I", x=4, y=0, rotation_code="N"
        )
        self.assertTrue(sim.legal)
        self.assertEqual(sim.lines_cleared, 0)
        self.assertTrue(np.array_equal(sim.board_after_clear, decode_playfield("NNNIIII")))

    def test_second_observed_z_transition(self):
        sim = simulate_mapped_dataset_placement(
            decode_playfield("NNNIIII"), piece="Z", x=4, y=1, rotation_code="N"
        )
        expected = decode_playfield("NNNIIIINNNNNNNZZNNNNNNNZZ")
        self.assertTrue(sim.legal)
        self.assertEqual(sim.lines_cleared, 0)
        self.assertTrue(np.array_equal(sim.board_after_clear, expected))


if __name__ == "__main__":
    unittest.main()
