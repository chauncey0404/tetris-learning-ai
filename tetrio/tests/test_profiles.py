from __future__ import annotations

import unittest

from tetrio import TETRIO_ALL_MINI_PLUS, TETRIO_MULTIPLAYER, TETRIO_TETRA_LEAGUE


class TetrioProfileTests(unittest.TestCase):
    def test_ranked_default_is_tetra_league_all_mini(self):
        self.assertIs(TETRIO_MULTIPLAYER, TETRIO_TETRA_LEAGUE)
        self.assertEqual(TETRIO_TETRA_LEAGUE.spins.name, "TETR.IO All-Mini")
        self.assertTrue(TETRIO_TETRA_LEAGUE.movement.allow_180)

    def test_all_mini_plus_remains_separate_profile(self):
        self.assertEqual(TETRIO_ALL_MINI_PLUS.spins.name, "TETR.IO All-Mini+")


if __name__ == "__main__":
    unittest.main()
