from __future__ import annotations

import unittest

from tetrio.datasets.battle_state import (
    CausalBattleHistory,
    PlacementOutcome,
    advance_causal_history,
    is_difficult_clear,
)


class ExpertBattleStateTests(unittest.TestCase):
    def test_combo_history_is_pre_action_and_causal(self):
        s = CausalBattleHistory()
        self.assertEqual(s.combo_chain, 0)
        self.assertEqual(s.combo_index, -1)

        s = advance_causal_history(s, PlacementOutcome(cleared=1))
        self.assertEqual(s.combo_chain, 1)
        self.assertEqual(s.combo_index, 0)

        s = advance_causal_history(s, PlacementOutcome(cleared=2))
        self.assertEqual(s.combo_chain, 2)
        self.assertEqual(s.combo_index, 1)

        s = advance_causal_history(s, PlacementOutcome(cleared=0))
        self.assertEqual(s.combo_chain, 0)
        self.assertEqual(s.combo_index, -1)

    def test_difficult_chain_preserves_through_no_clear(self):
        s = CausalBattleHistory()
        s = advance_causal_history(s, PlacementOutcome(cleared=4))
        self.assertTrue(s.difficult_active)
        self.assertEqual(s.difficult_chain, 1)

        s = advance_causal_history(s, PlacementOutcome(cleared=0))
        self.assertTrue(s.difficult_active)
        self.assertEqual(s.difficult_chain, 1)

        s = advance_causal_history(s, PlacementOutcome(cleared=1))
        self.assertFalse(s.difficult_active)
        self.assertEqual(s.difficult_chain, 0)

    def test_spin_clear_is_difficult_but_spin_no_clear_is_not_increment(self):
        self.assertTrue(is_difficult_clear(lines_cleared=2, t_spin="T_SPIN"))
        self.assertFalse(is_difficult_clear(lines_cleared=0, t_spin="T_SPIN"))

    def test_previous_outcome_is_lagged_history(self):
        s = CausalBattleHistory()
        o = PlacementOutcome(
            cleared=2,
            t_spin="T_SPIN",
            attack=4.0,
            garbage_cleared=1,
        )
        s = advance_causal_history(s, o)
        self.assertEqual(s.previous.cleared, 2)
        self.assertEqual(s.previous.t_spin, "T_SPIN")
        self.assertEqual(s.previous.attack, 4.0)
        self.assertEqual(s.previous.garbage_cleared, 1)


if __name__ == "__main__":
    unittest.main()
