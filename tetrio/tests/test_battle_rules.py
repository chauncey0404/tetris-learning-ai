from __future__ import annotations

import unittest

from tetrio.attack import base_attack, calculate_attack
from tetrio.b2b import B2BState, advance_b2b, split_surge
from tetrio.battle_state import BattleState, resolve_battle_step
from tetrio.clear import ClearEvent
from tetrio.combo import advance_combo, multiplier_attack
from tetris_ai.core.spins.base import SpinKind


class TetrioBattleRulesTests(unittest.TestCase):
    def test_base_attack_table(self):
        self.assertEqual(base_attack(ClearEvent("I", 1)), 0)
        self.assertEqual(base_attack(ClearEvent("O", 2)), 1)
        self.assertEqual(base_attack(ClearEvent("L", 3)), 2)
        self.assertEqual(base_attack(ClearEvent("I", 4)), 4)
        self.assertEqual(base_attack(ClearEvent("T", 1, SpinKind.FULL)), 2)
        self.assertEqual(base_attack(ClearEvent("T", 2, SpinKind.FULL)), 4)
        self.assertEqual(base_attack(ClearEvent("T", 3, SpinKind.FULL)), 6)
        self.assertEqual(base_attack(ClearEvent("T", 1, SpinKind.MINI)), 0)
        self.assertEqual(base_attack(ClearEvent("T", 2, SpinKind.MINI)), 1)

    def test_multiplier_down_rounding(self):
        self.assertEqual(multiplier_attack(1, 1), 1)
        self.assertEqual(multiplier_attack(4, 1), 5)
        self.assertEqual(multiplier_attack(0, 1), 0)
        self.assertEqual(multiplier_attack(0, 2), 1)
        self.assertEqual(multiplier_attack(0, 6), 2)

    def test_combo_contract(self):
        c = -1
        c = advance_combo(c, 1); self.assertEqual(c, 0)
        c = advance_combo(c, 2); self.assertEqual(c, 1)
        c = advance_combo(c, 0); self.assertEqual(c, -1)

    def test_b2b_bonus_starts_on_second_difficult_clear(self):
        state = B2BState()
        t1 = advance_b2b(state, ClearEvent("I", 4))
        self.assertEqual(t1.after.count, 0)
        self.assertEqual(t1.bonus_for_current_attack, 0)
        t2 = advance_b2b(t1.after, ClearEvent("T", 2, SpinKind.FULL))
        self.assertEqual(t2.after.count, 1)
        self.assertEqual(t2.bonus_for_current_attack, 1)

    def test_surge_begins_at_displayed_x4_and_splits_three_ways(self):
        state = B2BState()
        for _ in range(5):
            state = advance_b2b(state, ClearEvent("I", 4)).after
        self.assertEqual(state.count, 4)
        self.assertEqual(state.surge, 4)
        br = advance_b2b(state, ClearEvent("O", 2))
        self.assertEqual(br.released_surge, (2, 1, 1))
        self.assertEqual(br.after, B2BState())
        self.assertEqual(split_surge(8), (3, 3, 2))

    def test_all_clear_is_flat_five_after_multiplier(self):
        event = ClearEvent("I", 4, all_clear=True)
        out = calculate_attack(event, combo=1, b2b_bonus=0)
        self.assertEqual(out.multiplied, 5)
        self.assertEqual(out.all_clear_bonus, 5)
        self.assertEqual(out.total, 10)

    def test_garbage_special_is_flat_one_not_multiplied(self):
        event = ClearEvent("I", 4, garbage_rows_cleared=1)
        out = calculate_attack(event, combo=4, b2b_bonus=0)
        self.assertEqual(out.multiplied, 8)
        self.assertEqual(out.garbage_special_bonus, 1)
        self.assertEqual(out.total, 9)

    def test_non_t_all_mini_has_zero_base_but_keeps_b2b_value(self):
        event = ClearEvent("J", 1, SpinKind.MINI)
        self.assertEqual(base_attack(event), 0)
        out = calculate_attack(event, combo=0, b2b_bonus=1)
        self.assertEqual(out.total, 0)
        tr = advance_b2b(B2BState(count=2, surge=0), event)
        self.assertEqual(tr.after.count, 3)

    def test_opener_phase_double_cancel(self):
        state = BattleState(pieces_placed=5, incoming_pending=6)
        result = resolve_battle_step(state, ClearEvent("T", 2, SpinKind.FULL))
        self.assertTrue(result.cancellation.opener_double_cancel)
        self.assertEqual(result.attack.total, 4)
        self.assertEqual(result.garbage_cancelled, 6)
        self.assertEqual(result.garbage_sent, 1)
        self.assertEqual(result.after.incoming_pending, 0)

    def test_after_opener_cancellation_is_one_for_one(self):
        state = BattleState(pieces_placed=14, incoming_pending=6)
        result = resolve_battle_step(state, ClearEvent("T", 2, SpinKind.FULL))
        self.assertFalse(result.cancellation.opener_double_cancel)
        self.assertEqual(result.garbage_cancelled, 4)
        self.assertEqual(result.garbage_sent, 0)
        self.assertEqual(result.after.incoming_pending, 2)


if __name__ == "__main__":
    unittest.main()
