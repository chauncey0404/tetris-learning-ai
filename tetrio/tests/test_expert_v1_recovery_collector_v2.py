from __future__ import annotations

import unittest

from tetrio.tools.collect_expert_v1_recovery_states import (
    RecoveryState,
    _fmt_eta,
    parse_seeds,
)


class ExpertV1RecoveryCollectorV2Tests(unittest.TestCase):
    def test_seed_parser(self):
        self.assertEqual(
            parse_seeds("9031-9033,9040"),
            [9031, 9032, 9033, 9040],
        )

    def test_eta_formatter(self):
        self.assertEqual(_fmt_eta(59), "59s")
        self.assertEqual(_fmt_eta(61), "1m01s")
        self.assertEqual(_fmt_eta(3660), "1h01m")

    def test_recovery_state_has_five_preview_pieces(self):
        s = RecoveryState(seed=9031, max_pieces=100)
        self.assertEqual(len(s.preview()), 5)

    def test_no_hold_branch_keeps_hold(self):
        s = RecoveryState(seed=9031, max_pieces=100)
        s.hold = "T"
        before = s.preview()
        plan = s.branch_plan(False)
        self.assertEqual(plan.selected_piece, s.active)
        self.assertEqual(plan.hold_after, "T")
        self.assertEqual(plan.next_active, before[0])
        self.assertEqual(plan.consume_count, 1)
        self.assertEqual(plan.mode, "no_hold")

    def test_hold_swap_branch_uses_hold_piece(self):
        s = RecoveryState(seed=9031, max_pieces=100)
        active = s.active
        s.hold = "T"
        plan = s.branch_plan(True)
        self.assertEqual(plan.selected_piece, "T")
        self.assertEqual(plan.hold_after, active)
        self.assertEqual(plan.consume_count, 1)
        self.assertEqual(plan.mode, "hold_swap")


if __name__ == "__main__":
    unittest.main()
