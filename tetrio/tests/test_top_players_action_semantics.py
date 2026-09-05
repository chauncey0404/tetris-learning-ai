from __future__ import annotations

import unittest

from tetrio.datasets.top_players_action_semantics import (
    ActionMode,
    infer_action_from_previous_post_state,
)


class TopPlayersActionSemanticsTests(unittest.TestCase):
    def test_first_observed_transition_is_empty_hold(self):
        # Row 1 after placing I:
        # hold=N, next=JZSOT...
        # Row 2 locks Z, leaves J in hold and preview begins SOT...
        inf = infer_action_from_previous_post_state(
            prev_hold="N",
            prev_next="JZSOTLSLIOJZTJ",
            placed="Z",
            cur_hold="J",
            cur_next="SOTLSLIOJZTJTZ",
        )
        self.assertTrue(inf.uniquely_classified)
        self.assertEqual(inf.modes, (ActionMode.HOLD_EMPTY,))
        self.assertEqual(inf.active_piece, "J")
        self.assertEqual(inf.consumed_preview, 2)

    def test_second_observed_transition_is_no_hold(self):
        # Row 2 post-state: hold=J, next=SOT...
        # Row 3 locks S, keeps J in hold, preview begins OT...
        inf = infer_action_from_previous_post_state(
            prev_hold="J",
            prev_next="SOTLSLIOJZTJTZ",
            placed="S",
            cur_hold="J",
            cur_next="OTLSLIOJZTJTZS",
        )
        self.assertTrue(inf.uniquely_classified)
        self.assertEqual(inf.modes, (ActionMode.NO_HOLD,))
        self.assertEqual(inf.active_piece, "S")
        self.assertEqual(inf.consumed_preview, 1)

    def test_nonempty_hold_swap(self):
        inf = infer_action_from_previous_post_state(
            prev_hold="T",
            prev_next="IJSZOL",
            placed="T",
            cur_hold="I",
            cur_next="JSZOLTI",
        )
        self.assertTrue(inf.uniquely_classified)
        self.assertEqual(inf.modes, (ActionMode.HOLD_SWAP,))
        self.assertEqual(inf.active_piece, "I")
        self.assertEqual(inf.consumed_preview, 1)

    def test_queue_mismatch_is_unmatched(self):
        inf = infer_action_from_previous_post_state(
            prev_hold="N",
            prev_next="IJSZ",
            placed="I",
            cur_hold="N",
            cur_next="ZZZZ",
        )
        self.assertFalse(inf.modes)


if __name__ == "__main__":
    unittest.main()
