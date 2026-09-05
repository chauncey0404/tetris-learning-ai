from __future__ import annotations

import unittest

from tetrio.datasets.top_players_action_semantics import (
    ActionMode,
    infer_action_from_previous_post_state,
)


class ExpertCorpusActionContractTests(unittest.TestCase):
    def test_first_observed_hold_empty_state(self):
        inf = infer_action_from_previous_post_state(
            prev_hold="N",
            prev_next="JZSOTLSLIOJZTJ",
            placed="Z",
            cur_hold="J",
            cur_next="SOTLSLIOJZTJTZ",
        )
        self.assertEqual(inf.modes, (ActionMode.HOLD_EMPTY,))

    def test_first_observed_no_hold_state(self):
        inf = infer_action_from_previous_post_state(
            prev_hold="J",
            prev_next="SOTLSLIOJZTJTZ",
            placed="S",
            cur_hold="J",
            cur_next="OTLSLIOJZTJTZS",
        )
        self.assertEqual(inf.modes, (ActionMode.NO_HOLD,))


if __name__ == "__main__":
    unittest.main()
