from __future__ import annotations

import unittest

from tetrio.parity.replay_inventory import inspect_ttrm_object
from tetrio.tools.extract_ttrm_garbage_events import _all_type_tags


class CurrentTtrmSchemaTests(unittest.TestCase):
    def test_current_replay_rounds_are_discovered(self):
        obj = {
            "replay": {
                "rounds": [
                    [
                        {"replay": {"frames": 100, "events": []}},
                        {"replay": {"frames": 120, "events": []}},
                    ],
                    [
                        {"replay": {"frames": 90, "events": []}},
                        {"replay": {"frames": 95, "events": []}},
                    ],
                ]
            }
        }
        inv = inspect_ttrm_object(obj)
        self.assertEqual(inv.round_count, 2)
        self.assertEqual(len(inv.player_replays), 4)
        self.assertEqual(inv.player_replays[0].frames, 100)
        self.assertEqual(inv.player_replays[3].frames, 95)

    def test_legacy_schema_still_works(self):
        obj = {
            "data": [
                {
                    "replays": [
                        {"frames": 10, "events": []},
                        {"frames": 20, "events": []},
                    ]
                }
            ]
        }
        inv = inspect_ttrm_object(obj)
        self.assertEqual(inv.round_count, 1)
        self.assertEqual(len(inv.player_replays), 2)

    def test_recursive_type_tags(self):
        event = {
            "type": "ige",
            "data": {
                "type": "interaction",
                "data": {
                    "type": "garbage",
                },
            },
        }
        self.assertEqual(
            _all_type_tags(event),
            ("ige", "interaction", "garbage"),
        )


if __name__ == "__main__":
    unittest.main()
