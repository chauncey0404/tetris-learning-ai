from __future__ import annotations

import unittest

from tetrio.tools.audit_ttrm_garbage_accounting import (
    _eq,
    _inbound_packets,
)


class TtrmGarbageAccountingTests(unittest.TestCase):
    def test_inbound_packet_extractor_ignores_confirm_copy(self):
        player = {
            "replay": {
                "events": [
                    {
                        "type": "ige",
                        "frame": 10,
                        "data": {
                            "type": "interaction",
                            "data": {
                                "type": "garbage",
                                "amt": 4,
                                "iid": 1,
                            },
                        },
                    },
                    {
                        "type": "ige",
                        "frame": 11,
                        "data": {
                            "type": "interaction_confirm",
                            "data": {
                                "type": "garbage",
                                "amt": 4,
                                "iid": 1,
                            },
                        },
                    },
                ]
            }
        }
        packets = _inbound_packets(player)
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0]["amt"], 4.0)

    def test_numeric_equality(self):
        self.assertTrue(_eq(4.0, 4.0))
        self.assertFalse(_eq(4.0, 5.0))
        self.assertIsNone(_eq(None, 5.0))


if __name__ == "__main__":
    unittest.main()
