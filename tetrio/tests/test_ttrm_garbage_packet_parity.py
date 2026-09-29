from __future__ import annotations

import unittest

from tetrio.tools.audit_ttrm_garbage_packets import (
    _packet_key,
    _payload,
)


class TtrmGarbagePacketAuditTests(unittest.TestCase):
    def test_extracts_interaction_garbage(self):
        event = {
            "type": "ige",
            "frame": 100,
            "data": {
                "frame": 90,
                "type": "interaction",
                "data": {
                    "type": "garbage",
                    "amt": 4,
                    "gameid": 22,
                    "frame": 80,
                    "cid": 1,
                    "iid": 1,
                    "ackiid": 0,
                    "x": 8,
                    "y": 37,
                    "size": 1,
                },
            },
        }
        typ, inner = _payload(event)
        self.assertEqual(typ, "interaction")
        self.assertEqual(inner["amt"], 4)

    def test_packet_key_matches_confirm_copy(self):
        a = {
            "cid": 1, "iid": 2, "ackiid": 0, "gameid": 9,
            "amt": 5, "frame": 100, "x": 8, "y": 37, "size": 1,
        }
        b = dict(a)
        self.assertEqual(_packet_key(a), _packet_key(b))


if __name__ == "__main__":
    unittest.main()
