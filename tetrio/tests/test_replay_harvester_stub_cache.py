from __future__ import annotations
import unittest
from tetrio.tools.harvest_top_player_replays import _record

class ReplayHarvesterStubTests(unittest.TestCase):
    def test_stub_record_is_parseable(self):
        row = {
            "replayid": "R:abc123",
            "stub": True,
            "ts": "2026-09-22T00:00:00Z",
            "user": {"id": "u1", "username": "alice"},
            "otherusers": [{"id": "u2", "username": "bob"}],
        }
        r = _record(row, 1, "u1")
        self.assertIsNotNone(r)
        self.assertEqual(r.replayid, "abc123")
        self.assertTrue(r.stub)
        self.assertEqual(r.owner_username, "alice")
        self.assertEqual(r.opponent_usernames, ["bob"])

    def test_wrapped_record_is_parseable(self):
        row = {
            "record": {
                "replayid": "xyz",
                "stub": False,
                "user": {"username": "alice"},
                "otherusers": [],
            }
        }
        r = _record(row, None, None)
        self.assertIsNotNone(r)
        self.assertEqual(r.replayid, "xyz")

if __name__ == "__main__":
    unittest.main()
