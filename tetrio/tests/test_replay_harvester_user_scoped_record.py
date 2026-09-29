from __future__ import annotations

import unittest

from tetrio.tools.harvest_top_player_replays import _record


class UserScopedRecordHotfixTests(unittest.TestCase):
    def test_current_user_scoped_record_without_user_field(self):
        row = {
            "_id": "record-id",
            "replayid": "R:replay123",
            "stub": True,
            "ts": "2026-09-22T00:00:00Z",
            "otherusers": [
                {"id": "opponent-id", "username": "opponent"}
            ],
            "gamemode": "league",
            "results": {},
        }
        r = _record(
            row,
            rank=1,
            user_id="owner-id",
            owner_username="5han",
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.replayid, "replay123")
        self.assertEqual(r.owner_username, "5han")
        self.assertEqual(r.opponent_usernames, ["opponent"])
        self.assertTrue(r.stub)

    def test_documented_record_user_still_wins(self):
        row = {
            "replayid": "abc",
            "stub": False,
            "user": {"username": "documented-owner"},
            "otherusers": [],
        }
        r = _record(
            row,
            rank=None,
            user_id=None,
            owner_username="fallback-owner",
        )
        self.assertEqual(r.owner_username, "documented-owner")

    def test_missing_owner_fails_closed(self):
        row = {
            "replayid": "abc",
            "stub": False,
            "otherusers": [],
        }
        self.assertIsNone(
            _record(row, rank=None, user_id=None, owner_username=None)
        )


if __name__ == "__main__":
    unittest.main()
