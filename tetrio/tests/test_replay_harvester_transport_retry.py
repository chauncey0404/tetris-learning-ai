from __future__ import annotations
import http.client
import unittest
from unittest.mock import patch

from tetrio.tools import harvest_top_player_replays as h


class ReplayHarvesterTransportRetryTests(unittest.TestCase):
    def test_download_catches_exhausted_incomplete_read(self):
        record = h.ReplayRecord(
            replayid="abc",
            owner_username="alice",
            opponent_usernames=["bob"],
            ts=None,
            stub=False,
            source_rank=1,
            source_user_id=None,
        )
        with patch.object(
            h,
            "_http",
            side_effect=http.client.IncompleteRead(b"partial"),
        ):
            ok, reason, identity = h._download(
                record,
                __import__("pathlib").Path("never_written.ttrm"),
                timeout=1.0,
                retries=0,
            )
        self.assertFalse(ok)
        self.assertIn("IncompleteRead", reason)
        self.assertIsNone(identity)

    def test_incomplete_read_is_http_exception(self):
        self.assertTrue(
            issubclass(http.client.IncompleteRead, http.client.HTTPException)
        )


if __name__ == "__main__":
    unittest.main()
