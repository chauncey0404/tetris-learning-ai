from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from tetrio.tools import harvest_top_player_replays as h


class CurlTransportHotfixTests(unittest.TestCase):
    def test_curl_missing_requests_fallback(self):
        with patch.object(h.shutil, "which", return_value=None):
            ok, reason = h._download_with_curl(
                "https://example.invalid/x",
                Path("x.ttrm"),
                timeout=10,
                retries=1,
            )
        self.assertIsNone(ok)
        self.assertEqual(reason, "curl not found")

    def test_curl_failure_is_reported_and_partial_removed(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "x.ttrm"
            with patch.object(h.shutil, "which", return_value="curl.exe"), \
                 patch.object(
                     h.subprocess,
                     "run",
                     return_value=Mock(returncode=18, stderr="transfer closed", stdout=""),
                 ):
                ok, reason = h._download_with_curl(
                    "https://example.invalid/x",
                    dest,
                    timeout=10,
                    retries=2,
                )
            self.assertFalse(ok)
            self.assertIn("curl exit 18", reason)
            self.assertFalse(dest.with_suffix(".ttrm.part").exists())


if __name__ == "__main__":
    unittest.main()
