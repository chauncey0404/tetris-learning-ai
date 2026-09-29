from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from tetrio.tools import harvest_top_player_replays as h


def valid_replay_bytes():
    obj = {
        "replay": {
            "rounds": [
                [
                    {"replay": {"frames": 10, "events": []}},
                    {"replay": {"frames": 10, "events": []}},
                ]
            ]
        }
    }
    return json.dumps(obj).encode("utf-8")


class ValidBodySalvageTests(unittest.TestCase):
    def test_curl_exit_18_valid_body_is_salvaged(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "x.ttrm"

            def fake_run(cmd, **kwargs):
                out = Path(cmd[cmd.index("--output") + 1])
                out.write_bytes(valid_replay_bytes())
                return Mock(
                    returncode=18,
                    stderr="curl: (18) transfer closed with outstanding read data",
                    stdout="",
                )

            with patch.object(h.shutil, "which", return_value="curl.exe"), \
                 patch.object(h.subprocess, "run", side_effect=fake_run):
                result = h._download_with_curl(
                    "https://example.invalid/x",
                    dest,
                    timeout=10,
                    retries=2,
                )

            self.assertTrue(result[0])
            self.assertEqual(result[2], "VALID_BODY_DESPITE_TRANSPORT_ERROR")

    def test_invalid_truncated_body_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "x.ttrm"

            def fake_run(cmd, **kwargs):
                out = Path(cmd[cmd.index("--output") + 1])
                out.write_bytes(b'{"replay":{"rounds":[')
                return Mock(returncode=18, stderr="curl: (18) transfer closed", stdout="")

            with patch.object(h.shutil, "which", return_value="curl.exe"), \
                 patch.object(h.subprocess, "run", side_effect=fake_run):
                result = h._download_with_curl(
                    "https://example.invalid/x",
                    dest,
                    timeout=10,
                    retries=2,
                )

            self.assertFalse(result[0])
            self.assertFalse(dest.with_suffix(".ttrm.part").exists())


if __name__ == "__main__":
    unittest.main()
