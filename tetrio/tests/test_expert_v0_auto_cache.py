from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tetrio.network.cache_prepare import inspect_expert_v0_cache
from tetrio.reachability import TETRIO_ENTRY_RAISE_ROWS


def contract() -> dict:
    return {
        "game": "tetrio",
        "entry_raise_rows_vs_generic": int(TETRIO_ENTRY_RAISE_ROWS),
        "rotation_system": "TETR.IO SRS+/180 production ruleset",
        "landing_identity": "(piece,x,y,rotation)",
        "exception_policy": "reference_confirmed_expert_unreachable_v1",
    }


class ExpertV0AutoCacheTests(unittest.TestCase):
    def _write(self, root: Path, status: str, built: int, excluded: int, failed: int) -> None:
        (root / "shard_00000.npz").write_bytes(b"placeholder")
        (root / "manifest.json").write_text(
            json.dumps({
                "status": status,
                "selected_rows": built + excluded + failed,
                "built_rows": built,
                "excluded_rows": excluded,
                "hard_failed_rows": failed,
                "failed_rows": failed,
                "backend": "fast",
                "shard_size": 8192,
                "reachability_contract": contract(),
            }),
            encoding="utf-8",
        )

    def test_missing_cache_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = inspect_expert_v0_cache(Path(tmp) / "missing")
            self.assertFalse(status.reusable)
            self.assertEqual(status.reason, "missing")

    def test_pass_manifest_and_shard_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "PASS", 123, 0, 0)
            self.assertTrue(inspect_expert_v0_cache(root).reusable)

    def test_pass_with_exclusions_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "PASS_WITH_EXCLUSIONS", 121, 2, 0)
            self.assertTrue(inspect_expert_v0_cache(root).reusable)

    def test_partial_cache_is_rebuilt_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shard_00000.npz").write_bytes(b"placeholder")
            status = inspect_expert_v0_cache(root)
            self.assertFalse(status.reusable)
            self.assertEqual(status.reason, "manifest_missing")

    def test_failed_manifest_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "FAIL", 122, 0, 1)
            self.assertFalse(inspect_expert_v0_cache(root).reusable)


if __name__ == "__main__":
    unittest.main()
