from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tetrio.network.cache_prepare import inspect_expert_v0_cache
from tetrio.reachability import TETRIO_ENTRY_RAISE_ROWS


def current_contract() -> dict:
    return {
        "game": "tetrio",
        "entry_raise_rows_vs_generic": int(TETRIO_ENTRY_RAISE_ROWS),
        "rotation_system": "TETR.IO SRS+/180 production ruleset",
        "landing_identity": "(piece,x,y,rotation)",
        "exception_policy": "reference_confirmed_expert_unreachable_v1",
    }


class ExpertV0CacheContractTests(unittest.TestCase):
    def _write_cache(
        self,
        root: Path,
        *,
        entry_raise: int = 1,
        status: str = "PASS",
        built_rows: int = 123,
        excluded_rows: int = 0,
        hard_failed_rows: int = 0,
        include_policy: bool = True,
    ) -> None:
        (root / "shard_00000.npz").write_bytes(b"placeholder")
        contract = current_contract()
        contract["entry_raise_rows_vs_generic"] = entry_raise
        if not include_policy:
            contract.pop("exception_policy", None)
        (root / "manifest.json").write_text(
            json.dumps({
                "status": status,
                "selected_rows": built_rows + excluded_rows + hard_failed_rows,
                "built_rows": built_rows,
                "excluded_rows": excluded_rows,
                "hard_failed_rows": hard_failed_rows,
                "failed_rows": hard_failed_rows,
                "backend": "fast",
                "shard_size": 8192,
                "reachability_contract": contract,
            }),
            encoding="utf-8",
        )

    def test_current_raise1_contract_is_reusable(self):
        self.assertEqual(TETRIO_ENTRY_RAISE_ROWS, 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cache(root)
            self.assertTrue(inspect_expert_v0_cache(root).reusable)

    def test_pass_with_reference_confirmed_exclusions_is_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cache(root, status="PASS_WITH_EXCLUSIONS", built_rows=121, excluded_rows=2)
            status = inspect_expert_v0_cache(root)
            self.assertTrue(status.reusable)
            self.assertEqual(status.reason, "PASS_WITH_EXCLUSIONS")

    def test_raise2_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cache(root, entry_raise=2)
            self.assertFalse(inspect_expert_v0_cache(root).reusable)

    def test_hard_failure_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cache(root, status="FAIL", built_rows=122, hard_failed_rows=1)
            self.assertFalse(inspect_expert_v0_cache(root).reusable)

    def test_old_cache_without_exception_policy_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cache(root, include_policy=False)
            self.assertFalse(inspect_expert_v0_cache(root).reusable)


if __name__ == "__main__":
    unittest.main()
