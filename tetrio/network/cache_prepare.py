from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tetrio.reachability import TETRIO_ENTRY_RAISE_ROWS


@dataclass(frozen=True)
class ExpertV0CacheBuildSpec:
    cache_dir: Path
    source: Path
    rows: int
    backend: str = "fast"
    workers: int = max(1, min(16, (os.cpu_count() or 2) - 2))
    shard_size: int = 8192
    seed: int = 20260906
    fast_max_states: int = 10_000


@dataclass(frozen=True)
class ExpertV0CacheStatus:
    reusable: bool
    reason: str
    shard_count: int
    manifest: dict[str, Any] | None


def inspect_expert_v0_cache(cache_dir: str | Path) -> ExpertV0CacheStatus:
    """Return whether an existing cache is safe to reuse.

    Reuse contract intentionally follows the user's desired behavior:
    if a completed PASS cache exists, use it instead of rebuilding it.
    A partial/failed cache is never silently reused.
    """
    cache_dir = Path(cache_dir)
    shards = sorted(cache_dir.glob("shard_*.npz"))
    manifest_path = cache_dir / "manifest.json"

    if not shards and not manifest_path.exists():
        return ExpertV0CacheStatus(
            reusable=False,
            reason="missing",
            shard_count=0,
            manifest=None,
        )

    if not manifest_path.is_file():
        return ExpertV0CacheStatus(
            reusable=False,
            reason="manifest_missing",
            shard_count=len(shards),
            manifest=None,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ExpertV0CacheStatus(
            reusable=False,
            reason=f"manifest_unreadable:{type(exc).__name__}",
            shard_count=len(shards),
            manifest=None,
        )

    if not shards:
        return ExpertV0CacheStatus(
            reusable=False,
            reason="shards_missing",
            shard_count=0,
            manifest=manifest,
        )

    manifest_status = str(manifest.get("status", "")).upper()
    if manifest_status not in ("PASS", "PASS_WITH_EXCLUSIONS"):
        return ExpertV0CacheStatus(
            reusable=False,
            reason=f"status={manifest.get('status')!r}",
            shard_count=len(shards),
            manifest=manifest,
        )

    built_rows = int(manifest.get("built_rows", 0) or 0)
    failed_rows = int(manifest.get("failed_rows", 0) or 0)
    hard_failed_rows = int(manifest.get("hard_failed_rows", failed_rows) or 0)
    excluded_rows = int(manifest.get("excluded_rows", 0) or 0)
    if built_rows <= 0:
        return ExpertV0CacheStatus(
            reusable=False,
            reason="built_rows_not_positive",
            shard_count=len(shards),
            manifest=manifest,
        )
    if failed_rows != 0 or hard_failed_rows != 0:
        return ExpertV0CacheStatus(
            reusable=False,
            reason=(
                f"hard_failed_rows={hard_failed_rows},"
                f"failed_rows={failed_rows}"
            ),
            shard_count=len(shards),
            manifest=manifest,
        )

    selected_rows = int(manifest.get("selected_rows", 0) or 0)
    if selected_rows and built_rows + excluded_rows != selected_rows:
        return ExpertV0CacheStatus(
            reusable=False,
            reason=(
                "row_accounting_mismatch:"
                f"built={built_rows},excluded={excluded_rows},"
                f"selected={selected_rows}"
            ),
            shard_count=len(shards),
            manifest=manifest,
        )

    contract = manifest.get("reachability_contract")
    if not isinstance(contract, dict):
        return ExpertV0CacheStatus(
            reusable=False,
            reason="reachability_contract_missing",
            shard_count=len(shards),
            manifest=manifest,
        )

    actual_raise = contract.get("entry_raise_rows_vs_generic")
    expected_raise = int(TETRIO_ENTRY_RAISE_ROWS)
    if actual_raise != expected_raise:
        return ExpertV0CacheStatus(
            reusable=False,
            reason=(
                "reachability_contract_mismatch:"
                f"entry_raise_rows={actual_raise!r},expected={expected_raise}"
            ),
            shard_count=len(shards),
            manifest=manifest,
        )

    exception_policy = contract.get("exception_policy")
    expected_policy = "reference_confirmed_expert_unreachable_v1"
    if exception_policy != expected_policy:
        return ExpertV0CacheStatus(
            reusable=False,
            reason=(
                "reachability_contract_mismatch:"
                f"exception_policy={exception_policy!r},"
                f"expected={expected_policy!r}"
            ),
            shard_count=len(shards),
            manifest=manifest,
        )

    return ExpertV0CacheStatus(
        reusable=True,
        reason=manifest_status,
        shard_count=len(shards),
        manifest=manifest,
    )


def ensure_expert_v0_cache(
    spec: ExpertV0CacheBuildSpec,
    *,
    auto_build: bool = True,
    label: str = "cache",
) -> ExpertV0CacheStatus:
    """Reuse a completed cache or build/rebuild it synchronously before training."""
    status = inspect_expert_v0_cache(spec.cache_dir)
    if status.reusable:
        manifest = status.manifest or {}
        built_rows = int(manifest.get("built_rows", 0) or 0)
        backend = manifest.get("backend", "unknown")
        shard_size = manifest.get("shard_size", "unknown")
        entry_raise = (
            (manifest.get("reachability_contract") or {})
            .get("entry_raise_rows_vs_generic")
        )
        excluded_rows = int(manifest.get("excluded_rows", 0) or 0)
        print(
            f"[cache] {label}: reuse {spec.cache_dir} "
            f"({built_rows:,} built, {excluded_rows:,} excluded, "
            f"{status.shard_count} shards, backend={backend}, "
            f"shard_size={shard_size}, entry_raise={entry_raise})"
        )
        return status

    if not auto_build:
        raise FileNotFoundError(
            f"{label} is not reusable at {spec.cache_dir}: {status.reason}. "
            "Automatic cache building is disabled."
        )

    if not spec.source.is_file():
        raise FileNotFoundError(
            f"Cannot build {label}; source parquet not found: {spec.source}"
        )
    if spec.rows < 0:
        raise ValueError("cache build rows must be >= 0 (0 means all source rows)")
    if spec.shard_size <= 0:
        raise ValueError("cache shard size must be positive")
    if spec.workers <= 0:
        raise ValueError("cache workers must be positive")
    if spec.backend not in ("reference", "fast"):
        raise ValueError("cache backend must be 'reference' or 'fast'")

    rebuild = status.reason != "missing"
    action = "rebuild" if rebuild else "build"
    rows_text = "ALL" if spec.rows == 0 else f"{spec.rows:,}"
    print()
    print("=" * 96)
    print(f"EXPERT V0 AUTO CACHE — {action.upper()} {label.upper()}")
    print("=" * 96)
    print(f"Cache        : {spec.cache_dir}")
    print(f"Source       : {spec.source}")
    print(f"Rows         : {rows_text}")
    print(f"Backend      : {spec.backend}")
    print(f"Workers      : {spec.workers}")
    print(f"Shard size   : {spec.shard_size}")
    if rebuild:
        print(f"Reason       : existing cache is incomplete/invalid ({status.reason})")
    else:
        print("Reason       : cache not found")
    print("Training will start automatically after this cache reaches PASS.")
    print()

    cmd = [
        sys.executable,
        "-m",
        "tetrio.tools.build_expert_v0_candidate_cache",
        "--input",
        str(spec.source),
        "--output-dir",
        str(spec.cache_dir),
        "--rows",
        str(spec.rows),
        "--workers",
        str(spec.workers),
        "--backend",
        spec.backend,
        "--shard-size",
        str(spec.shard_size),
        "--seed",
        str(spec.seed),
    ]
    if spec.backend == "fast":
        cmd.extend(["--fast-max-states", str(spec.fast_max_states)])
    if rebuild:
        cmd.append("--overwrite")

    subprocess.run(cmd, check=True)

    final_status = inspect_expert_v0_cache(spec.cache_dir)
    if not final_status.reusable:
        raise RuntimeError(
            f"{label} build command finished but cache is still not reusable: "
            f"{final_status.reason}"
        )

    manifest = final_status.manifest or {}
    print(
        f"[cache] {label}: {manifest.get('status')} -> training may continue "
        f"({int(manifest.get('built_rows', 0)):,} built, "
        f"{int(manifest.get('excluded_rows', 0) or 0):,} excluded)"
    )
    print()
    return final_status
