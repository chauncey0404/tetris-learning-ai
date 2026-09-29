from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from tetrio.stateful.features import (
    STATEFUL_FEATURE_NAMES,
    STATEFUL_FEATURE_SIZE,
    encode_battle_state_row,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Align audited causal battle state to an existing V1.1 future "
            "cache without rebuilding reachability/future features."
        )
    )
    p.add_argument("--future-cache", type=Path, required=True)
    p.add_argument("--battle-state-parquet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, min(20, os.cpu_count() or 1)),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required; it was also used by the battle-state audit."
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def _future_paths(cache_dir: Path) -> list[Path]:
    paths = sorted(cache_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No future shards found in {cache_dir}")
    return paths


def main() -> None:
    args = parse_args()
    if not args.battle_state_parquet.is_file():
        raise SystemExit(f"Battle-state parquet not found: {args.battle_state_parquet}")

    future_paths = _future_paths(args.future_cache)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    existing = list(args.output_dir.glob("shard_*.npz")) + [manifest_path]
    existing = [p for p in existing if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Output already exists in {args.output_dir}. Use --overwrite."
        )
    if args.overwrite:
        for p in existing:
            p.unlink()

    shard_counts: list[int] = []
    all_game: list[np.ndarray] = []
    all_sub: list[np.ndarray] = []
    for path in future_paths:
        with np.load(path, allow_pickle=False) as d:
            game = np.asarray(d["game_id"], dtype=np.int64)
            sub = np.asarray(d["subframe"], dtype=np.int64)
            if game.shape != sub.shape:
                raise SystemExit(f"Identity shape mismatch in {path}")
            shard_counts.append(len(game))
            all_game.append(game)
            all_sub.append(sub)

    target_game = np.concatenate(all_game)
    target_sub = np.concatenate(all_sub)
    total = len(target_game)
    target_keys = set(zip(target_game.tolist(), target_sub.tolist()))

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={max(1, int(args.threads))}")
    con.execute("SET preserve_insertion_order=true")
    battle = qpath(args.battle_state_parquet)

    # Fail closed if the future cache touches an ambiguous game/subframe key.
    # The historical train split has only a handful; V1.2A must never guess.
    duplicate_rows = con.execute(
        f"""
        SELECT game_id, subframe, count(*) AS n
        FROM read_parquet('{battle}')
        GROUP BY game_id, subframe
        HAVING count(*) > 1
        """
    ).fetchall()
    duplicate_keys = {(int(g), int(s)) for g, s, _ in duplicate_rows}
    ambiguous = sorted(target_keys.intersection(duplicate_keys))
    if ambiguous:
        preview = ambiguous[:10]
        raise SystemExit(
            "Future cache intersects ambiguous duplicate (game_id,subframe) "
            f"keys: {preview}. Refusing to guess. Enhance future-cache row "
            "identity before scaling this split."
        )

    print("=" * 108)
    print("TETR.IO EXPERT V1.2A — CAUSAL STATE CACHE ADAPTER")
    print("=" * 108)
    print(f"Future cache : {args.future_cache}")
    print(f"Battle state : {args.battle_state_parquet}")
    print(f"Rows         : {total:,}")
    print(f"Shards       : {len(future_paths)}")
    print(f"Duplicate keys in split: {len(duplicate_keys):,}")
    print("Alignment    : exact unordered key lookup, fail-closed on ambiguous keys")
    print()

    # The V1/V1.1 cache order is NOT guaranteed to follow the original expert
    # split Parquet order.  Order is therefore not part of row identity.
    #
    # Because ambiguous duplicate (game_id, subframe) keys were rejected above,
    # every remaining key maps to exactly one audited battle-state row.  Build
    # an index from exact key -> one or more future-cache positions, stream the
    # battle-state split once, and write each matched feature back to its
    # original future-cache position.
    target_positions: dict[tuple[int, int], list[int]] = {}
    for i, (g, s) in enumerate(zip(target_game.tolist(), target_sub.tolist())):
        target_positions.setdefault((int(g), int(s)), []).append(i)

    future_duplicate_keys = sum(
        1 for positions in target_positions.values() if len(positions) > 1
    )
    unique_target_keys = len(target_positions)

    query = con.execute(
        f"""
        SELECT
            game_id,
            subframe,
            raw_combo_before,
            raw_btb_before,
            previous_cleared,
            previous_t_spin,
            previous_attack,
            previous_garbage_cleared
        FROM read_parquet('{battle}')
        """
    )

    state_features = np.empty((total, STATEFUL_FEATURE_SIZE), dtype=np.float32)
    matched = np.zeros(total, dtype=np.bool_)
    matched_keys = 0
    scanned_rows = 0
    started = time.perf_counter()

    while matched_keys < unique_target_keys:
        rows = query.fetchmany(65536)
        if not rows:
            break

        for row in rows:
            scanned_rows += 1
            key = (int(row[0]), int(row[1]))
            positions = target_positions.get(key)
            if positions is None:
                continue

            feature = encode_battle_state_row(
                raw_combo_before=row[2],
                raw_btb_before=row[3],
                previous_cleared=row[4],
                previous_t_spin=row[5],
                previous_attack=row[6],
                previous_garbage_cleared=row[7],
            )
            for pos in positions:
                state_features[pos] = feature
                matched[pos] = True

            matched_keys += 1
            # Removing the resolved key both prevents accidental double-match
            # and makes the remaining dictionary smaller during the long scan.
            del target_positions[key]

            if matched_keys >= unique_target_keys:
                break

    matched_rows = int(np.count_nonzero(matched))
    if matched_rows != total:
        missing_positions = np.flatnonzero(~matched)[:10].tolist()
        missing_preview = [
            (
                int(target_game[i]),
                int(target_sub[i]),
            )
            for i in missing_positions
        ]
        raise SystemExit(
            "Exact battle-state alignment failed: "
            f"matched={matched_rows:,}/{total:,}; "
            f"missing preview={missing_preview}. Refusing approximate alignment."
        )

    offset = 0
    manifest_shards = []
    for path, count in zip(future_paths, shard_counts):
        hi = offset + count
        out = args.output_dir / path.name
        np.savez(
            out,
            state_features=state_features[offset:hi],
            game_id=target_game[offset:hi],
            subframe=target_sub[offset:hi],
        )
        manifest_shards.append({"file": path.name, "rows": count})
        offset = hi

    elapsed = time.perf_counter() - started
    manifest = {
        "format": "tetrio_expert_v1_2_state_cache_v1",
        "status": "PASS",
        "future_cache": str(args.future_cache),
        "battle_state_parquet": str(args.battle_state_parquet),
        "rows": total,
        "feature_names": list(STATEFUL_FEATURE_NAMES),
        "alignment": "exact_unordered_key_lookup_fail_closed_duplicate_keys",
        "scanned_battle_rows": scanned_rows,
        "future_duplicate_keys": future_duplicate_keys,
        "duplicate_keys_in_split": len(duplicate_keys),
        "seconds": elapsed,
        "shards": manifest_shards,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Matched rows : {matched_rows:,}/{total:,} PASS")
    print(f"Scanned rows : {scanned_rows:,}")
    print(f"Future duplicate keys: {future_duplicate_keys:,}")
    print(f"Elapsed      : {elapsed:.2f}s")
    print(f"Manifest     : {manifest_path}")
    print("Status       : PASS")


if __name__ == "__main__":
    main()
