from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from tetrio.datasets.top_players_s1 import decode_playfield
from tetrio.fast_reachability import fast_unique_geometry_keys
from tetrio.reachability import enumerate_tetrio_reachable_placements


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate geometry-set parity and measure single-process speed for "
            "the TETR.IO fast reachability backend."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\expert\top_players_s1_test.parquet"),
    )
    p.add_argument("--rows", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"data\tetrio\processed\fast_reachability_parity.json"),
    )
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required. Install with:\n"
            r".venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def fetch_rows(path: Path, rows: int, seed: int) -> list[dict[str, Any]]:
    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    cur = con.execute(
        f"""
        SELECT
            game_id, subframe, board_before, active_piece, hold_piece,
            preview_queue, placed_piece, use_hold
        FROM read_parquet('{qpath(path)}')
        WHERE hold_label_valid = TRUE
        ORDER BY hash(
            CAST(game_id AS VARCHAR)
            || ':' || CAST(subframe AS VARCHAR)
            || ':{int(seed)}'
        )
        LIMIT {int(rows)}
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, tup)) for tup in cur.fetchall()]


def selected_piece(row: dict[str, Any]) -> str | None:
    active = str(row["active_piece"])
    hold = str(row["hold_piece"])
    preview = str(row["preview_queue"])
    if int(row["use_hold"]) == 0:
        return active
    if hold != "N":
        return hold
    return preview[0] if preview else None


def reference_keys(board, piece: str, max_states: int):
    return {
        p.landing_state.geometry_key()
        for p in enumerate_tetrio_reachable_placements(
            board,
            piece,
            max_states=max_states,
        )
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input parquet not found: {args.input}")
    if args.rows <= 0:
        raise SystemExit("--rows must be positive")

    rows = fetch_rows(args.input, args.rows, args.seed)
    print("=" * 104)
    print("TETR.IO FAST REACHABILITY — REFERENCE GEOMETRY PARITY")
    print("=" * 104)
    print(f"Input       : {args.input}")
    print(f"Rows        : {len(rows):,}")
    print(f"Seed        : {args.seed}")
    print()

    reference_seconds = 0.0
    fast_seconds = 0.0
    total_reference = 0
    total_fast = 0
    mismatches = []
    by_piece = Counter()

    wall_started = time.perf_counter()

    for pos, row in enumerate(rows, 1):
        piece = selected_piece(row)
        if piece is None or piece != str(row["placed_piece"]):
            raise RuntimeError(
                f"validated corpus hold/source mismatch at "
                f"{row['game_id']}/{row['subframe']}"
            )
        board = decode_playfield(str(row["board_before"] or ""))

        t0 = time.perf_counter()
        ref = reference_keys(board, piece, args.reference_max_states)
        reference_seconds += time.perf_counter() - t0

        t0 = time.perf_counter()
        fast = fast_unique_geometry_keys(
            board,
            piece,
            max_states=args.fast_max_states,
        )
        fast_seconds += time.perf_counter() - t0

        total_reference += len(ref)
        total_fast += len(fast)
        by_piece[piece] += 1

        if ref != fast and len(mismatches) < args.max_examples:
            missing = sorted(ref - fast)
            extra = sorted(fast - ref)
            mismatches.append(
                {
                    "game_id": int(row["game_id"]),
                    "subframe": int(row["subframe"]),
                    "piece": piece,
                    "reference_count": len(ref),
                    "fast_count": len(fast),
                    "missing_from_fast": missing[:50],
                    "extra_in_fast": extra[:50],
                }
            )

        if pos % 100 == 0 or pos == len(rows):
            print(
                f"  checked={pos:,}/{len(rows):,} "
                f"mismatch={len(mismatches):,} "
                f"ref={reference_seconds:.2f}s "
                f"fast={fast_seconds:.2f}s"
            )

    wall_seconds = time.perf_counter() - wall_started
    matched_rows = len(rows) - len(mismatches)
    speedup = None if fast_seconds == 0 else reference_seconds / fast_seconds
    exact = len(mismatches) == 0

    report = {
        "metadata": {
            "input": str(args.input),
            "rows": len(rows),
            "seed": args.seed,
            "reference_max_states": args.reference_max_states,
            "fast_max_states": args.fast_max_states,
            "wall_seconds": wall_seconds,
        },
        "result": "PASS" if exact else "FAIL",
        "matched_rows": matched_rows,
        "mismatch_rows": len(mismatches),
        "reference_seconds": reference_seconds,
        "fast_seconds": fast_seconds,
        "speedup": speedup,
        "reference_geometry_count": total_reference,
        "fast_geometry_count": total_fast,
        "by_piece": dict(by_piece),
        "mismatch_examples": mismatches,
        "contract": (
            "unique landing geometry parity only; fast backend does not "
            "preserve spin/path metadata"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Matched rows       : {matched_rows:,}/{len(rows):,}")
    print(f"Mismatch rows      : {len(mismatches):,}")
    print(f"Reference geometry : {total_reference:,}")
    print(f"Fast geometry      : {total_fast:,}")
    print(f"Reference time     : {reference_seconds:.2f}s")
    print(f"Fast time          : {fast_seconds:.2f}s")
    print(f"Speedup            : {speedup:.2f}x" if speedup is not None else "Speedup            : n/a")
    print(f"Output             : {args.output}")
    print(f"Result             : {'PASS' if exact else 'FAIL'}")


if __name__ == "__main__":
    main()
