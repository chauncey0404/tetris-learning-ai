from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from tetrio.datasets.top_players_s1 import decode_playfield
from tetrio.reachability import enumerate_tetrio_reachable_placements
from tetris_ai.core.types import PieceState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strict historical-expert ↔ production TETR.IO reachability gate."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\expert\top_players_s1_test.parquet"),
    )
    p.add_argument("--per-group", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 2)),
    )
    p.add_argument("--max-states", type=int, default=50_000)
    p.add_argument("--max-examples", type=int, default=30)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"data\tetrio\processed\top_players_s1_reachability_gate.json"
        ),
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


def stable_row_key(row: dict[str, Any], seed: int) -> str:
    raw = (
        f"{seed}|{row['game_id']}|{row['subframe']}|{row['placed_piece']}|"
        f"{row['hold_mode']}|{row['final_rotation']}"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fetch_stratified_rows(
    parquet: Path,
    *,
    per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    path = qpath(parquet)
    cur = con.execute(
        f"""
        WITH ranked AS (
            SELECT
                game_id,
                subframe,
                board_before,
                active_piece,
                hold_piece,
                preview_queue,
                placed_piece,
                final_x,
                final_y,
                final_rotation,
                use_hold,
                hold_mode,
                row_number() OVER (
                    PARTITION BY placed_piece, hold_mode, final_rotation
                    ORDER BY hash(
                        CAST(game_id AS VARCHAR)
                        || ':'
                        || CAST(subframe AS VARCHAR)
                        || ':{int(seed)}'
                    )
                ) AS rn
            FROM read_parquet('{path}')
            WHERE hold_label_valid = TRUE
        )
        SELECT * EXCLUDE(rn)
        FROM ranked
        WHERE rn <= {int(per_group)}
        """
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, tup)) for tup in cur.fetchall()]
    rows.sort(key=lambda row: stable_row_key(row, seed))
    return rows


def expected_piece(row: dict[str, Any]) -> str | None:
    active = str(row["active_piece"])
    hold = str(row["hold_piece"])
    preview = str(row["preview_queue"])
    mode = str(row["hold_mode"])
    if mode == "no_hold":
        return active
    if mode == "hold_swap":
        return None if hold == "N" else hold
    if mode == "hold_empty":
        return preview[0] if preview else None
    return None


def evaluate_one(item: tuple[dict[str, Any], int]) -> dict[str, Any]:
    row, max_states = item
    target_piece = str(row["placed_piece"])
    source_piece = expected_piece(row)
    result = {
        "game_id": int(row["game_id"]),
        "subframe": int(row["subframe"]),
        "piece": target_piece,
        "hold_mode": str(row["hold_mode"]),
        "rotation": int(row["final_rotation"]),
        "hold_ok": source_piece == target_piece,
        "exact": False,
        "candidate_count": 0,
        "error": None,
    }

    if source_piece != target_piece:
        result["error"] = "hold_piece_selection_mismatch"
        return result

    target = PieceState(
        piece=target_piece,
        x=int(row["final_x"]),
        y=int(row["final_y"]),
        rotation=int(row["final_rotation"]),
    )

    try:
        board = decode_playfield(str(row["board_before"] or ""))
        placements = enumerate_tetrio_reachable_placements(
            board,
            target_piece,
            max_states=max_states,
        )
        result["candidate_count"] = len(placements)
        result["exact"] = any(
            p.landing_state.geometry_key() == target.geometry_key()
            for p in placements
        )
        if not placements:
            result["error"] = "no_reachable_placements"
        elif not result["exact"]:
            result["error"] = "expert_geometry_not_reachable"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"

    return result


def rate(n: int, d: int) -> float | None:
    return None if d == 0 else n / d


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100.0 * v:.4f}%"


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Expert Parquet not found: {args.input}")

    rows = fetch_stratified_rows(
        args.input,
        per_group=args.per_group,
        seed=args.seed,
    )
    if not rows:
        raise SystemExit("No rows selected")

    print("=" * 104)
    print("TETR.IO EXPERT ↔ PRODUCTION REACHABILITY GATE")
    print("=" * 104)
    print(f"Rows        : {len(rows):,}")
    print(f"Per group   : {args.per_group}")
    print(f"Workers     : {args.workers}")
    print(f"Seed        : {args.seed}")
    print()

    started = time.perf_counter()
    work = [(row, args.max_states) for row in rows]
    if args.workers == 1:
        results = [evaluate_one(x) for x in work]
    else:
        chunksize = max(1, len(work) // (args.workers * 8))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(evaluate_one, work, chunksize=chunksize))
    elapsed = time.perf_counter() - started

    counts = Counter()
    by_piece = defaultdict(Counter)
    by_hold = defaultdict(Counter)
    by_rotation = defaultdict(Counter)
    examples: list[dict[str, Any]] = []

    for result in results:
        counts["rows"] += 1
        buckets = (
            by_piece[result["piece"]],
            by_hold[result["hold_mode"]],
            by_rotation[str(result["rotation"])],
        )
        for bucket in buckets:
            bucket["rows"] += 1

        if result["hold_ok"]:
            counts["hold_ok"] += 1
            for bucket in buckets:
                bucket["hold_ok"] += 1

        if result["exact"]:
            counts["exact"] += 1
            for bucket in buckets:
                bucket["exact"] += 1
        else:
            counts["miss"] += 1
            if len(examples) < args.max_examples:
                examples.append(result)

    print(f"Hold semantics : {pct(rate(counts['hold_ok'], counts['rows']))}")
    print(f"Exact geometry : {pct(rate(counts['exact'], counts['rows']))}")
    print(f"Miss           : {counts['miss']:,}")
    print(f"Eval time      : {elapsed:.2f}s")
    print(f"Throughput     : {(len(results) / elapsed if elapsed else 0):.2f} rows/s")
    print()

    print("Per piece:")
    for piece in ("I", "O", "T", "S", "Z", "J", "L"):
        c = by_piece[piece]
        if c["rows"]:
            print(
                f"  {piece}: n={c['rows']:,} "
                f"exact={pct(rate(c['exact'], c['rows'])):>9}"
            )

    print()
    print("Per hold mode:")
    for mode in ("no_hold", "hold_swap", "hold_empty"):
        c = by_hold[mode]
        if c["rows"]:
            print(
                f"  {mode:10s}: n={c['rows']:,} "
                f"exact={pct(rate(c['exact'], c['rows'])):>9}"
            )

    print()
    print("Per rotation:")
    for rotation in ("0", "1", "2", "3"):
        c = by_rotation[rotation]
        if c["rows"]:
            print(
                f"  r={rotation}: n={c['rows']:,} "
                f"exact={pct(rate(c['exact'], c['rows'])):>9}"
            )

    passed = (
        counts["hold_ok"] == counts["rows"]
        and counts["exact"] == counts["rows"]
    )
    report = {
        "metadata": {
            "input": str(args.input),
            "rows": len(rows),
            "per_group": args.per_group,
            "workers": args.workers,
            "max_states": args.max_states,
            "seed": args.seed,
            "eval_seconds": elapsed,
            "entry_semantics": "TETR.IO-specific generic spawn shifted upward by 1 row",
        },
        "counts": dict(counts),
        "rates": {
            "hold_semantics": rate(counts["hold_ok"], counts["rows"]),
            "exact_geometry": rate(counts["exact"], counts["rows"]),
        },
        "by_piece": {k: dict(v) for k, v in by_piece.items()},
        "by_hold_mode": {k: dict(v) for k, v in by_hold.items()},
        "by_rotation": {k: dict(v) for k, v in by_rotation.items()},
        "mismatch_examples": examples,
        "result": "PASS" if passed else "FAIL",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Output       : {args.output}")
    print(f"Result       : {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
