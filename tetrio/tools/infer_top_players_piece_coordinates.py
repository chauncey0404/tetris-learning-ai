from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

from tetrio.datasets.top_players_s1 import (
    CoordinateConvention,
    binary_board_diff,
    decode_playfield,
    simulate_dataset_placement,
)

PIECES = ("I", "O", "T", "S", "Z", "J", "L")
ROT_CODES = ("N", "E", "S", "W")

# Keep the search physically plausible and small.
# J/L and S/Z swaps are included because a dataset extractor can legitimately
# use a different enum/name mapping even when board geometry is otherwise sane.
PIECE_CANDIDATES = {
    "I": ("I",),
    "O": ("O",),
    "T": ("T",),
    "S": ("S", "Z"),
    "Z": ("Z", "S"),
    "J": ("J", "L"),
    "L": ("L", "J"),
}


@dataclass(frozen=True)
class Sample:
    game_id: int
    subframe: int
    playfield: str
    next_playfield: str
    x: int
    y: int
    r: str
    placed: str
    cleared: int
    immediate_garbage: float
    incoming_garbage: float
    next_immediate_garbage: float
    next_incoming_garbage: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Infer piece/orientation-specific coordinate conventions for the "
            "historical TETR.IO top-player corpus using exact board continuity."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument(
        "--x-offsets",
        default="-3,-2,-1,0,1",
        help="Comma-separated PieceState.x offsets to test.",
    )
    p.add_argument(
        "--y-bases",
        default="36,37,38,39,40",
        help="Comma-separated PieceState.y bases to test.",
    )
    p.add_argument(
        "--rotation-maps",
        default="cw,ccw",
        help="Comma-separated rotation maps to test.",
    )
    p.add_argument(
        "--max-per-group",
        type=int,
        default=2000,
        help="Maximum quiet transitions retained for each placed-piece/r group.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"data\tetrio\processed\top_players_s1_piece_coordinate_inference.json"
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


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def is_zeroish(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def fetch_quiet_samples(
    con,
    parquet: Path,
    games: int,
    seed: int,
    max_per_group: int,
) -> dict[tuple[str, str], list[Sample]]:
    """Fetch deterministic adjacent-row transitions and keep conservative quiet pairs."""

    path = qpath(parquet)
    cur = con.execute(
        f"""
        WITH raw AS (
            SELECT
                game_id, subframe, playfield, x, y, r, placed, cleared,
                immediate_garbage, incoming_garbage
            FROM read_parquet('{path}')
        ),
        chosen AS (
            SELECT game_id
            FROM (SELECT DISTINCT game_id FROM raw)
            ORDER BY hash(CAST(game_id AS VARCHAR) || ':{int(seed)}')
            LIMIT {int(games)}
        ),
        seq AS (
            SELECT
                r.*,
                lead(playfield) OVER (
                    PARTITION BY game_id ORDER BY subframe
                ) AS next_playfield,
                lead(immediate_garbage) OVER (
                    PARTITION BY game_id ORDER BY subframe
                ) AS next_immediate_garbage,
                lead(incoming_garbage) OVER (
                    PARTITION BY game_id ORDER BY subframe
                ) AS next_incoming_garbage
            FROM raw r
            JOIN chosen USING (game_id)
        )
        SELECT *
        FROM seq
        WHERE next_playfield IS NOT NULL
        ORDER BY game_id, subframe
        """
    )
    cols = [d[0] for d in cur.description]
    groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)

    for tup in cur.fetchall():
        row = dict(zip(cols, tup))
        if not all(
            is_zeroish(row.get(field))
            for field in (
                "immediate_garbage",
                "incoming_garbage",
                "next_immediate_garbage",
                "next_incoming_garbage",
            )
        ):
            continue
        key = (str(row["placed"]), str(row["r"]))
        bucket = groups[key]
        if len(bucket) >= max_per_group:
            continue
        bucket.append(
            Sample(
                game_id=int(row["game_id"]),
                subframe=int(row["subframe"]),
                playfield=str(row["playfield"] or ""),
                next_playfield=str(row["next_playfield"] or ""),
                x=int(row["x"]),
                y=int(row["y"]),
                r=str(row["r"]),
                placed=str(row["placed"]),
                cleared=int(row["cleared"]),
                immediate_garbage=float(row["immediate_garbage"] or 0),
                incoming_garbage=float(row["incoming_garbage"] or 0),
                next_immediate_garbage=float(row["next_immediate_garbage"] or 0),
                next_incoming_garbage=float(row["next_incoming_garbage"] or 0),
            )
        )
    return groups


def score_candidate(
    samples: list[Sample],
    *,
    canonical_piece: str,
    x_offset: int,
    y_base: int,
    rotation_map: str,
) -> dict[str, Any]:
    legal = 0
    exact = 0
    clear_match = 0
    total_diff = 0
    examples = []

    convention = CoordinateConvention(
        x_offset=x_offset,
        y_base=y_base,
        rotation_map_name=rotation_map,
    )

    for sample in samples:
        before = decode_playfield(sample.playfield)
        expected = decode_playfield(sample.next_playfield)
        sim = simulate_dataset_placement(
            before,
            piece=canonical_piece,
            x=sample.x,
            y=sample.y,
            rotation_code=sample.r,
            convention=convention,
        )
        if not sim.legal:
            continue
        legal += 1
        if sim.lines_cleared == sample.cleared:
            clear_match += 1
        diff = binary_board_diff(sim.board_after_clear, expected)
        total_diff += diff
        if diff == 0:
            exact += 1
        elif len(examples) < 3:
            examples.append(
                {
                    "game_id": sample.game_id,
                    "subframe": sample.subframe,
                    "x": sample.x,
                    "y": sample.y,
                    "r": sample.r,
                    "reported_cleared": sample.cleared,
                    "simulated_cleared": sim.lines_cleared,
                    "board_diff_cells": diff,
                }
            )

    n = len(samples)
    return {
        "dataset_piece": samples[0].placed if samples else None,
        "rotation_code": samples[0].r if samples else None,
        "canonical_piece": canonical_piece,
        "x_offset": x_offset,
        "y_base": y_base,
        "rotation_map": rotation_map,
        "samples": n,
        "legal": legal,
        "exact": exact,
        "clear_match": clear_match,
        "legal_rate": 0.0 if n == 0 else legal / n,
        "exact_rate_all": 0.0 if n == 0 else exact / n,
        "exact_rate_legal": 0.0 if legal == 0 else exact / legal,
        "clear_match_rate_legal": 0.0 if legal == 0 else clear_match / legal,
        "mean_board_diff_legal": None if legal == 0 else total_diff / legal,
        "examples": examples,
    }


def candidate_key(row: dict[str, Any]) -> tuple:
    # Exact continuity over all sampled quiet transitions is the strongest signal.
    # Then prefer legality, line-clear agreement and smaller board difference.
    mean_diff = row["mean_board_diff_legal"]
    if mean_diff is None:
        mean_diff = 1e9
    return (
        row["exact_rate_all"],
        row["legal_rate"],
        row["clear_match_rate_legal"],
        -mean_diff,
    )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Parquet not found: {args.input}")

    x_offsets = parse_ints(args.x_offsets)
    y_bases = parse_ints(args.y_bases)
    rotation_maps = tuple(
        x.strip() for x in args.rotation_maps.split(",") if x.strip()
    )

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {max(1, min(20, os.cpu_count() or 1))}")

    print("=" * 108)
    print("TETR.IO TOP-PLAYER CORPUS — PIECE/ORIENTATION CONVENTION INFERENCE")
    print("=" * 108)
    print(f"Input          : {args.input}")
    print(f"Sample games   : {args.games}")
    print(f"Seed           : {args.seed}")
    print(f"Max/group      : {args.max_per_group}")
    print(f"x offsets      : {x_offsets}")
    print(f"y bases        : {y_bases}")
    print(f"rotation maps  : {rotation_maps}")
    print()

    started = time.perf_counter()
    groups = fetch_quiet_samples(
        con,
        args.input,
        args.games,
        args.seed,
        args.max_per_group,
    )
    print(
        f"Fetched {sum(len(v) for v in groups.values()):,} quiet transitions "
        f"across {len(groups)} piece/orientation groups in "
        f"{time.perf_counter() - started:.2f}s."
    )
    print()

    report: dict[str, Any] = {
        "metadata": {
            "input": str(args.input),
            "games": args.games,
            "seed": args.seed,
            "max_per_group": args.max_per_group,
            "x_offsets": x_offsets,
            "y_bases": y_bases,
            "rotation_maps": rotation_maps,
            "purpose": (
                "Infer dataset label/coordinate conventions from exact quiet "
                "board continuity before expert-model training."
            ),
        },
        "groups": {},
    }

    for dataset_piece in PIECES:
        print(f"[{dataset_piece}]")
        piece_report: dict[str, Any] = {}
        for rcode in ROT_CODES:
            samples = groups.get((dataset_piece, rcode), [])
            if not samples:
                print(f"  {rcode}: no samples")
                piece_report[rcode] = {"samples": 0, "top": []}
                continue

            rows: list[dict[str, Any]] = []
            for canonical_piece in PIECE_CANDIDATES[dataset_piece]:
                for rotation_map in rotation_maps:
                    for x_offset in x_offsets:
                        for y_base in y_bases:
                            rows.append(
                                score_candidate(
                                    samples,
                                    canonical_piece=canonical_piece,
                                    x_offset=x_offset,
                                    y_base=y_base,
                                    rotation_map=rotation_map,
                                )
                            )
            rows.sort(key=candidate_key, reverse=True)
            top = rows[:8]
            best = top[0]
            print(
                f"  {rcode}: n={len(samples):>4} -> "
                f"piece={best['canonical_piece']} "
                f"xoff={best['x_offset']:+d} "
                f"ybase={best['y_base']} "
                f"rot={best['rotation_map']} | "
                f"exact={100*best['exact_rate_all']:.2f}% "
                f"legal={100*best['legal_rate']:.2f}% "
                f"clear={100*best['clear_match_rate_legal']:.2f}%"
            )
            piece_report[rcode] = {
                "samples": len(samples),
                "top": top,
            }
        report["groups"][dataset_piece] = piece_report
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Output: {args.output}")
    print("=" * 108)


if __name__ == "__main__":
    main()
