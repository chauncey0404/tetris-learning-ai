from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time
from typing import Any

from tetrio.datasets.top_players_s1 import (
    binary_board_diff,
    decode_playfield,
    resolve_dataset_mapping,
    simulate_mapped_dataset_placement,
)

FIELDS = [
    "game_id","subframe","won","playfield","x","y","r","placed","hold","next",
    "cleared","garbage_cleared","attack","t_spin","btb","combo",
    "immediate_garbage","incoming_garbage","rating","glicko","glicko_rd",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path(r"data\tetrio\processed\top_players_s1.parquet"))
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--output", type=Path,
                   default=Path(r"data\tetrio\processed\top_players_s1_semantic_validation.json"))
    p.add_argument("--max-examples", type=int, default=50)
    return p.parse_args()


def require_duckdb():
    import duckdb
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def fetch_sample(con, parquet: Path, games: int, seed: int):
    path = qpath(parquet)
    select_fields = ", ".join(FIELDS)
    cur = con.execute(f"""
        WITH raw AS (
            SELECT {select_fields}
            FROM read_parquet('{path}')
        ),
        chosen AS (
            SELECT game_id
            FROM (SELECT DISTINCT game_id FROM raw)
            ORDER BY hash(CAST(game_id AS VARCHAR) || ':{int(seed)}')
            LIMIT {int(games)}
        )
        SELECT {select_fields}
        FROM raw JOIN chosen USING (game_id)
        ORDER BY game_id, subframe
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def group_games(rows):
    result, current, current_id = [], [], None
    for row in rows:
        gid = row["game_id"]
        if current and gid != current_id:
            result.append(current)
            current = []
        current_id = gid
        current.append(row)
    if current:
        result.append(current)
    return result


def is_zeroish(v: Any) -> bool:
    if v is None or v == "":
        return True
    try:
        return float(v) == 0.0
    except Exception:
        return False


def quiet_pair(a, b) -> bool:
    return all(
        is_zeroish(row.get(field))
        for row in (a, b)
        for field in ("immediate_garbage", "incoming_garbage")
    )


def rate(n, d):
    return None if not d else n / d


def pct(v):
    return "n/a" if v is None else f"{100*v:.4f}%"


def main():
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Parquet not found: {args.input}")

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {max(1, min(20, os.cpu_count() or 1))}")

    t0 = time.perf_counter()
    rows = fetch_sample(con, args.input, args.games, args.seed)
    games = group_games(rows)
    load_s = time.perf_counter() - t0

    counts = Counter()
    per_piece = defaultdict(Counter)
    per_group = defaultdict(Counter)
    unmapped = Counter()
    examples = []

    t1 = time.perf_counter()
    for game in games:
        for i in range(len(game) - 1):
            row, nxt = game[i], game[i + 1]
            piece, rcode = str(row["placed"]), str(row["r"])
            counts["transitions"] += 1
            per_piece[piece]["transitions"] += 1
            per_group[(piece, rcode)]["transitions"] += 1

            try:
                mapping = resolve_dataset_mapping(piece, rcode)
            except ValueError:
                counts["unmapped"] += 1
                per_piece[piece]["unmapped"] += 1
                per_group[(piece, rcode)]["unmapped"] += 1
                unmapped[(piece, rcode)] += 1
                continue

            board = decode_playfield(row["playfield"])
            expected = decode_playfield(nxt["playfield"])
            sim = simulate_mapped_dataset_placement(
                board, piece=piece, x=row["x"], y=row["y"], rotation_code=rcode
            )

            if not sim.legal:
                counts["illegal"] += 1
                per_piece[piece]["illegal"] += 1
                per_group[(piece, rcode)]["illegal"] += 1
                continue

            counts["legal"] += 1
            per_piece[piece]["legal"] += 1
            per_group[(piece, rcode)]["legal"] += 1

            if sim.lines_cleared == int(row["cleared"]):
                counts["clear_match"] += 1
                per_piece[piece]["clear_match"] += 1
                per_group[(piece, rcode)]["clear_match"] += 1

            diff = binary_board_diff(sim.board_after_clear, expected)
            if diff == 0:
                counts["board_exact"] += 1
                per_piece[piece]["board_exact"] += 1
                per_group[(piece, rcode)]["board_exact"] += 1

            if quiet_pair(row, nxt):
                counts["quiet"] += 1
                per_piece[piece]["quiet"] += 1
                per_group[(piece, rcode)]["quiet"] += 1
                if diff == 0:
                    counts["quiet_exact"] += 1
                    per_piece[piece]["quiet_exact"] += 1
                    per_group[(piece, rcode)]["quiet_exact"] += 1
                elif len(examples) < args.max_examples:
                    examples.append({
                        "game_id": row["game_id"],
                        "subframe": row["subframe"],
                        "piece": piece,
                        "rotation": rcode,
                        "canonical_piece": mapping.canonical_piece,
                        "x": row["x"],
                        "y": row["y"],
                        "board_diff_cells": diff,
                    })

    eval_s = time.perf_counter() - t1
    mapped = counts["transitions"] - counts["unmapped"]

    print("=" * 104)
    print("TETR.IO TOP-PLAYER CORPUS — LOCKED MAPPING CONTINUITY VALIDATION")
    print("=" * 104)
    print(f"Games       : {len(games):,}")
    print(f"Rows        : {len(rows):,}")
    print(f"Transitions : {counts['transitions']:,}")
    print(f"Mapped      : {pct(rate(mapped, counts['transitions']))}")
    print(f"Legal       : {pct(rate(counts['legal'], mapped))}")
    print(f"Clear match : {pct(rate(counts['clear_match'], counts['legal']))}")
    print(f"Board exact : {pct(rate(counts['board_exact'], counts['legal']))}")
    print(f"Quiet exact : {pct(rate(counts['quiet_exact'], counts['quiet']))} (n={counts['quiet']:,})")
    print(f"Load time   : {load_s:.2f}s")
    print(f"Eval time   : {eval_s:.2f}s")
    print()
    print("Per piece:")
    for piece in ("I","O","T","S","Z","J","L"):
        c = per_piece[piece]
        pmapped = c["transitions"] - c["unmapped"]
        print(
            f"  {piece}: n={c['transitions']:,} "
            f"mapped={pct(rate(pmapped,c['transitions'])):>9} "
            f"legal={pct(rate(c['legal'],pmapped)):>9} "
            f"clear={pct(rate(c['clear_match'],c['legal'])):>9} "
            f"quiet={pct(rate(c['quiet_exact'],c['quiet'])):>9}"
        )

    report = {
        "metadata": {
            "input": str(args.input),
            "sampled_games": len(games),
            "sampled_rows": len(rows),
            "seed": args.seed,
            "load_seconds": load_s,
            "eval_seconds": eval_s,
        },
        "counts": dict(counts),
        "rates": {
            "mapped": rate(mapped, counts["transitions"]),
            "legal_given_mapped": rate(counts["legal"], mapped),
            "clear_match_given_legal": rate(counts["clear_match"], counts["legal"]),
            "board_exact_given_legal": rate(counts["board_exact"], counts["legal"]),
            "quiet_exact": rate(counts["quiet_exact"], counts["quiet"]),
        },
        "per_piece": {k: dict(v) for k, v in per_piece.items()},
        "per_piece_orientation": {
            f"{p}-{r}": dict(v) for (p,r), v in per_group.items()
        },
        "unmapped": {f"{p}-{r}": n for (p,r), n in unmapped.items()},
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Output      : {args.output}")


if __name__ == "__main__":
    main()
