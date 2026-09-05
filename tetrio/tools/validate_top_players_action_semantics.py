from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

from tetrio.datasets.top_players_action_semantics import (
    ActionMode,
    infer_action_from_previous_post_state,
)


FIELDS = ["game_id", "subframe", "placed", "hold", "next"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Validate whether top-player corpus hold/next fields are post-placement "
            "state and reconstruct no-hold / hold-swap / empty-hold actions."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--max-examples", type=int, default=30)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"data\tetrio\processed\top_players_s1_action_semantics.json"
        ),
    )
    return p.parse_args()


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required. Install it with:\n"
            r".venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def fetch_rows(con, parquet: Path, games: int, seed: int):
    path = qpath(parquet)
    fields = ", ".join(FIELDS)
    cur = con.execute(
        f"""
        WITH raw AS (
            SELECT {fields}
            FROM read_parquet('{path}')
        ),
        chosen AS (
            SELECT game_id
            FROM (SELECT DISTINCT game_id FROM raw)
            ORDER BY hash(CAST(game_id AS VARCHAR) || ':{int(seed)}')
            LIMIT {int(games)}
        )
        SELECT {fields}
        FROM raw JOIN chosen USING (game_id)
        ORDER BY game_id, subframe
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def group_games(rows):
    games, current, gid = [], [], None
    for row in rows:
        if current and row["game_id"] != gid:
            games.append(current)
            current = []
        gid = row["game_id"]
        current.append(row)
    if current:
        games.append(current)
    return games


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
    rows = fetch_rows(con, args.input, args.games, args.seed)
    games = group_games(rows)
    load_s = time.perf_counter() - t0

    counts = Counter()
    placed_counts = defaultdict(Counter)
    unmatched = []
    ambiguous = []

    for game in games:
        for i in range(1, len(game)):
            prev = game[i - 1]
            cur = game[i]
            counts["transitions"] += 1

            inf = infer_action_from_previous_post_state(
                prev_hold=prev["hold"],
                prev_next=prev["next"],
                placed=cur["placed"],
                cur_hold=cur["hold"],
                cur_next=cur["next"],
            )

            if inf.uniquely_classified:
                mode = inf.modes[0]
                counts["unique"] += 1
                counts[mode.value] += 1
                placed_counts[str(cur["placed"])][mode.value] += 1
            elif inf.modes:
                counts["ambiguous"] += 1
                if len(ambiguous) < args.max_examples:
                    ambiguous.append(
                        {
                            "game_id": cur["game_id"],
                            "prev_subframe": prev["subframe"],
                            "subframe": cur["subframe"],
                            "prev_hold": prev["hold"],
                            "prev_next": prev["next"],
                            "placed": cur["placed"],
                            "cur_hold": cur["hold"],
                            "cur_next": cur["next"],
                            "modes": [m.value for m in inf.modes],
                        }
                    )
            else:
                counts["unmatched"] += 1
                if inf.queue_prefix_match:
                    counts["unmatched_but_queue_plausible"] += 1
                if len(unmatched) < args.max_examples:
                    unmatched.append(
                        {
                            "game_id": cur["game_id"],
                            "prev_subframe": prev["subframe"],
                            "subframe": cur["subframe"],
                            "prev_hold": prev["hold"],
                            "prev_next": prev["next"],
                            "placed": cur["placed"],
                            "cur_hold": cur["hold"],
                            "cur_next": cur["next"],
                            "active_piece_hypothesis": inf.active_piece,
                            "reason": inf.reason,
                        }
                    )

    classified = counts["unique"] + counts["ambiguous"]

    print("=" * 104)
    print("TETR.IO TOP-PLAYER CORPUS — HOLD / PREVIEW ACTION SEMANTICS")
    print("=" * 104)
    print(f"Games        : {len(games):,}")
    print(f"Rows         : {len(rows):,}")
    print(f"Transitions  : {counts['transitions']:,}")
    print(f"Classified   : {pct(rate(classified, counts['transitions']))}")
    print(f"Unique       : {pct(rate(counts['unique'], counts['transitions']))}")
    print(f"Ambiguous    : {pct(rate(counts['ambiguous'], counts['transitions']))}")
    print(f"Unmatched    : {pct(rate(counts['unmatched'], counts['transitions']))}")
    print(f"Load time    : {load_s:.2f}s")
    print()
    print("Unique action modes:")
    if counts["unique"]:
        for mode in (ActionMode.NO_HOLD, ActionMode.HOLD_SWAP, ActionMode.HOLD_EMPTY):
            n = counts[mode.value]
            print(f"  {mode.value:10s}: {n:,} ({pct(rate(n, counts['unique']))})")
    print()
    print("By placed piece (unique classifications):")
    for piece in ("I","O","T","S","Z","J","L"):
        c = placed_counts[piece]
        n = sum(c.values())
        print(
            f"  {piece}: n={n:,} "
            f"no_hold={c['no_hold']:,} "
            f"hold_swap={c['hold_swap']:,} "
            f"hold_empty={c['hold_empty']:,}"
        )

    report = {
        "metadata": {
            "input": str(args.input),
            "sampled_games": len(games),
            "sampled_rows": len(rows),
            "seed": args.seed,
            "load_seconds": load_s,
            "hypothesis": (
                "row hold/next are post-placement state; the next row's active "
                "piece begins at previous row next[0]"
            ),
        },
        "counts": dict(counts),
        "rates": {
            "classified": rate(classified, counts["transitions"]),
            "unique": rate(counts["unique"], counts["transitions"]),
            "ambiguous": rate(counts["ambiguous"], counts["transitions"]),
            "unmatched": rate(counts["unmatched"], counts["transitions"]),
        },
        "by_placed_piece": {k: dict(v) for k, v in placed_counts.items()},
        "unmatched_examples": unmatched,
        "ambiguous_examples": ambiguous,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Output       : {args.output}")


if __name__ == "__main__":
    main()
