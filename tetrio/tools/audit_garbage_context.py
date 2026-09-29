from __future__ import annotations

import argparse
import bisect
from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import median
import time
from typing import Any


FIELDS = ("incoming_garbage", "immediate_garbage")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fail-closed timing/transport evidence audit for the historical "
            "placement-level incoming_garbage and immediate_garbage columns. "
            "This does not auto-approve them as model inputs."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, min(20, os.cpu_count() or 1)),
    )
    p.add_argument(
        "--lead-window-frames",
        type=int,
        default=120,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"artifacts\tetrio\garbage_context_audit.json"
        ),
    )
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required. Install it with:\n"
            r"  .venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def nearest_signed_lags(
    source_frames: list[int],
    target_frames: list[int],
    window: int,
) -> list[int]:
    if not source_frames or not target_frames:
        return []
    out = []
    for f in source_frames:
        i = bisect.bisect_left(target_frames, f)
        candidates = []
        if i < len(target_frames):
            candidates.append(target_frames[i] - f)
        if i > 0:
            candidates.append(target_frames[i - 1] - f)
        if not candidates:
            continue
        lag = min(candidates, key=lambda x: abs(x))
        if abs(lag) <= window:
            out.append(int(lag))
    return out


def process_game(
    rows: list[tuple],
    *,
    lead_window_frames: int,
    agg: dict,
) -> None:
    if len(rows) < 3:
        return

    # Row tuple:
    # subframe, attack, garbage_cleared, incoming_garbage, immediate_garbage
    parsed = []
    for row in rows:
        parsed.append(
            (
                int(row[0]),
                finite_number(row[1]) or 0.0,
                finite_number(row[2]) or 0.0,
                finite_number(row[3]),
                finite_number(row[4]),
            )
        )

    inc_rises: list[int] = []
    imm_rises: list[int] = []

    for i in range(1, len(parsed)):
        prev = parsed[i - 1]
        cur = parsed[i]
        inc_prev, inc_cur = prev[3], cur[3]
        imm_prev, imm_cur = prev[4], cur[4]

        if inc_prev is not None and inc_cur is not None:
            if inc_cur > inc_prev:
                inc_rises.append(cur[0])
        if imm_prev is not None and imm_cur is not None:
            if imm_cur > imm_prev:
                imm_rises.append(cur[0])

        if inc_cur is not None and imm_cur is not None:
            agg["pair_rows"] += 1
            if imm_cur <= inc_cur:
                agg["immediate_le_incoming"] += 1
            if inc_cur <= imm_cur:
                agg["incoming_le_immediate"] += 1
            if inc_cur == imm_cur:
                agg["equal_pair"] += 1

    lags = nearest_signed_lags(
        inc_rises,
        imm_rises,
        int(lead_window_frames),
    )
    agg["incoming_to_immediate_lags"].extend(lags)

    for i in range(1, len(parsed) - 1):
        prev = parsed[i - 1]
        cur = parsed[i]
        nxt = parsed[i + 1]
        attack = max(0.0, cur[1])

        if attack <= 0.0:
            continue

        for field, idx in (
            ("incoming_garbage", 3),
            ("immediate_garbage", 4),
        ):
            pv, cv, nv = prev[idx], cur[idx], nxt[idx]
            if pv is None or cv is None or nv is None:
                continue

            stat = agg["attack_conditioned"][field]

            if pv > 0.0:
                possible = min(attack, pv)
                if possible > 0.0:
                    drop = max(0.0, pv - cv)
                    stat["post_n"] += 1
                    stat["post_support"] += int(drop > 0.0)
                    stat["post_capture"].append(
                        min(drop, possible) / possible
                    )

            if cv > 0.0:
                possible = min(attack, cv)
                if possible > 0.0:
                    drop = max(0.0, cv - nv)
                    stat["pre_n"] += 1
                    stat["pre_support"] += int(drop > 0.0)
                    stat["pre_capture"].append(
                        min(drop, possible) / possible
                    )


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={int(args.threads)}")
    con.execute("SET preserve_insertion_order=true")
    inp = qpath(args.input)

    print("=" * 112)
    print("TETR.IO HISTORICAL GARBAGE CONTEXT — TIMING / TRANSPORT EVIDENCE AUDIT")
    print("=" * 112)
    print(f"Input       : {args.input}")
    print(f"Games       : {args.games:,}")
    print(f"Seed        : {args.seed}")
    print(f"Lead window : ±{args.lead_window_frames} frames")
    print("Approval    : FAIL-CLOSED; historical placement data alone never auto-approves")
    print()

    started = time.perf_counter()

    con.execute(
        f"""
        CREATE TEMP VIEW chosen_games AS
        SELECT game_id
        FROM (
            SELECT DISTINCT game_id
            FROM read_parquet('{inp}')
        )
        ORDER BY hash(CAST(game_id AS VARCHAR) || ':{int(args.seed)}')
        LIMIT {int(args.games)}
        """
    )

    profile = {}
    for field in FIELDS:
        row = con.execute(
            f"""
            SELECT
                count(*) AS rows,
                count({field}) AS non_null,
                min(CAST({field} AS DOUBLE)) AS min_value,
                max(CAST({field} AS DOUBLE)) AS max_value,
                avg(CAST({field} AS DOUBLE)) AS mean_value,
                avg(CAST(CAST({field} AS DOUBLE)=0 AS DOUBLE))
                    FILTER (WHERE {field} IS NOT NULL) AS zero_rate
            FROM read_parquet('{inp}') r
            JOIN chosen_games c USING (game_id)
            """
        ).fetchone()
        profile[field] = {
            "rows": int(row[0]),
            "non_null": int(row[1]),
            "min": finite_number(row[2]),
            "max": finite_number(row[3]),
            "mean": finite_number(row[4]),
            "zero_rate": finite_number(row[5]),
        }

    query = con.execute(
        f"""
        SELECT
            r.game_id,
            r.subframe,
            r.attack,
            r.garbage_cleared,
            r.incoming_garbage,
            r.immediate_garbage
        FROM read_parquet('{inp}') r
        JOIN chosen_games c USING (game_id)
        ORDER BY r.game_id, r.subframe
        """
    )

    agg = {
        "games": 0,
        "rows": 0,
        "pair_rows": 0,
        "immediate_le_incoming": 0,
        "incoming_le_immediate": 0,
        "equal_pair": 0,
        "incoming_to_immediate_lags": [],
        "attack_conditioned": {
            field: {
                "pre_n": 0,
                "pre_support": 0,
                "pre_capture": [],
                "post_n": 0,
                "post_support": 0,
                "post_capture": [],
            }
            for field in FIELDS
        },
    }

    current_game = None
    rows: list[tuple] = []
    while True:
        batch = query.fetchmany(65536)
        if not batch:
            break
        for row in batch:
            game_id = int(row[0])
            if current_game is None:
                current_game = game_id
            if game_id != current_game:
                process_game(
                    rows,
                    lead_window_frames=args.lead_window_frames,
                    agg=agg,
                )
                agg["games"] += 1
                agg["rows"] += len(rows)
                rows = []
                current_game = game_id
            rows.append(row[1:])

    if rows:
        process_game(
            rows,
            lead_window_frames=args.lead_window_frames,
            agg=agg,
        )
        agg["games"] += 1
        agg["rows"] += len(rows)

    pair_n = max(1, int(agg["pair_rows"]))
    nesting = {
        "rows": int(agg["pair_rows"]),
        "immediate_le_incoming_rate": (
            agg["immediate_le_incoming"] / pair_n
        ),
        "incoming_le_immediate_rate": (
            agg["incoming_le_immediate"] / pair_n
        ),
        "equal_rate": agg["equal_pair"] / pair_n,
    }

    lags = list(agg["incoming_to_immediate_lags"])
    lag_counter = Counter(lags)
    positive = [x for x in lags if x > 0]
    zero = [x for x in lags if x == 0]
    negative = [x for x in lags if x < 0]
    lag_summary = {
        "matched_events": len(lags),
        "signed_lag_frames": summarize([float(x) for x in lags]),
        "positive_lag_rate": (
            len(positive) / len(lags) if lags else None
        ),
        "zero_lag_rate": (
            len(zero) / len(lags) if lags else None
        ),
        "negative_lag_rate": (
            len(negative) / len(lags) if lags else None
        ),
        "top_lags": [
            {"lag": int(k), "n": int(v)}
            for k, v in lag_counter.most_common(12)
        ],
    }

    attack_evidence = {}
    for field, stat in agg["attack_conditioned"].items():
        pre_n = max(1, int(stat["pre_n"]))
        post_n = max(1, int(stat["post_n"]))
        pre_capture = summarize(stat["pre_capture"])
        post_capture = summarize(stat["post_capture"])
        pre_support = stat["pre_support"] / pre_n
        post_support = stat["post_support"] / post_n
        pre_mean = (
            pre_capture["mean"]
            if pre_capture["mean"] is not None
            else 0.0
        )
        post_mean = (
            post_capture["mean"]
            if post_capture["mean"] is not None
            else 0.0
        )

        if (
            stat["pre_n"] >= 1000
            and pre_support >= post_support + 0.15
            and pre_mean >= post_mean + 0.15
        ):
            timing_hint = "PRE_ACTION_CANDIDATE"
        elif (
            stat["post_n"] >= 1000
            and post_support >= pre_support + 0.15
            and post_mean >= pre_mean + 0.15
        ):
            timing_hint = "POST_ACTION_CANDIDATE"
        else:
            timing_hint = "UNRESOLVED"

        attack_evidence[field] = {
            "pre_rows": int(stat["pre_n"]),
            "pre_drop_support_rate": pre_support,
            "pre_attack_capture": pre_capture,
            "post_rows": int(stat["post_n"]),
            "post_drop_support_rate": post_support,
            "post_attack_capture": post_capture,
            "timing_hint": timing_hint,
            "warning": (
                "Opponent arrivals/tanking and unknown historical attack-field "
                "semantics confound this evidence; it is not sufficient for "
                "state-input approval."
            ),
        }

    report = {
        "format": "tetrio_historical_garbage_context_audit",
        "metadata": {
            "input": str(args.input),
            "games_requested": int(args.games),
            "games_processed": int(agg["games"]),
            "rows_processed": int(agg["rows"]),
            "seed": int(args.seed),
            "lead_window_frames": int(args.lead_window_frames),
            "seconds": time.perf_counter() - started,
        },
        "profiles": profile,
        "cross_field_nesting": nesting,
        "incoming_to_immediate_lead_lag": lag_summary,
        "attack_conditioned_timing_evidence": attack_evidence,
        "state_input_status": {
            "incoming_garbage": "BLOCKED_PENDING_REAL_REPLAY_PARITY",
            "immediate_garbage": "BLOCKED_PENDING_REAL_REPLAY_PARITY",
        },
        "why_blocked": (
            "The placement-only historical corpus has no authoritative opponent "
            "send/arrival/activation/tank event stream. Timing correlations can "
            "rank hypotheses but cannot prove exact decision-time queue semantics."
        ),
        "next_required_evidence": [
            "real .ttrm or controlled capture",
            "send -> queue arrival",
            "pre-activation cancellation",
            "activation timing",
            "tank/insertion timing",
            "packet boundary / cap behavior",
        ],
        "status": "EVIDENCE_ONLY_FAIL_CLOSED",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Rows processed : {agg['rows']:,}")
    print(f"Games processed: {agg['games']:,}")
    print()
    print("CROSS-FIELD")
    print(
        "  immediate<=incoming : "
        f"{nesting['immediate_le_incoming_rate']:.4f}"
    )
    print(
        "  incoming<=immediate : "
        f"{nesting['incoming_le_immediate_rate']:.4f}"
    )
    print(f"  equal              : {nesting['equal_rate']:.4f}")
    print()
    print("LEAD/LAG incoming rise -> nearest immediate rise")
    print(
        f"  matched={lag_summary['matched_events']:,} "
        f"positive={lag_summary['positive_lag_rate']} "
        f"zero={lag_summary['zero_lag_rate']} "
        f"negative={lag_summary['negative_lag_rate']}"
    )
    print(f"  top lags: {lag_summary['top_lags']}")
    print()
    print("ATTACK-CONDITIONED TIMING HINTS")
    for field in FIELDS:
        ev = attack_evidence[field]
        print(
            f"  {field:18s}: {ev['timing_hint']} | "
            f"PRE support={ev['pre_drop_support_rate']:.4f} "
            f"capture={ev['pre_attack_capture']['mean']} | "
            f"POST support={ev['post_drop_support_rate']:.4f} "
            f"capture={ev['post_attack_capture']['mean']}"
        )
    print()
    print("MODEL INPUT")
    print("  incoming_garbage  : BLOCKED_PENDING_REAL_REPLAY_PARITY")
    print("  immediate_garbage : BLOCKED_PENDING_REAL_REPLAY_PARITY")
    print(f"Status : {report['status']}")
    print(f"Report : {args.output}")


if __name__ == "__main__":
    main()
