from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


REQUIRED = {
    "game_id",
    "subframe",
    "attack",
    "garbage_cleared",
    "incoming_garbage",
    "immediate_garbage",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full-corpus causal timing audit for historical incoming_garbage. "
            "Uses current action garbage_cleared as an outcome-only diagnostic "
            "and compares PRE-action (row[t] incoming) against shifted POST-action "
            "(row[t-1] incoming) hypotheses."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument("--threads", type=int, default=max(1, min(20, os.cpu_count() or 1)))
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"artifacts\tetrio\historical_incoming_causality_audit.json"),
    )
    p.add_argument(
        "--candidate-output",
        type=Path,
        default=Path(r"artifacts\tetrio\historical_clean_cancellation_candidates.json"),
    )
    p.add_argument("--candidate-limit", type=int, default=100)
    p.add_argument(
        "--min-cancel-rows",
        type=int,
        default=10000,
        help="Minimum positive garbage_cleared rows required for a strong verdict.",
    )
    p.add_argument(
        "--max-current-capacity-violation-rate",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--min-prev-minus-current-violation-gap",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--min-current-vs-prev-exact-drop-gap",
        type=float,
        default=0.10,
    )
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required. Install with:\n"
            r"  .venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rate(num: int | float, den: int | float) -> float | None:
    return float(num) / float(den) if den else None


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={int(args.threads)}")
    con.execute("SET preserve_insertion_order=true")
    inp = qpath(args.input)

    columns = {
        str(r[0])
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{inp}')"
        ).fetchall()
    }
    missing = sorted(REQUIRED - columns)
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    optional = [
        c for c in ("cleared", "combo", "btb", "t_spin", "won")
        if c in columns
    ]

    print("=" * 116)
    print("TETR.IO HISTORICAL INCOMING_GARBAGE — FULL-CORPUS CAUSAL TIMING AUDIT")
    print("=" * 116)
    print(f"Input   : {args.input}")
    print(f"Threads : {args.threads}")
    print(
        "Core idea: current garbage_cleared[t] is an action outcome. "
        "It may diagnose timing, but is NEVER admitted as a pre-action model input."
    )
    print()

    # Source insertion row id is a deterministic tie-breaker for the handful of
    # duplicate (game_id, subframe) keys already discovered by the sidecar audit.
    con.execute(
        f"""
        CREATE TEMP TABLE ordered AS
        SELECT
            row_number() OVER () AS source_row_id,
            *
        FROM read_parquet('{inp}')
        """
    )

    select_optional = "".join(f", {c}" for c in optional)
    con.execute(
        f"""
        CREATE TEMP TABLE causal AS
        SELECT
            source_row_id,
            game_id,
            subframe,
            CAST(attack AS DOUBLE) AS attack,
            CAST(garbage_cleared AS DOUBLE) AS garbage_cleared,
            CAST(incoming_garbage AS DOUBLE) AS incoming,
            CAST(immediate_garbage AS DOUBLE) AS immediate
            {select_optional},
            LAG(CAST(incoming_garbage AS DOUBLE))
                OVER (PARTITION BY game_id ORDER BY subframe, source_row_id)
                AS prev_incoming,
            LEAD(CAST(incoming_garbage AS DOUBLE))
                OVER (PARTITION BY game_id ORDER BY subframe, source_row_id)
                AS next_incoming,
            LAG(CAST(immediate_garbage AS DOUBLE))
                OVER (PARTITION BY game_id ORDER BY subframe, source_row_id)
                AS prev_immediate,
            LEAD(CAST(immediate_garbage AS DOUBLE))
                OVER (PARTITION BY game_id ORDER BY subframe, source_row_id)
                AS next_immediate,
            LAG(subframe)
                OVER (PARTITION BY game_id ORDER BY subframe, source_row_id)
                AS prev_subframe,
            LEAD(subframe)
                OVER (PARTITION BY game_id ORDER BY subframe, source_row_id)
                AS next_subframe
        FROM ordered
        """
    )

    row = con.execute(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT game_id) AS games,
            count(*) FILTER (WHERE garbage_cleared > 0) AS cancel_rows,
            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND incoming IS NOT NULL
            ) AS cancel_current_rows,
            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND prev_incoming IS NOT NULL
            ) AS cancel_prev_rows,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND incoming IS NOT NULL
                  AND garbage_cleared > incoming + 1e-9
            ) AS current_capacity_violations,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND prev_incoming IS NOT NULL
                  AND garbage_cleared > prev_incoming + 1e-9
            ) AS prev_capacity_violations,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND next_incoming IS NOT NULL
                  AND abs(
                    greatest(0.0, incoming - garbage_cleared)
                    - next_incoming
                  ) < 1e-9
            ) AS pre_exact_next,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND prev_incoming IS NOT NULL
                  AND abs(
                    greatest(0.0, prev_incoming - garbage_cleared)
                    - incoming
                  ) < 1e-9
            ) AS post_exact_current,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND next_incoming IS NOT NULL
            ) AS pre_exact_den,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND prev_incoming IS NOT NULL
            ) AS post_exact_den,

            avg(
                abs(greatest(0.0, incoming - garbage_cleared) - next_incoming)
            ) FILTER (
                WHERE garbage_cleared > 0
                  AND next_incoming IS NOT NULL
            ) AS pre_mae,

            avg(
                abs(greatest(0.0, prev_incoming - garbage_cleared) - incoming)
            ) FILTER (
                WHERE garbage_cleared > 0
                  AND prev_incoming IS NOT NULL
            ) AS post_mae,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND attack IS NOT NULL
                  AND garbage_cleared <= attack + 1e-9
            ) AS cancel_le_attack,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND attack IS NOT NULL
            ) AS cancel_attack_den,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND incoming > 0
                  AND attack > 0
                  AND next_incoming IS NOT NULL
                  AND next_incoming < incoming
            ) AS positive_cancel_with_next_drop,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND incoming > 0
                  AND attack > 0
                  AND prev_incoming IS NOT NULL
                  AND incoming < prev_incoming
            ) AS positive_cancel_with_current_drop,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND incoming > 0
                  AND attack > 0
                  AND next_incoming IS NOT NULL
            ) AS positive_cancel_next_den,

            count(*) FILTER (
                WHERE garbage_cleared > 0
                  AND incoming > 0
                  AND attack > 0
                  AND prev_incoming IS NOT NULL
            ) AS positive_cancel_prev_den
        FROM causal
        """
    ).fetchone()

    keys = [
        "rows","games","cancel_rows","cancel_current_rows","cancel_prev_rows",
        "current_capacity_violations","prev_capacity_violations",
        "pre_exact_next","post_exact_current","pre_exact_den","post_exact_den",
        "pre_mae","post_mae","cancel_le_attack","cancel_attack_den",
        "positive_cancel_with_next_drop","positive_cancel_with_current_drop",
        "positive_cancel_next_den","positive_cancel_prev_den",
    ]
    s = dict(zip(keys, row))

    current_violation_rate = rate(
        int(s["current_capacity_violations"]),
        int(s["cancel_current_rows"]),
    )
    prev_violation_rate = rate(
        int(s["prev_capacity_violations"]),
        int(s["cancel_prev_rows"]),
    )
    pre_exact_rate = rate(int(s["pre_exact_next"]), int(s["pre_exact_den"]))
    post_exact_rate = rate(int(s["post_exact_current"]), int(s["post_exact_den"]))
    next_drop_rate = rate(
        int(s["positive_cancel_with_next_drop"]),
        int(s["positive_cancel_next_den"]),
    )
    current_drop_rate = rate(
        int(s["positive_cancel_with_current_drop"]),
        int(s["positive_cancel_prev_den"]),
    )
    cancel_le_attack_rate = rate(
        int(s["cancel_le_attack"]),
        int(s["cancel_attack_den"]),
    )

    violation_gap = (
        None
        if current_violation_rate is None or prev_violation_rate is None
        else prev_violation_rate - current_violation_rate
    )
    exact_gap = (
        None
        if pre_exact_rate is None or post_exact_rate is None
        else pre_exact_rate - post_exact_rate
    )

    strong = (
        int(s["cancel_rows"]) >= args.min_cancel_rows
        and current_violation_rate is not None
        and current_violation_rate <= args.max_current_capacity_violation_rate
        and violation_gap is not None
        and violation_gap >= args.min_prev_minus_current_violation_gap
        and exact_gap is not None
        and exact_gap >= args.min_current_vs_prev_exact_drop_gap
    )

    if strong:
        status = "STRONG_HISTORICAL_PRE_ACTION_EVIDENCE"
    elif (
        current_violation_rate is not None
        and prev_violation_rate is not None
        and current_violation_rate < prev_violation_rate
        and pre_exact_rate is not None
        and post_exact_rate is not None
        and pre_exact_rate > post_exact_rate
    ):
        status = "PRE_ACTION_EVIDENCE_BELOW_STRONG_GATE"
    else:
        status = "TIMING_UNRESOLVED"

    summary = {
        "rows": int(s["rows"]),
        "games": int(s["games"]),
        "positive_garbage_cleared_rows": int(s["cancel_rows"]),
        "capacity_bound": {
            "current_incoming_rows": int(s["cancel_current_rows"]),
            "current_violations": int(s["current_capacity_violations"]),
            "current_violation_rate": current_violation_rate,
            "previous_incoming_rows": int(s["cancel_prev_rows"]),
            "previous_violations": int(s["prev_capacity_violations"]),
            "previous_violation_rate": prev_violation_rate,
            "previous_minus_current_violation_rate": violation_gap,
            "interpretation": (
                "If incoming[t] is pre-action, garbage_cleared[t] should almost "
                "never exceed incoming[t]. The shifted incoming[t-1] should be a "
                "worse capacity bound when new garbage arrives between placements."
            ),
        },
        "queue_transition_exactness": {
            "pre_hypothesis": "next_incoming == max(0, incoming - garbage_cleared)",
            "pre_rows": int(s["pre_exact_den"]),
            "pre_exact": int(s["pre_exact_next"]),
            "pre_exact_rate": pre_exact_rate,
            "pre_mae": finite(s["pre_mae"]),
            "post_hypothesis": "incoming == max(0, prev_incoming - garbage_cleared)",
            "post_rows": int(s["post_exact_den"]),
            "post_exact": int(s["post_exact_current"]),
            "post_exact_rate": post_exact_rate,
            "post_mae": finite(s["post_mae"]),
            "pre_minus_post_exact_rate": exact_gap,
            "warning": (
                "Neither equation is expected to be universally exact because "
                "opponent arrivals, queue maturation, and tanking may occur between rows."
            ),
        },
        "action_outcome_consistency": {
            "garbage_cleared_le_attack_rate": cancel_le_attack_rate,
            "next_drop_on_positive_cancel_rate": next_drop_rate,
            "current_drop_on_positive_cancel_rate": current_drop_rate,
        },
    }

    # Natural cases: positive current cancellation, exact current capacity,
    # and a next-row drop consistent with the current-row cancellation amount.
    optional_select = "".join(f", {c}" for c in optional)
    candidates = con.execute(
        f"""
        SELECT
            game_id,
            subframe,
            source_row_id,
            prev_incoming,
            incoming,
            next_incoming,
            prev_immediate,
            immediate,
            next_immediate,
            attack,
            garbage_cleared
            {optional_select},
            abs(
                greatest(0.0, incoming - garbage_cleared)
                - next_incoming
            ) AS pre_residual_error,
            abs(
                greatest(0.0, prev_incoming - garbage_cleared)
                - incoming
            ) AS post_residual_error
        FROM causal
        WHERE garbage_cleared > 0
          AND attack > 0
          AND incoming > 0
          AND next_incoming IS NOT NULL
          AND garbage_cleared <= incoming + 1e-9
          AND abs(
              greatest(0.0, incoming - garbage_cleared)
              - next_incoming
          ) < 1e-9
        ORDER BY
            garbage_cleared ASC,
            incoming ASC,
            attack ASC,
            game_id,
            subframe,
            source_row_id
        LIMIT {int(args.candidate_limit)}
        """
    ).fetchall()

    cand_cols = [
        "game_id","subframe","source_row_id",
        "prev_incoming","incoming","next_incoming",
        "prev_immediate","immediate","next_immediate",
        "attack","garbage_cleared",
        *optional,
        "pre_residual_error","post_residual_error",
    ]
    candidate_rows = [
        {k: v for k, v in zip(cand_cols, row)}
        for row in candidates
    ]

    report = {
        "format": "tetrio_historical_incoming_causality_audit",
        "input": str(args.input),
        "status": status,
        "gates": {
            "min_cancel_rows": args.min_cancel_rows,
            "max_current_capacity_violation_rate": args.max_current_capacity_violation_rate,
            "min_prev_minus_current_violation_gap": args.min_prev_minus_current_violation_gap,
            "min_current_vs_prev_exact_drop_gap": args.min_current_vs_prev_exact_drop_gap,
        },
        "summary": summary,
        "historical_model_input_interpretation": (
            "APPROVE_PRE_ACTION_FOR_HISTORICAL_CORPUS"
            if strong
            else "KEEP_BLOCKED"
        ),
        "cross_season_interpretation": (
            "This audit proves only the historical corpus timing if it passes. "
            "It does not claim current Season 2 queue semantics."
        ),
        "current_action_outcome_fields": {
            "garbage_cleared": "DIAGNOSTIC_ONLY_NOT_MODEL_INPUT",
            "attack": "DIAGNOSTIC_ONLY_NOT_MODEL_INPUT",
        },
        "candidate_file": str(args.candidate_output),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    args.candidate_output.write_text(
        json.dumps(
            {
                "format": "tetrio_historical_clean_cancellation_candidates",
                "input": str(args.input),
                "count": len(candidate_rows),
                "definition": (
                    "garbage_cleared>0, attack>0, incoming>0, "
                    "garbage_cleared<=incoming, and next_incoming exactly equals "
                    "max(0, incoming-garbage_cleared)"
                ),
                "rows": candidate_rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"Rows                  : {int(s['rows']):,}")
    print(f"Games                 : {int(s['games']):,}")
    print(f"Positive cancel rows  : {int(s['cancel_rows']):,}")
    print()
    print("CAPACITY BOUND (garbage_cleared cannot exceed pre-action queue)")
    print(
        f"  CURRENT incoming[t] : violations={int(s['current_capacity_violations']):,}/"
        f"{int(s['cancel_current_rows']):,} rate={current_violation_rate}"
    )
    print(
        f"  PREVIOUS incoming   : violations={int(s['prev_capacity_violations']):,}/"
        f"{int(s['cancel_prev_rows']):,} rate={prev_violation_rate}"
    )
    print(f"  prev-current gap    : {violation_gap}")
    print()
    print("QUEUE TRANSITION EXACTNESS")
    print(
        f"  PRE  t -> t+1       : {int(s['pre_exact_next']):,}/"
        f"{int(s['pre_exact_den']):,} rate={pre_exact_rate} MAE={finite(s['pre_mae'])}"
    )
    print(
        f"  POST t-1 -> t       : {int(s['post_exact_current']):,}/"
        f"{int(s['post_exact_den']):,} rate={post_exact_rate} MAE={finite(s['post_mae'])}"
    )
    print(f"  PRE-POST exact gap  : {exact_gap}")
    print()
    print("ACTION-OUTCOME CONSISTENCY")
    print(f"  garbage_cleared<=attack : {cancel_le_attack_rate}")
    print(f"  next-row drop            : {next_drop_rate}")
    print(f"  current-row drop         : {current_drop_rate}")
    print()
    print(f"Clean natural cases    : {len(candidate_rows):,}")
    print(f"Status                 : {status}")
    print(
        "Historical input       : "
        + (
            "APPROVE_PRE_ACTION_FOR_HISTORICAL_CORPUS"
            if strong
            else "KEEP_BLOCKED"
        )
    )
    print("Current Season 2       : NOT CLAIMED")
    print(f"Report                 : {args.output}")
    print(f"Candidates             : {args.candidate_output}")


if __name__ == "__main__":
    main()
