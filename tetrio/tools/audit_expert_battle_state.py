from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any


AUDIT_SCHEMA_VERSION = 1
OFFSETS = (-2, -1, 0, 1, 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit timing/causality of historical TETR.IO battle-state columns "
            "before any of them are allowed into expert-model inputs."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_battle_state_audit.json"),
    )
    p.add_argument("--games", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, min(20, os.cpu_count() or 1)),
    )
    p.add_argument(
        "--timing-threshold",
        type=float,
        default=0.999,
        help=(
            "Minimum informative-row match rate to approve raw combo/B2B "
            "timing. Default 0.999."
        ),
    )
    p.add_argument(
        "--timing-margin",
        type=float,
        default=0.10,
        help=(
            "Required advantage of PRE vs POST timing (or vice versa) on "
            "informative rows. Default 0.10."
        ),
    )
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required. Install it in the project venv with:\n"
            r"  .venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def classify_timing(
    *,
    before_score: float | None,
    after_score: float | None,
    threshold: float,
    margin: float,
) -> str:
    if before_score is None or after_score is None:
        return "unresolved"
    if (
        after_score >= threshold
        and after_score - before_score >= margin
    ):
        return "post_action"
    if (
        before_score >= threshold
        and before_score - after_score >= margin
    ):
        return "pre_action"
    return "unresolved"


def _safe_rate(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def _rows_to_dicts(cur) -> list[dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _scalar(con, sql: str):
    return con.execute(sql).fetchone()[0]


def _best_offset_alignment(
    con,
    *,
    raw_col: str,
    before_col: str,
    after_col: str,
) -> dict[str, Any]:
    rows = []
    for offset in OFFSETS:
        cur = con.execute(
            f"""
            SELECT
                {int(offset)} AS offset,
                count(*) FILTER (WHERE {raw_col} IS NOT NULL) AS n_all,
                avg(CAST(
                    CAST({raw_col} AS DOUBLE)
                    = CAST({before_col} AS DOUBLE) + {int(offset)}
                    AS DOUBLE
                )) FILTER (WHERE {raw_col} IS NOT NULL) AS before_all,
                avg(CAST(
                    CAST({raw_col} AS DOUBLE)
                    = CAST({after_col} AS DOUBLE) + {int(offset)}
                    AS DOUBLE
                )) FILTER (WHERE {raw_col} IS NOT NULL) AS after_all,
                count(*) FILTER (
                    WHERE {raw_col} IS NOT NULL
                      AND {before_col} <> {after_col}
                ) AS n_informative,
                avg(CAST(
                    CAST({raw_col} AS DOUBLE)
                    = CAST({before_col} AS DOUBLE) + {int(offset)}
                    AS DOUBLE
                )) FILTER (
                    WHERE {raw_col} IS NOT NULL
                      AND {before_col} <> {after_col}
                ) AS before_informative,
                avg(CAST(
                    CAST({raw_col} AS DOUBLE)
                    = CAST({after_col} AS DOUBLE) + {int(offset)}
                    AS DOUBLE
                )) FILTER (
                    WHERE {raw_col} IS NOT NULL
                      AND {before_col} <> {after_col}
                ) AS after_informative
            FROM battle_states
            """
        )
        rows.extend(_rows_to_dicts(cur))

    def score(row: dict[str, Any], key: str) -> float:
        val = _safe_rate(row.get(key))
        return -1.0 if val is None else val

    best_before = max(rows, key=lambda r: score(r, "before_informative"))
    best_after = max(rows, key=lambda r: score(r, "after_informative"))

    return {
        "offset_trials": rows,
        "best_before": best_before,
        "best_after": best_after,
        "best_before_informative_rate": _safe_rate(
            best_before.get("before_informative")
        ),
        "best_after_informative_rate": _safe_rate(
            best_after.get("after_informative")
        ),
    }


def _field_profile(con, field: str) -> dict[str, Any]:
    cur = con.execute(
        f"""
        SELECT
            count(*) AS rows,
            count({field}) AS non_null,
            min(CAST({field} AS DOUBLE)) AS min,
            max(CAST({field} AS DOUBLE)) AS max,
            avg(CAST({field} AS DOUBLE)) AS mean,
            avg(CAST(COALESCE(CAST({field} AS DOUBLE), 0.0) = 0.0 AS DOUBLE))
                AS zero_rate,
            avg(CAST(
                {field} IS NOT NULL
                AND lag_{field} IS NOT NULL
                AND CAST({field} AS DOUBLE) <> CAST(lag_{field} AS DOUBLE)
                AS DOUBLE
            )) AS change_rate
        FROM battle_states
        """
    )
    return _rows_to_dicts(cur)[0]


def _top_values(con, field: str, limit: int = 12) -> list[dict[str, Any]]:
    cur = con.execute(
        f"""
        SELECT CAST({field} AS VARCHAR) AS value, count(*) AS n
        FROM battle_states
        GROUP BY 1
        ORDER BY n DESC, value
        LIMIT {int(limit)}
        """
    )
    return _rows_to_dicts(cur)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input Parquet not found: {args.input}")
    if args.games <= 0:
        raise SystemExit("--games must be positive")

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={max(1, int(args.threads))}")
    con.execute("SET preserve_insertion_order=false")

    inp = qpath(args.input)
    started = time.perf_counter()

    print("=" * 108)
    print("TETR.IO HISTORICAL EXPERT — BATTLE-STATE CAUSALITY AUDIT")
    print("=" * 108)
    print(f"Input       : {args.input}")
    print(f"Games       : {args.games:,}")
    print(f"Seed        : {args.seed}")
    print(f"Threads     : {args.threads}")
    print(f"Threshold   : {args.timing_threshold:.6f}")
    print(f"Margin      : {args.timing_margin:.4f}")
    print()

    # Keep the audit independent of train/val/test split. We sample complete
    # games, then reconstruct state strictly from rows that occur earlier in
    # each game.
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

    # t_spin is treated only as a historical outcome label here. The
    # difficult-clear predicate is intentionally conservative and is NOT a
    # claim that the 2024 corpus follows current Season-2 B2B Charging rules.
    con.execute(
        f"""
        CREATE TEMP TABLE audit_base AS
        SELECT
            r.*,
            row_number() OVER (
                PARTITION BY r.game_id ORDER BY r.subframe
            ) AS rn,
            lag(combo) OVER (
                PARTITION BY r.game_id ORDER BY r.subframe
            ) AS lag_combo,
            lag(btb) OVER (
                PARTITION BY r.game_id ORDER BY r.subframe
            ) AS lag_btb,
            lag(immediate_garbage) OVER (
                PARTITION BY r.game_id ORDER BY r.subframe
            ) AS lag_immediate_garbage,
            lag(incoming_garbage) OVER (
                PARTITION BY r.game_id ORDER BY r.subframe
            ) AS lag_incoming_garbage,
            (
                COALESCE(cleared, 0) = 4
                OR (
                    COALESCE(cleared, 0) > 0
                    AND lower(trim(COALESCE(CAST(t_spin AS VARCHAR), '')))
                        NOT IN ('', '0', 'false', 'n', 'no', 'none', 'null', 'normal')
                )
            ) AS difficult_now,
            (
                COALESCE(cleared, 0) > 0
                AND NOT (
                    COALESCE(cleared, 0) = 4
                    OR (
                        lower(trim(COALESCE(CAST(t_spin AS VARCHAR), '')))
                            NOT IN ('', '0', 'false', 'n', 'no', 'none', 'null', 'normal')
                    )
                )
            ) AS ordinary_clear_now
        FROM read_parquet('{inp}') r
        JOIN chosen_games c USING (game_id)
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE grouped AS
        SELECT
            *,
            sum(CASE WHEN COALESCE(cleared, 0) <= 0 THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id
                    ORDER BY subframe
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS combo_group_after,
            sum(CASE WHEN ordinary_clear_now THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id
                    ORDER BY subframe
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS difficult_group_after
        FROM audit_base
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE chains AS
        SELECT
            *,
            sum(CASE WHEN COALESCE(cleared, 0) > 0 THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id, combo_group_after
                    ORDER BY subframe
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS combo_chain_after,
            sum(CASE WHEN difficult_now THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id, difficult_group_after
                    ORDER BY subframe
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS difficult_chain_after
        FROM grouped
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE battle_states AS
        SELECT
            *,
            COALESCE(
                lag(combo_chain_after) OVER (
                    PARTITION BY game_id ORDER BY subframe
                ),
                0
            ) AS combo_chain_before,
            COALESCE(
                lag(difficult_chain_after) OVER (
                    PARTITION BY game_id ORDER BY subframe
                ),
                0
            ) AS difficult_chain_before
        FROM chains
        """
    )

    rows = int(_scalar(con, "SELECT count(*) FROM battle_states"))
    games = int(_scalar(con, "SELECT count(DISTINCT game_id) FROM battle_states"))

    combo_align = _best_offset_alignment(
        con,
        raw_col="combo",
        before_col="combo_chain_before",
        after_col="combo_chain_after",
    )
    combo_timing = classify_timing(
        before_score=combo_align["best_before_informative_rate"],
        after_score=combo_align["best_after_informative_rate"],
        threshold=float(args.timing_threshold),
        margin=float(args.timing_margin),
    )

    btb_align = _best_offset_alignment(
        con,
        raw_col="btb",
        before_col="difficult_chain_before",
        after_col="difficult_chain_after",
    )
    btb_timing = classify_timing(
        before_score=btb_align["best_before_informative_rate"],
        after_score=btb_align["best_after_informative_rate"],
        threshold=float(args.timing_threshold),
        margin=float(args.timing_margin),
    )

    # Current garbage snapshots are intentionally not auto-approved. The
    # placement-only corpus lacks opponent/timing events needed to establish
    # whether these values are exactly the decision-time queue. This audit
    # profiles them so a later board/transport parity gate can target the right
    # hypotheses without leaking them into V1.2A prematurely.
    garbage = {}
    for field in ("immediate_garbage", "incoming_garbage"):
        garbage[field] = {
            "profile": _field_profile(con, field),
            "top_values": _top_values(con, field),
            "state_input_status": "BLOCKED_PENDING_GARBAGE_TIMING_PARITY",
        }

    raw_profiles = {
        "combo": {
            "top_values": _top_values(con, "combo"),
            "timing": combo_timing,
            "alignment": combo_align,
        },
        "btb": {
            "top_values": _top_values(con, "btb"),
            "timing": btb_timing,
            "alignment": btb_align,
        },
        "t_spin": {
            "top_values": _top_values(con, "t_spin"),
            "role": "post-action outcome label; only lagged/history use is causal",
        },
    }

    def raw_state_contract(field: str, timing: str) -> dict[str, Any]:
        if timing == "pre_action":
            return {
                "approved": True,
                "source": f"row[t].{field}",
                "reason": "audit identifies raw field as pre-action on informative rows",
            }
        if timing == "post_action":
            return {
                "approved": True,
                "source": f"row[t-1].{field}",
                "reason": "audit identifies raw field as post-action; lag is causal state",
            }
        return {
            "approved": False,
            "source": None,
            "reason": "raw timing unresolved; use only reconstructed/history-safe features",
        }

    approvals = {
        "combo_chain_before_reconstructed": {
            "approved": True,
            "source": "prior cleared outcomes only",
            "semantics": (
                "number of consecutive immediately preceding placements with "
                "cleared>0; 0 means inactive"
            ),
        },
        "combo_index_before_reconstructed": {
            "approved": True,
            "source": "combo_chain_before_reconstructed",
            "semantics": "-1 inactive, first prior clear=0",
        },
        "difficult_chain_before_reconstructed": {
            "approved": True,
            "source": "prior cleared+t_spin outcomes only",
            "semantics": (
                "diagnostic difficult-clear chain; not asserted to equal "
                "historical/current TETR.IO B2B unless raw audit supports it"
            ),
        },
        "previous_cleared": {
            "approved": True,
            "source": "row[t-1].cleared",
        },
        "previous_t_spin": {
            "approved": True,
            "source": "row[t-1].t_spin",
        },
        "previous_attack": {
            "approved": True,
            "source": "row[t-1].attack",
        },
        "previous_garbage_cleared": {
            "approved": True,
            "source": "row[t-1].garbage_cleared",
        },
        "raw_combo_before": raw_state_contract("combo", combo_timing),
        "raw_btb_before": raw_state_contract("btb", btb_timing),
        "current_immediate_garbage": {
            "approved": False,
            "source": None,
            "reason": "requires independent garbage timing/transport parity",
        },
        "current_incoming_garbage": {
            "approved": False,
            "source": None,
            "reason": "requires independent garbage timing/transport parity",
        },
        "won": {
            "approved": False,
            "source": None,
            "reason": "future game result",
        },
        "current_cleared_attack_tspin": {
            "approved": False,
            "source": None,
            "reason": "current action outcome; target leakage if used as state input",
        },
    }

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "metadata": {
            "input": str(args.input),
            "sample_games_requested": int(args.games),
            "sample_games_actual": games,
            "rows": rows,
            "seed": int(args.seed),
            "threads": int(args.threads),
            "timing_threshold": float(args.timing_threshold),
            "timing_margin": float(args.timing_margin),
            "seconds": time.perf_counter() - started,
        },
        "definitions": {
            "combo_chain": (
                "normalized consecutive-clear length; no clear resets to 0"
            ),
            "difficult_clear": (
                "cleared==4 OR (cleared>0 AND t_spin label is non-empty/non-NONE)"
            ),
            "difficult_chain": (
                "difficult clear increments; ordinary line clear resets; no-clear preserves"
            ),
            "warning": (
                "difficult_chain is an audit/reconstruction feature for the "
                "historical corpus, not a claim that the 2024 data follows "
                "current Season-2 B2B Charging semantics"
            ),
        },
        "raw_profiles": raw_profiles,
        "garbage": garbage,
        "approvals": approvals,
        "overall_status": "PASS_CAUSAL_HISTORY_WITH_FAIL_CLOSED_GARBAGE",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows audited : {rows:,}")
    print(f"Games audited: {games:,}")
    print()
    print("RAW COMBO")
    print(
        f"  best PRE informative : "
        f"{combo_align['best_before_informative_rate']}"
    )
    print(
        f"  best POST informative: "
        f"{combo_align['best_after_informative_rate']}"
    )
    print(f"  timing               : {combo_timing}")
    print(
        f"  raw state input      : "
        f"{'APPROVED' if approvals['raw_combo_before']['approved'] else 'BLOCKED'}"
    )
    print()
    print("RAW B2B")
    print(
        f"  best PRE informative : "
        f"{btb_align['best_before_informative_rate']}"
    )
    print(
        f"  best POST informative: "
        f"{btb_align['best_after_informative_rate']}"
    )
    print(f"  timing               : {btb_timing}")
    print(
        f"  raw state input      : "
        f"{'APPROVED' if approvals['raw_btb_before']['approved'] else 'BLOCKED'}"
    )
    print()
    print("ALWAYS CAUSAL")
    print("  combo_chain_before_reconstructed : APPROVED")
    print("  previous_cleared                 : APPROVED")
    print("  previous_t_spin                  : APPROVED")
    print("  previous_attack                  : APPROVED")
    print("  previous_garbage_cleared         : APPROVED")
    print()
    print("FAIL-CLOSED")
    print("  current incoming_garbage         : BLOCKED")
    print("  current immediate_garbage        : BLOCKED")
    print("  won/current cleared/attack/spin  : BLOCKED")
    print()
    print(f"Status : {report['overall_status']}")
    print(f"Report : {args.output}")
    print("=" * 108)


if __name__ == "__main__":
    main()
