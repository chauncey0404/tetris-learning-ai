from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build leakage-safe causal battle-history sidecars for the expert "
            "train/val/test corpus. Raw combo/B2B are included only if the "
            "audit explicitly approved their timing."
        )
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument(
        "--expert-dir",
        type=Path,
        default=Path(r"data\tetrio\expert"),
    )
    p.add_argument(
        "--audit",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_battle_state_audit.json"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"data\tetrio\expert_stateful"),
    )
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
            "DuckDB is required. Install it with:\n"
            r"  .venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _raw_expr(
    *,
    audit: dict,
    approval_key: str,
    current_col: str,
    lag_col: str,
) -> tuple[str, str]:
    item = audit["approvals"][approval_key]
    if not bool(item.get("approved")):
        return "NULL::BIGINT", "blocked"

    source = str(item.get("source") or "")
    if source.startswith("row[t]."):
        return f"CAST({current_col} AS BIGINT)", "current"
    if source.startswith("row[t-1]."):
        return f"CAST({lag_col} AS BIGINT)", "lag"
    raise SystemExit(
        f"Unsupported approved source for {approval_key}: {source!r}"
    )


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"Source Parquet not found: {args.source}")
    if not args.audit.is_file():
        raise SystemExit(
            f"Audit report not found: {args.audit}\n"
            "Run tetrio.tools.audit_expert_battle_state first."
        )

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if int(audit.get("schema_version", -1)) != 1:
        raise SystemExit(
            f"Unsupported audit schema: {audit.get('schema_version')!r}"
        )

    combo_expr, combo_source = _raw_expr(
        audit=audit,
        approval_key="raw_combo_before",
        current_col="combo",
        lag_col="lag_combo",
    )
    btb_expr, btb_source = _raw_expr(
        audit=audit,
        approval_key="raw_btb_before",
        current_col="btb",
        lag_col="lag_btb",
    )

    split_files = {
        "train": args.expert_dir / "top_players_s1_train.parquet",
        "val": args.expert_dir / "top_players_s1_val.parquet",
        "test": args.expert_dir / "top_players_s1_test.parquet",
    }
    for name, path in split_files.items():
        if not path.is_file():
            raise SystemExit(f"Expert {name} split not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        name: args.output_dir / f"top_players_s1_{name}_battle_state.parquet"
        for name in split_files
    }
    manifest_path = args.output_dir / "top_players_s1_battle_state_manifest.json"

    existing = [p for p in [*outputs.values(), manifest_path] if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Output already exists:\n  "
            + "\n  ".join(str(p) for p in existing)
            + "\nUse --overwrite to replace it."
        )
    if args.overwrite:
        for p in existing:
            p.unlink()

    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads={max(1, int(args.threads))}")
    con.execute("SET preserve_insertion_order=true")

    src = qpath(args.source)
    started = time.perf_counter()

    con.execute(
        f"""
        CREATE TEMP TABLE source_indexed AS
        SELECT
            *,
            row_number() OVER () AS source_row_id
        FROM read_parquet('{src}')
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE history_base AS
        SELECT
            *,
            lag(combo) OVER (
                PARTITION BY game_id ORDER BY subframe, source_row_id
            ) AS lag_combo,
            lag(btb) OVER (
                PARTITION BY game_id ORDER BY subframe, source_row_id
            ) AS lag_btb,
            lag(cleared) OVER (
                PARTITION BY game_id ORDER BY subframe, source_row_id
            ) AS previous_cleared,
            lag(t_spin) OVER (
                PARTITION BY game_id ORDER BY subframe, source_row_id
            ) AS previous_t_spin,
            lag(attack) OVER (
                PARTITION BY game_id ORDER BY subframe, source_row_id
            ) AS previous_attack,
            lag(garbage_cleared) OVER (
                PARTITION BY game_id ORDER BY subframe, source_row_id
            ) AS previous_garbage_cleared,
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
        FROM source_indexed
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE history_grouped AS
        SELECT
            *,
            sum(CASE WHEN COALESCE(cleared, 0) <= 0 THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id
                    ORDER BY subframe, source_row_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS combo_group_after,
            sum(CASE WHEN ordinary_clear_now THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id
                    ORDER BY subframe, source_row_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS difficult_group_after
        FROM history_base
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE history_chains AS
        SELECT
            *,
            sum(CASE WHEN COALESCE(cleared, 0) > 0 THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id, combo_group_after
                    ORDER BY subframe, source_row_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS combo_chain_after,
            sum(CASE WHEN difficult_now THEN 1 ELSE 0 END)
                OVER (
                    PARTITION BY game_id, difficult_group_after
                    ORDER BY subframe, source_row_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS difficult_chain_after
        FROM history_grouped
        """
    )

    con.execute(
        f"""
        CREATE TEMP VIEW causal_state AS
        SELECT
            source_row_id,
            game_id,
            subframe,
            CAST(COALESCE(
                lag(combo_chain_after) OVER (
                    PARTITION BY game_id ORDER BY subframe, source_row_id
                ),
                0
            ) AS INTEGER) AS combo_chain_before,
            CAST(
                CASE
                    WHEN COALESCE(
                        lag(combo_chain_after) OVER (
                            PARTITION BY game_id ORDER BY subframe, source_row_id
                        ),
                        0
                    ) > 0
                    THEN COALESCE(
                        lag(combo_chain_after) OVER (
                            PARTITION BY game_id ORDER BY subframe, source_row_id
                        ),
                        0
                    ) - 1
                    ELSE -1
                END
                AS INTEGER
            ) AS combo_index_before,
            CAST(COALESCE(
                lag(difficult_chain_after) OVER (
                    PARTITION BY game_id ORDER BY subframe, source_row_id
                ),
                0
            ) AS INTEGER) AS difficult_chain_before,
            CAST(
                COALESCE(
                    lag(difficult_chain_after) OVER (
                        PARTITION BY game_id ORDER BY subframe, source_row_id
                    ),
                    0
                ) > 0
                AS BOOLEAN
            ) AS difficult_active_before,

            CAST(COALESCE(previous_cleared, 0) AS INTEGER)
                AS previous_cleared,
            COALESCE(CAST(previous_t_spin AS VARCHAR), 'NONE')
                AS previous_t_spin,
            CAST(COALESCE(previous_attack, 0.0) AS DOUBLE)
                AS previous_attack,
            CAST(COALESCE(previous_garbage_cleared, 0) AS INTEGER)
                AS previous_garbage_cleared,

            {combo_expr} AS raw_combo_before,
            {btb_expr} AS raw_btb_before
        FROM history_chains
        """
    )

    source_columns = [
        str(row[0])
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{src}')"
        ).fetchall()
    ]

    stats = {}
    for split, expert_path in split_files.items():
        out_path = outputs[split]
        expert_qpath = qpath(expert_path)
        expert_columns = {
            str(row[0])
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{expert_qpath}')"
            ).fetchall()
        }
        identity_columns = [
            c for c in source_columns if c in expert_columns
        ]
        if "game_id" not in identity_columns or "subframe" not in identity_columns:
            raise SystemExit(
                f"{split}: expert split lacks game_id/subframe identity columns"
            )
        if len(identity_columns) < 3:
            raise SystemExit(
                f"{split}: too few common columns for exact row alignment: "
                f"{identity_columns}"
            )

        hash_args = ", ".join(qident(c) for c in identity_columns)
        identity_hash = f"hash({hash_args})"

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE source_match AS
            SELECT
                source_row_id,
                {identity_hash} AS identity_hash,
                row_number() OVER (
                    PARTITION BY {identity_hash}
                    ORDER BY source_row_id
                ) AS identity_occurrence
            FROM source_indexed
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE expert_indexed AS
            SELECT
                *,
                {identity_hash} AS identity_hash,
                row_number() OVER () AS expert_row_id
            FROM read_parquet('{expert_qpath}')
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE expert_match AS
            SELECT
                expert_row_id,
                identity_hash,
                row_number() OVER (
                    PARTITION BY identity_hash
                    ORDER BY expert_row_id
                ) AS identity_occurrence
            FROM expert_indexed
            """
        )

        mapped_rows = int(
            con.execute(
                """
                SELECT count(*)
                FROM expert_match e
                JOIN source_match m
                  USING (identity_hash, identity_occurrence)
                """
            ).fetchone()[0]
        )

        con.execute(
            f"""
            COPY (
                SELECT
                    s.* EXCLUDE (source_row_id)
                FROM expert_match e
                JOIN source_match m
                  USING (identity_hash, identity_occurrence)
                JOIN causal_state s
                  USING (source_row_id)
                ORDER BY e.expert_row_id
            )
            TO '{qpath(out_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
            """
        )

        expert_rows = int(
            con.execute(
                f"SELECT count(*) FROM read_parquet('{expert_qpath}')"
            ).fetchone()[0]
        )
        output_rows = int(
            con.execute(
                f"SELECT count(*) FROM read_parquet('{qpath(out_path)}')"
            ).fetchone()[0]
        )
        duplicate_key_groups = int(
            con.execute(
                f"""
                SELECT count(*)
                FROM (
                    SELECT game_id, subframe
                    FROM read_parquet('{expert_qpath}')
                    GROUP BY game_id, subframe
                    HAVING count(*) > 1
                )
                """
            ).fetchone()[0]
        )

        stats[split] = {
            "expert_rows": expert_rows,
            "mapped_rows": mapped_rows,
            "sidecar_rows": output_rows,
            "row_count_match": (
                expert_rows == mapped_rows == output_rows
            ),
            "identity_column_count": len(identity_columns),
            "duplicate_game_subframe_groups": duplicate_key_groups,
            "bytes": out_path.stat().st_size,
            "path": str(out_path),
        }

    status = (
        "PASS"
        if all(x["row_count_match"] for x in stats.values())
        else "FAIL"
    )

    manifest = {
        "schema_version": 1,
        "source": str(args.source),
        "audit": str(args.audit),
        "raw_combo_source": combo_source,
        "raw_btb_source": btb_source,
        "alignment_contract": {
            "method": "common-column fingerprint + duplicate occurrence",
            "reason": (
                "game_id+subframe is not unique for every historical row; "
                "joining only on those two fields can create many-to-many "
                "row inflation"
            ),
            "output_order": "exact expert split row order",
        },
        "approved_state_columns": [
            "combo_chain_before",
            "combo_index_before",
            "difficult_chain_before",
            "difficult_active_before",
            "previous_cleared",
            "previous_t_spin",
            "previous_attack",
            "previous_garbage_cleared",
        ]
        + (["raw_combo_before"] if combo_source != "blocked" else [])
        + (["raw_btb_before"] if btb_source != "blocked" else []),
        "explicitly_excluded_from_state": [
            "won",
            "current cleared",
            "current attack",
            "current t_spin",
            "current immediate_garbage",
            "current incoming_garbage",
        ],
        "splits": stats,
        "seconds": time.perf_counter() - started,
        "status": status,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 108)
    print("TETR.IO EXPERT CAUSAL BATTLE-STATE SIDECAR")
    print("=" * 108)
    print(f"Raw combo source: {combo_source}")
    print(f"Raw B2B source  : {btb_source}")
    print()
    for split, item in stats.items():
        print(
            f"{split.upper():5s}: "
            f"{item['sidecar_rows']:,}/{item['expert_rows']:,} rows "
            f"{'PASS' if item['row_count_match'] else 'FAIL'} "
            f"mapped={item['mapped_rows']:,} "
            f"duplicate_keys={item['duplicate_game_subframe_groups']:,}"
        )
    print()
    print(f"Status  : {status}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
