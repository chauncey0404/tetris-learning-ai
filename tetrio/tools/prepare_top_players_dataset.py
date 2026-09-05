from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

EXPECTED_COLUMNS = [
    "game_id",
    "subframe",
    "won",
    "playfield",
    "x",
    "y",
    "r",
    "placed",
    "hold",
    "next",
    "cleared",
    "garbage_cleared",
    "attack",
    "t_spin",
    "btb",
    "combo",
    "immediate_garbage",
    "incoming_garbage",
    "rating",
    "glicko",
    "glicko_rd",
]

DUCKDB_TYPES = {
    "game_id": "BIGINT",
    "subframe": "BIGINT",
    "won": "BIGINT",
    "playfield": "VARCHAR",
    "x": "BIGINT",
    "y": "BIGINT",
    "r": "VARCHAR",
    "placed": "VARCHAR",
    "hold": "VARCHAR",
    "next": "VARCHAR",
    "cleared": "BIGINT",
    "garbage_cleared": "BIGINT",
    "attack": "DOUBLE",
    "t_spin": "VARCHAR",
    "btb": "BIGINT",
    "combo": "BIGINT",
    "immediate_garbage": "DOUBLE",
    "incoming_garbage": "DOUBLE",
    "rating": "DOUBLE",
    "glicko": "DOUBLE",
    "glicko_rd": "DOUBLE",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Profile the Kaggle TETR.IO top-player placement dataset and optionally "
            "convert it to ZSTD-compressed Parquet using DuckDB's multi-threaded CSV scanner."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\raw\data.csv"),
        help=r"Raw CSV path. Default: data\tetrio\raw\data.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"data\tetrio\processed"),
        help=r"Output directory. Default: data\tetrio\processed",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="DuckDB worker threads. Default: all logical CPUs.",
    )
    p.add_argument(
        "--profile-only",
        action="store_true",
        help="Only generate the JSON profile; do not write Parquet.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing Parquet output.",
    )
    p.add_argument(
        "--parquet-name",
        default="top_players_s1.parquet",
        help="Parquet filename inside --output-dir.",
    )
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required for this data-prep tool.\n"
            "Install it in the project venv with:\n"
            r"  .venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def validate_header(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    if header != EXPECTED_COLUMNS:
        raise SystemExit(
            "Dataset header does not match the expected Kaggle schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual  : {header}"
        )


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def columns_sql() -> str:
    pairs = ", ".join(f"'{name}': '{DUCKDB_TYPES[name]}'" for name in EXPECTED_COLUMNS)
    return "{" + pairs + "}"


def rows_to_dicts(cursor) -> list[dict[str, Any]]:
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def one_row(cursor) -> dict[str, Any]:
    rows = rows_to_dicts(cursor)
    return rows[0] if rows else {}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [jsonable(x) for x in value]
    if isinstance(value, list):
        return [jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    validate_header(input_path)
    duckdb = require_duckdb()

    threads = max(1, int(args.threads))
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {threads}")
    con.execute("SET preserve_insertion_order = true")

    csv_path = qpath(input_path)
    schema = columns_sql()

    con.execute(
        f"""
        CREATE VIEW raw AS
        SELECT *
        FROM read_csv(
            '{csv_path}',
            header = true,
            auto_detect = false,
            columns = {schema},
            strict_mode = true
        )
        """
    )

    print("=" * 96)
    print("TETR.IO TOP-PLAYER DATASET PROFILE")
    print("=" * 96)
    print(f"Input   : {input_path}")
    print(f"Size    : {input_path.stat().st_size / (1024**3):.3f} GiB")
    print(f"Threads : {threads}")
    print()

    started = time.perf_counter()

    overview = one_row(
        con.execute(
            """
            SELECT
                count(*) AS rows,
                count(DISTINCT game_id) AS games,
                min(game_id) AS min_game_id,
                max(game_id) AS max_game_id,
                min(subframe) AS min_subframe,
                max(subframe) AS max_subframe,
                min(x) AS min_x,
                max(x) AS max_x,
                min(y) AS min_y,
                max(y) AS max_y,
                min(cleared) AS min_cleared,
                max(cleared) AS max_cleared,
                min(garbage_cleared) AS min_garbage_cleared,
                max(garbage_cleared) AS max_garbage_cleared,
                min(attack) AS min_attack,
                max(attack) AS max_attack,
                min(btb) AS min_btb,
                max(btb) AS max_btb,
                min(combo) AS min_combo,
                max(combo) AS max_combo,
                min(immediate_garbage) AS min_immediate_garbage,
                max(immediate_garbage) AS max_immediate_garbage,
                min(incoming_garbage) AS min_incoming_garbage,
                max(incoming_garbage) AS max_incoming_garbage,
                min(length(playfield)) AS min_playfield_len,
                max(length(playfield)) AS max_playfield_len,
                avg(length(playfield)) AS avg_playfield_len,
                min(length(next)) AS min_next_len,
                max(length(next)) AS max_next_len,
                avg(length(next)) AS avg_next_len
            FROM raw
            """
        )
    )

    per_game = one_row(
        con.execute(
            """
            SELECT
                min(n) AS min_rows_per_game,
                max(n) AS max_rows_per_game,
                avg(n) AS avg_rows_per_game,
                median(n) AS median_rows_per_game
            FROM (
                SELECT game_id, count(*) AS n
                FROM raw
                GROUP BY game_id
            )
            """
        )
    )

    rating = one_row(
        con.execute(
            """
            SELECT
                min(rating) AS min_rating,
                avg(rating) AS avg_rating,
                median(rating) AS median_rating,
                max(rating) AS max_rating,
                quantile_cont(rating, 0.05) AS p05_rating,
                quantile_cont(rating, 0.25) AS p25_rating,
                quantile_cont(rating, 0.75) AS p75_rating,
                quantile_cont(rating, 0.95) AS p95_rating,
                min(glicko) AS min_glicko,
                avg(glicko) AS avg_glicko,
                max(glicko) AS max_glicko,
                min(glicko_rd) AS min_glicko_rd,
                avg(glicko_rd) AS avg_glicko_rd,
                max(glicko_rd) AS max_glicko_rd
            FROM raw
            """
        )
    )

    distributions: dict[str, list[dict[str, Any]]] = {}
    for col in ("won", "r", "placed", "hold", "t_spin", "cleared"):
        distributions[col] = rows_to_dicts(
            con.execute(
                f"""
                SELECT {col} AS value, count(*) AS rows
                FROM raw
                GROUP BY {col}
                ORDER BY rows DESC, value
                """
            )
        )

    null_counts = one_row(
        con.execute(
            "SELECT "
            + ", ".join(
                f"sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}"
                for c in EXPECTED_COLUMNS
            )
            + " FROM raw"
        )
    )

    # Cheap ordering/continuity diagnostics. This does not assume semantics;
    # it only checks whether rows are grouped and time-ordered by game_id.
    ordering = one_row(
        con.execute(
            """
            WITH x AS (
                SELECT
                    game_id,
                    subframe,
                    lag(game_id) OVER () AS prev_game_id,
                    lag(subframe) OVER () AS prev_subframe
                FROM raw
            )
            SELECT
                sum(
                    CASE
                        WHEN prev_game_id = game_id AND subframe < prev_subframe
                        THEN 1 ELSE 0
                    END
                ) AS within_game_subframe_regressions
            FROM x
            """
        )
    )

    elapsed_profile = time.perf_counter() - started
    profile = {
        "source": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "schema": EXPECTED_COLUMNS,
        },
        "execution": {
            "threads": threads,
            "profile_seconds": elapsed_profile,
        },
        "overview": overview,
        "per_game": per_game,
        "rating": rating,
        "distributions": distributions,
        "null_counts": null_counts,
        "ordering_diagnostics": ordering,
        "usage_notes": {
            "source_era": "Treat as historical expert-play data unless independently verified otherwise.",
            "input_candidate_fields_to_review": [
                "playfield",
                "hold",
                "next",
                "incoming_garbage",
                "immediate_garbage",
                "btb",
                "combo",
            ],
            "action_label_fields_to_review": ["placed", "x", "y", "r"],
            "post_action_or_future_fields_do_not_use_as_model_input_without_proof": [
                "won",
                "cleared",
                "garbage_cleared",
                "attack",
                "t_spin",
            ],
            "player_skill_metadata": ["rating", "glicko", "glicko_rd"],
        },
    }

    profile_path = output_dir / "top_players_s1_profile.json"
    profile_path.write_text(
        json.dumps(jsonable(profile), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Rows    : {overview.get('rows'):,}")
    print(f"Games   : {overview.get('games'):,}")
    print(
        "Rotation:",
        ", ".join(
            f"{x['value']}={x['rows']:,}" for x in distributions["r"]
        ),
    )
    print(
        "T-spin :",
        ", ".join(
            f"{x['value']}={x['rows']:,}" for x in distributions["t_spin"]
        ),
    )
    print(
        f"X range : {overview.get('min_x')} .. {overview.get('max_x')}"
    )
    print(
        f"Y range : {overview.get('min_y')} .. {overview.get('max_y')}"
    )
    print(
        f"Combo   : {overview.get('min_combo')} .. {overview.get('max_combo')}"
    )
    print(
        f"B2B     : {overview.get('min_btb')} .. {overview.get('max_btb')}"
    )
    print(
        f"Attack  : {overview.get('min_attack')} .. {overview.get('max_attack')}"
    )
    print(f"Profile : {profile_path}")
    print(f"Profile time: {elapsed_profile:.2f}s")

    if args.profile_only:
        print("Parquet : skipped (--profile-only)")
        return

    parquet_path = output_dir / args.parquet_name
    if parquet_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Parquet already exists: {parquet_path}\n"
            "Use --overwrite to replace it."
        )

    if parquet_path.exists():
        parquet_path.unlink()

    pq_path = qpath(parquet_path)
    convert_started = time.perf_counter()
    con.execute(
        f"""
        COPY (
            SELECT * FROM raw
        )
        TO '{pq_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 262144
        )
        """
    )
    convert_seconds = time.perf_counter() - convert_started

    # Validate row count after conversion.
    parquet_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{pq_path}')"
    ).fetchone()[0]
    expected_rows = int(overview["rows"])
    if int(parquet_rows) != expected_rows:
        raise SystemExit(
            f"Parquet row-count mismatch: CSV={expected_rows}, Parquet={parquet_rows}"
        )

    manifest = {
        "source_csv": str(input_path),
        "source_rows": expected_rows,
        "parquet": str(parquet_path),
        "parquet_rows": int(parquet_rows),
        "parquet_bytes": parquet_path.stat().st_size,
        "compression": "zstd",
        "row_group_size": 262144,
        "threads": threads,
        "convert_seconds": convert_seconds,
        "status": "PASS",
    }
    manifest_path = output_dir / "top_players_s1_parquet_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("PARQUET CONVERSION")
    print("-" * 96)
    print(f"Output  : {parquet_path}")
    print(f"Size    : {parquet_path.stat().st_size / (1024**2):.2f} MiB")
    print(f"Rows    : {parquet_rows:,}")
    print(f"Time    : {convert_seconds:.2f}s")
    print(f"Manifest: {manifest_path}")
    print("Result  : PASS")


if __name__ == "__main__":
    main()
