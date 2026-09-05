from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe expert placement corpus from the historical "
            "TETR.IO top-player Parquet dataset. Splits are deterministic by game_id."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path(r"data\tetrio\processed\top_players_s1.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"data\tetrio\expert"),
    )
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--split-seed", type=int, default=20260905)
    p.add_argument("--train-pct", type=int, default=90)
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def require_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "DuckDB is required. Install it with:\n"
            r".venv\Scripts\python.exe -m pip install duckdb"
        ) from exc
    return duckdb


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def canonical_piece_sql(expr: str) -> str:
    # Historical corpus/extractor names J/L opposite to this project's
    # canonical geometry. translate() swaps them without cascading replace().
    return f"translate({expr}, 'JL', 'LJ')"


def main():
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input Parquet not found: {args.input}")

    if args.train_pct <= 0 or args.val_pct <= 0:
        raise SystemExit("train/val percentages must be positive")
    if args.train_pct + args.val_pct >= 100:
        raise SystemExit("train_pct + val_pct must be < 100")
    test_pct = 100 - args.train_pct - args.val_pct

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "train": out / "top_players_s1_train.parquet",
        "val": out / "top_players_s1_val.parquet",
        "test": out / "top_players_s1_test.parquet",
        "ambiguous": out / "top_players_s1_ambiguous_hold.parquet",
        "unmatched": out / "top_players_s1_unmatched.parquet",
        "manifest": out / "top_players_s1_expert_manifest.json",
    }

    existing = [p for p in files.values() if p.exists()]
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
    threads = max(1, int(args.threads))
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")

    inp = qpath(args.input)
    t0 = time.perf_counter()

    # `hold` and `next` on a row were validated as POST-placement state.
    # Therefore the PRE-action state for row t comes from:
    #   board_before = row[t].playfield
    #   active_before = row[t-1].next[0]
    #   hold_before = row[t-1].hold
    #   preview_before = row[t-1].next[1:]
    #
    # `combo`, `btb`, garbage and attack fields are intentionally retained only
    # as raw/outcome metadata. They are NOT part of the validated state-input
    # contract yet.
    con.execute(
        f"""
        CREATE TEMP VIEW sequenced AS
        SELECT
            *,
            lag(hold) OVER (PARTITION BY game_id ORDER BY subframe) AS prev_hold,
            lag(next) OVER (PARTITION BY game_id ORDER BY subframe) AS prev_next
        FROM read_parquet('{inp}')
        """
    )

    con.execute(
        """
        CREATE TEMP VIEW inferred AS
        SELECT
            *,
            substr(prev_next, 1, 1) AS active_before_raw,
            substr(prev_next, 2) AS preview_before_raw,

            (
                placed = substr(prev_next, 1, 1)
                AND hold = prev_hold
                AND starts_with(next, substr(prev_next, 2))
            ) AS fits_no_hold,

            (
                prev_hold <> 'N'
                AND placed = prev_hold
                AND hold = substr(prev_next, 1, 1)
                AND starts_with(next, substr(prev_next, 2))
            ) AS fits_hold_swap,

            (
                prev_hold = 'N'
                AND length(prev_next) >= 2
                AND placed = substr(prev_next, 2, 1)
                AND hold = substr(prev_next, 1, 1)
                AND starts_with(next, substr(prev_next, 3))
            ) AS fits_hold_empty
        FROM sequenced
        WHERE prev_next IS NOT NULL AND length(prev_next) > 0
        """
    )

    con.execute(
        """
        CREATE TEMP VIEW classified AS
        SELECT
            *,
            CAST(fits_no_hold AS INTEGER)
              + CAST(fits_hold_swap AS INTEGER)
              + CAST(fits_hold_empty AS INTEGER) AS fit_count,
            CASE
                WHEN fits_no_hold AND NOT fits_hold_swap AND NOT fits_hold_empty
                    THEN 'no_hold'
                WHEN fits_hold_swap AND NOT fits_no_hold AND NOT fits_hold_empty
                    THEN 'hold_swap'
                WHEN fits_hold_empty AND NOT fits_no_hold AND NOT fits_hold_swap
                    THEN 'hold_empty'
                WHEN (
                    CAST(fits_no_hold AS INTEGER)
                    + CAST(fits_hold_swap AS INTEGER)
                    + CAST(fits_hold_empty AS INTEGER)
                ) > 1 THEN 'ambiguous'
                ELSE 'unmatched'
            END AS hold_mode
        FROM inferred
        """
    )

    active_canon = canonical_piece_sql("active_before_raw")
    hold_canon = canonical_piece_sql("prev_hold")
    preview_canon = canonical_piece_sql("preview_before_raw")
    placed_canon = canonical_piece_sql("placed")

    # Canonical final placement anchor from the 5,000-game mapping validation.
    con.execute(
        f"""
        CREATE TEMP VIEW expert_base AS
        SELECT
            game_id,
            subframe,

            -- Validated PRE-action state.
            playfield AS board_before,
            {active_canon} AS active_piece,
            {hold_canon} AS hold_piece,
            {preview_canon} AS preview_queue,

            -- Expert final placement target in canonical project geometry.
            {placed_canon} AS placed_piece,
            CASE
                WHEN placed = 'I' AND r IN ('E','S') THEN x - 2
                WHEN placed = 'O' AND r = 'N' THEN x
                ELSE x - 1
            END AS final_x,
            CASE
                WHEN placed = 'I' AND r IN ('S','W') THEN 37 - y
                ELSE 38 - y
            END AS final_y,
            CASE r
                WHEN 'N' THEN 0
                WHEN 'E' THEN 1
                WHEN 'S' THEN 2
                WHEN 'W' THEN 3
                ELSE NULL
            END AS final_rotation,

            -- Hold target: reliable only when hold_label_valid=true.
            CASE
                WHEN hold_mode = 'no_hold' THEN 0
                WHEN hold_mode IN ('hold_swap','hold_empty') THEN 1
                ELSE NULL
            END AS use_hold,
            hold_mode,
            (fit_count = 1) AS hold_label_valid,

            -- Skill / grouping metadata; not game-state inputs.
            won,
            rating,
            glicko,
            glicko_rd,

            -- Outcome/raw fields retained for research only.
            -- Do NOT silently feed these to the model as pre-action state.
            cleared AS outcome_cleared,
            garbage_cleared AS outcome_garbage_cleared,
            attack AS outcome_attack,
            t_spin AS outcome_t_spin,
            btb AS raw_btb,
            combo AS raw_combo,
            immediate_garbage AS raw_immediate_garbage,
            incoming_garbage AS raw_incoming_garbage,

            -- Deterministic game-level split key.
            (
                hash(CAST(game_id AS VARCHAR) || ':{int(args.split_seed)}')
                % 10000
            ) AS split_bucket
        FROM classified
        """
    )

    # Unambiguous hold rows are the clean first-stage supervised set.
    train_hi = args.train_pct * 100
    val_hi = (args.train_pct + args.val_pct) * 100

    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE(split_bucket)
            FROM expert_base
            WHERE hold_mode IN ('no_hold','hold_swap','hold_empty')
              AND split_bucket < {train_hi}
            ORDER BY game_id, subframe
        )
        TO '{qpath(files["train"])}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE(split_bucket)
            FROM expert_base
            WHERE hold_mode IN ('no_hold','hold_swap','hold_empty')
              AND split_bucket >= {train_hi}
              AND split_bucket < {val_hi}
            ORDER BY game_id, subframe
        )
        TO '{qpath(files["val"])}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE(split_bucket)
            FROM expert_base
            WHERE hold_mode IN ('no_hold','hold_swap','hold_empty')
              AND split_bucket >= {val_hi}
            ORDER BY game_id, subframe
        )
        TO '{qpath(files["test"])}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
        """
    )

    # Ambiguous hold transitions remain valuable placement labels because their
    # final board/action geometry is observed. Keep them separately so later
    # training can mask the hold-head loss rather than discarding the placement.
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE(split_bucket)
            FROM expert_base
            WHERE hold_mode = 'ambiguous'
            ORDER BY game_id, subframe
        )
        TO '{qpath(files["ambiguous"])}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)
        """
    )

    # Fail-closed quarantine for the tiny unmatched set.
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM classified
            WHERE hold_mode = 'unmatched'
            ORDER BY game_id, subframe
        )
        TO '{qpath(files["unmatched"])}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    build_s = time.perf_counter() - t0

    def scalar(sql: str):
        return con.execute(sql).fetchone()[0]

    stats = {}
    for split in ("train", "val", "test", "ambiguous", "unmatched"):
        p = files[split]
        stats[split] = {
            "rows": int(scalar(f"SELECT count(*) FROM read_parquet('{qpath(p)}')")),
            "games": int(
                scalar(f"SELECT count(DISTINCT game_id) FROM read_parquet('{qpath(p)}')")
            ),
            "bytes": p.stat().st_size,
        }

    # Explicitly verify game-level split isolation.
    overlap_train_val = int(
        scalar(
            f"""
            SELECT count(*)
            FROM (
                SELECT DISTINCT game_id FROM read_parquet('{qpath(files["train"])}')
                INTERSECT
                SELECT DISTINCT game_id FROM read_parquet('{qpath(files["val"])}')
            )
            """
        )
    )
    overlap_train_test = int(
        scalar(
            f"""
            SELECT count(*)
            FROM (
                SELECT DISTINCT game_id FROM read_parquet('{qpath(files["train"])}')
                INTERSECT
                SELECT DISTINCT game_id FROM read_parquet('{qpath(files["test"])}')
            )
            """
        )
    )
    overlap_val_test = int(
        scalar(
            f"""
            SELECT count(*)
            FROM (
                SELECT DISTINCT game_id FROM read_parquet('{qpath(files["val"])}')
                INTERSECT
                SELECT DISTINCT game_id FROM read_parquet('{qpath(files["test"])}')
            )
            """
        )
    )

    total_source_rows = int(
        scalar(f"SELECT count(*) FROM read_parquet('{inp}')")
    )
    total_source_games = int(
        scalar(f"SELECT count(DISTINCT game_id) FROM read_parquet('{inp}')")
    )
    total_transitions = int(scalar("SELECT count(*) FROM classified"))
    unique_rows = stats["train"]["rows"] + stats["val"]["rows"] + stats["test"]["rows"]
    ambiguous_rows = stats["ambiguous"]["rows"]
    unmatched_rows = stats["unmatched"]["rows"]

    manifest = {
        "source": {
            "path": str(args.input),
            "rows": total_source_rows,
            "games": total_source_games,
        },
        "corpus": {
            "transitions_with_previous_state": total_transitions,
            "unique_hold_rows": unique_rows,
            "ambiguous_hold_rows": ambiguous_rows,
            "unmatched_rows": unmatched_rows,
            "classified_rows": unique_rows + ambiguous_rows,
            "classification_rate": (
                None if total_transitions == 0
                else (unique_rows + ambiguous_rows) / total_transitions
            ),
        },
        "split": {
            "seed": args.split_seed,
            "train_pct": args.train_pct,
            "val_pct": args.val_pct,
            "test_pct": test_pct,
            "method": "deterministic hash bucket by game_id",
            "overlap_train_val_games": overlap_train_val,
            "overlap_train_test_games": overlap_train_test,
            "overlap_val_test_games": overlap_val_test,
        },
        "files": {
            name: {
                "path": str(files[name]),
                **stats[name],
            }
            for name in ("train", "val", "test", "ambiguous", "unmatched")
        },
        "validated_state_columns": [
            "board_before",
            "active_piece",
            "hold_piece",
            "preview_queue",
        ],
        "expert_action_columns": [
            "placed_piece",
            "final_x",
            "final_y",
            "final_rotation",
            "use_hold",
            "hold_label_valid",
        ],
        "metadata_not_model_input": [
            "game_id",
            "subframe",
            "won",
            "rating",
            "glicko",
            "glicko_rd",
        ],
        "raw_or_post_action_columns_not_validated_as_state_input": [
            "outcome_cleared",
            "outcome_garbage_cleared",
            "outcome_attack",
            "outcome_t_spin",
            "raw_btb",
            "raw_combo",
            "raw_immediate_garbage",
            "raw_incoming_garbage",
        ],
        "mapping_notes": {
            "dataset_J_to_canonical": "L",
            "dataset_L_to_canonical": "J",
            "rotation": "N/E/S/W -> 0/1/2/3 clockwise",
            "I_origin": {
                "N": {"x_offset": -1, "y_base": 38},
                "E": {"x_offset": -2, "y_base": 38},
                "S": {"x_offset": -2, "y_base": 37},
                "W": {"x_offset": -1, "y_base": 37},
            },
            "O_N_origin": {"x_offset": 0, "y_base": 38},
            "TSZJL_origin": {"x_offset": -1, "y_base": 38},
        },
        "build": {
            "threads": threads,
            "seconds": build_s,
            "status": "PASS"
            if overlap_train_val == overlap_train_test == overlap_val_test == 0
            else "FAIL",
        },
    }

    files["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("TETR.IO HISTORICAL TOP-PLAYER EXPERT CORPUS")
    print("=" * 100)
    print(f"Source rows      : {total_source_rows:,}")
    print(f"Source games     : {total_source_games:,}")
    print(f"Transitions      : {total_transitions:,}")
    print(f"Unique hold      : {unique_rows:,}")
    print(f"Ambiguous hold   : {ambiguous_rows:,}")
    print(f"Unmatched        : {unmatched_rows:,}")
    print(
        "Classified       : "
        f"{(100*(unique_rows+ambiguous_rows)/total_transitions if total_transitions else 0):.5f}%"
    )
    print()
    for split in ("train", "val", "test"):
        s = stats[split]
        print(
            f"{split.upper():5s}: rows={s['rows']:,} games={s['games']:,} "
            f"size={s['bytes']/(1024**2):.2f} MiB"
        )
    print(
        f"AMBIG: rows={stats['ambiguous']['rows']:,} "
        f"games={stats['ambiguous']['games']:,}"
    )
    print(
        f"BAD  : rows={stats['unmatched']['rows']:,} "
        f"games={stats['unmatched']['games']:,}"
    )
    print()
    print(
        "Split overlaps   : "
        f"train/val={overlap_train_val}, "
        f"train/test={overlap_train_test}, "
        f"val/test={overlap_val_test}"
    )
    print(f"Build time       : {build_s:.2f}s")
    print(f"Manifest         : {files['manifest']}")
    print(f"Result           : {manifest['build']['status']}")


if __name__ == "__main__":
    main()
