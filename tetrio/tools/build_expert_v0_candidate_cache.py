from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from tetrio.datasets.top_players_s1 import decode_playfield
from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetrio.network.encoding import EMPTY_PIECE_ID, PREVIEW_DEPTH, pack_board, piece_id
from tetrio.reachability import (
    TETRIO_ENTRY_RAISE_ROWS,
    enumerate_tetrio_reachable_placements,
)
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.types import PieceState


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    excluded: bool
    error: str | None
    game_id: int
    subframe: int
    state_board_packed: np.ndarray | None
    active_id: int | None
    hold_id: int | None
    preview_ids: np.ndarray | None
    candidate_board_packed: np.ndarray | None
    candidate_piece: np.ndarray | None
    candidate_rotation: np.ndarray | None
    candidate_x: np.ndarray | None
    candidate_y: np.ndarray | None
    candidate_use_hold: np.ndarray | None
    candidate_lines: np.ndarray | None
    expert_index: int | None
    use_hold: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build compact candidate shards for TETR.IO Expert v0. "
            "Only the expert-selected hold branch is enumerated in v0."
        )
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--rows",
        type=int,
        default=20_000,
        help="Rows to select. Use 0 to build every eligible row in the source parquet.",
    )
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    p.add_argument("--max-states", type=int, default=50_000)
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--backend", choices=("reference", "fast"), default="reference")
    p.add_argument("--shard-size", type=int, default=512)
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


def fetch_rows(path: Path, rows: int, seed: int) -> list[dict[str, Any]]:
    duckdb = require_duckdb()
    con = duckdb.connect(database=":memory:")
    limit_clause = "" if int(rows) == 0 else f"LIMIT {int(rows)}"
    cur = con.execute(
        f"""
        SELECT
            game_id, subframe, board_before, active_piece, hold_piece,
            preview_queue, placed_piece, final_x, final_y, final_rotation,
            use_hold, hold_mode
        FROM read_parquet('{qpath(path)}')
        WHERE hold_label_valid = TRUE
        ORDER BY hash(
            CAST(game_id AS VARCHAR)
            || ':' || CAST(subframe AS VARCHAR)
            || ':{int(seed)}'
        )
        {limit_clause}
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, tup)) for tup in cur.fetchall()]


def selected_piece(row: dict[str, Any]) -> str | None:
    active = str(row["active_piece"])
    hold = str(row["hold_piece"])
    preview = str(row["preview_queue"])
    use_hold = int(row["use_hold"])
    if use_hold == 0:
        return active
    if hold != "N":
        return hold
    return preview[0] if preview else None


def _reference_landings(board: np.ndarray, piece: str, max_states: int) -> list[PieceState]:
    best = {}
    for placement in enumerate_tetrio_reachable_placements(
        board,
        piece,
        max_states=max_states,
    ):
        key = placement.landing_state.geometry_key()
        old = best.get(key)
        if old is None or len(placement.path) < len(old.path):
            best[key] = placement

    placements = sorted(
        best.values(),
        key=lambda p: (
            p.landing_state.rotation % 4,
            p.landing_state.x,
            p.landing_state.y,
            len(p.path),
        ),
    )
    return [p.landing_state for p in placements]


def _candidate_landings(
    board: np.ndarray,
    piece: str,
    backend: str,
    max_states: int,
    fast_max_states: int,
) -> list[PieceState]:
    if backend == "fast":
        return enumerate_tetrio_reachable_geometries_fast(
            board,
            piece,
            max_states=fast_max_states,
        )
    return _reference_landings(board, piece, max_states)


def _reference_contains_target(
    board: np.ndarray,
    piece: str,
    target_key: tuple,
    max_states: int,
) -> tuple[list[PieceState], bool]:
    landings = _reference_landings(board, piece, max_states)
    return landings, any(
        landing.geometry_key() == target_key
        for landing in landings
    )


def build_one(item: tuple[dict[str, Any], int, int, str]) -> BuildResult:
    row, max_states, fast_max_states, backend = item
    game_id = int(row["game_id"])
    subframe = int(row["subframe"])
    try:
        source_piece = selected_piece(row)
        target_piece = str(row["placed_piece"])
        if source_piece != target_piece:
            raise RuntimeError(
                f"hold/source mismatch: selected={source_piece!r}, target={target_piece!r}"
            )

        board = decode_playfield(str(row["board_before"] or ""))
        target_state = PieceState(
            piece=target_piece,
            x=int(row["final_x"]),
            y=int(row["final_y"]),
            rotation=int(row["final_rotation"]),
        )
        target = target_state.geometry_key()

        landings = _candidate_landings(
            board,
            source_piece,
            backend,
            max_states,
            fast_max_states,
        )
        backend_has_target = any(
            landing.geometry_key() == target
            for landing in landings
        )

        if not backend_has_target:
            if backend == "fast":
                reference_landings, reference_has_target = _reference_contains_target(
                    board,
                    source_piece,
                    target,
                    max_states,
                )
                if reference_has_target:
                    raise RuntimeError(
                        "fast/reference parity failure: expert target missing "
                        f"from fast candidate set (fast={len(landings)}, "
                        f"reference={len(reference_landings)})"
                    )
                return BuildResult(
                    ok=False,
                    excluded=True,
                    error=(
                        "reference_expert_unreachable:"
                        f"fast_candidates={len(landings)},"
                        f"reference_candidates={len(reference_landings)}"
                    ),
                    game_id=game_id,
                    subframe=subframe,
                    state_board_packed=None,
                    active_id=None,
                    hold_id=None,
                    preview_ids=None,
                    candidate_board_packed=None,
                    candidate_piece=None,
                    candidate_rotation=None,
                    candidate_x=None,
                    candidate_y=None,
                    candidate_use_hold=None,
                    candidate_lines=None,
                    expert_index=None,
                    use_hold=None,
                )

            return BuildResult(
                ok=False,
                excluded=True,
                error=(
                    "reference_expert_unreachable:"
                    f"reference_candidates={len(landings)}"
                ),
                game_id=game_id,
                subframe=subframe,
                state_board_packed=None,
                active_id=None,
                hold_id=None,
                preview_ids=None,
                candidate_board_packed=None,
                candidate_piece=None,
                candidate_rotation=None,
                candidate_x=None,
                candidate_y=None,
                candidate_use_hold=None,
                candidate_lines=None,
                expert_index=None,
                use_hold=None,
            )

        use_hold = int(row["use_hold"])
        candidate_boards = []
        c_piece = []
        c_rot = []
        c_x = []
        c_y = []
        c_hold = []
        c_lines = []
        expert_index = None

        for idx, landing in enumerate(landings):
            locked = lock_piece(board, landing, TETRIO_MOVEMENT)
            after, lines = clear_lines(locked, TETRIO_MOVEMENT)

            candidate_boards.append(pack_board(after))
            c_piece.append(piece_id(landing.piece))
            c_rot.append(int(landing.rotation) % 4)
            c_x.append(int(landing.x))
            c_y.append(int(landing.y))
            c_hold.append(use_hold)
            c_lines.append(int(lines))

            if landing.geometry_key() == target:
                expert_index = idx

        if expert_index is None:
            raise RuntimeError("internal error: expert target disappeared")

        preview = str(row["preview_queue"])
        preview_ids = np.full(PREVIEW_DEPTH, EMPTY_PIECE_ID, dtype=np.uint8)
        for i, p in enumerate(preview[:PREVIEW_DEPTH]):
            preview_ids[i] = piece_id(p)

        return BuildResult(
            ok=True,
            excluded=False,
            error=None,
            game_id=game_id,
            subframe=subframe,
            state_board_packed=pack_board(board),
            active_id=piece_id(str(row["active_piece"])),
            hold_id=piece_id(str(row["hold_piece"])),
            preview_ids=preview_ids,
            candidate_board_packed=np.stack(candidate_boards).astype(np.uint8),
            candidate_piece=np.asarray(c_piece, dtype=np.uint8),
            candidate_rotation=np.asarray(c_rot, dtype=np.uint8),
            candidate_x=np.asarray(c_x, dtype=np.int8),
            candidate_y=np.asarray(c_y, dtype=np.int8),
            candidate_use_hold=np.asarray(c_hold, dtype=np.uint8),
            candidate_lines=np.asarray(c_lines, dtype=np.uint8),
            expert_index=int(expert_index),
            use_hold=use_hold,
        )
    except Exception as exc:
        return BuildResult(
            ok=False,
            excluded=False,
            error=f"{type(exc).__name__}:{exc}",
            game_id=game_id,
            subframe=subframe,
            state_board_packed=None,
            active_id=None,
            hold_id=None,
            preview_ids=None,
            candidate_board_packed=None,
            candidate_piece=None,
            candidate_rotation=None,
            candidate_x=None,
            candidate_y=None,
            candidate_use_hold=None,
            candidate_lines=None,
            expert_index=None,
            use_hold=None,
        )


def write_shard(path: Path, samples: list[BuildResult]) -> dict[str, int]:
    counts = np.asarray(
        [len(s.candidate_piece) for s in samples],
        dtype=np.int32,
    )
    offsets = np.zeros(len(samples) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)

    np.savez(
        path,
        state_board_packed=np.stack([s.state_board_packed for s in samples]),
        state_active=np.asarray([s.active_id for s in samples], dtype=np.uint8),
        state_hold=np.asarray([s.hold_id for s in samples], dtype=np.uint8),
        state_preview=np.stack([s.preview_ids for s in samples]).astype(np.uint8),
        candidate_offsets=offsets,
        candidate_board_packed=np.concatenate([s.candidate_board_packed for s in samples]),
        candidate_piece=np.concatenate([s.candidate_piece for s in samples]),
        candidate_rotation=np.concatenate([s.candidate_rotation for s in samples]),
        candidate_x=np.concatenate([s.candidate_x for s in samples]),
        candidate_y=np.concatenate([s.candidate_y for s in samples]),
        candidate_use_hold=np.concatenate([s.candidate_use_hold for s in samples]),
        candidate_lines=np.concatenate([s.candidate_lines for s in samples]),
        expert_index=np.asarray([s.expert_index for s in samples], dtype=np.int16),
        use_hold=np.asarray([s.use_hold for s in samples], dtype=np.uint8),
        game_id=np.asarray([s.game_id for s in samples], dtype=np.int64),
        subframe=np.asarray([s.subframe for s in samples], dtype=np.int64),
    )
    return {
        "rows": len(samples),
        "candidates": int(counts.sum()),
        "min_candidates": int(counts.min()),
        "max_candidates": int(counts.max()),
    }


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input expert parquet not found: {args.input}")
    if args.rows < 0 or args.shard_size <= 0:
        raise SystemExit("--rows must be >= 0 (0 means all); --shard-size must be positive")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("shard_*.npz")) + [out / "manifest.json"]
    existing = [p for p in existing if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f"Output directory already contains Expert-v0 data: {out}\n"
            "Use --overwrite to replace it."
        )
    if args.overwrite:
        for p in existing:
            p.unlink()

    rows = fetch_rows(args.input, args.rows, args.seed)
    title_backend = "FAST GEOMETRY" if args.backend == "fast" else "REFERENCE"
    print("=" * 100)
    print(f"TETR.IO EXPERT V0 — {title_backend} CANDIDATE CACHE")
    print("=" * 100)
    print(f"Input       : {args.input}")
    print(f"Rows        : {len(rows):,}")
    print(f"Backend     : {args.backend}")
    print(f"Workers     : {args.workers}")
    print(f"Shard size  : {args.shard_size}")
    if args.backend == "reference":
        print(f"Max states  : {args.max_states:,}")
    else:
        print(f"Max states  : {args.fast_max_states:,} (geometry-only)")
    print(f"Seed        : {args.seed}")
    print()

    started = time.perf_counter()
    exclusions = []
    hard_failures = []
    shard_stats = []
    buffer: list[BuildResult] = []
    built_rows = 0
    total_candidates = 0
    shard_index = 0

    work = [
        (row, args.max_states, args.fast_max_states, args.backend)
        for row in rows
    ]
    if args.workers == 1:
        iterator = map(build_one, work)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        iterator = pool.map(
            build_one,
            work,
            chunksize=max(1, len(work) // (args.workers * 32)),
        )

    try:
        for pos, result in enumerate(iterator, 1):
            if not result.ok:
                record = {
                    "game_id": result.game_id,
                    "subframe": result.subframe,
                    "error": result.error,
                }
                if result.excluded:
                    exclusions.append(record)
                else:
                    hard_failures.append(record)
            else:
                buffer.append(result)
                if len(buffer) >= args.shard_size:
                    path = out / f"shard_{shard_index:05d}.npz"
                    stats = write_shard(path, buffer)
                    shard_stats.append({"file": path.name, **stats})
                    built_rows += stats["rows"]
                    total_candidates += stats["candidates"]
                    shard_index += 1
                    buffer = []
            if pos % 1000 == 0 or pos == len(work):
                elapsed = time.perf_counter() - started
                print(
                    f"  processed={pos:,}/{len(work):,} "
                    f"built={built_rows + len(buffer):,} "
                    f"excluded={len(exclusions):,} "
                    f"hard_failed={len(hard_failures):,} "
                    f"rate={pos/elapsed:.2f} rows/s"
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if buffer:
        path = out / f"shard_{shard_index:05d}.npz"
        stats = write_shard(path, buffer)
        shard_stats.append({"file": path.name, **stats})
        built_rows += stats["rows"]
        total_candidates += stats["candidates"]

    elapsed = time.perf_counter() - started
    excluded_rows = len(exclusions)
    hard_failed_rows = len(hard_failures)
    accounted_rows = built_rows + excluded_rows + hard_failed_rows
    all_counts = [s["min_candidates"] for s in shard_stats] + [s["max_candidates"] for s in shard_stats]

    if hard_failed_rows:
        status = "FAIL"
    elif accounted_rows != len(rows):
        status = "FAIL"
        hard_failures.append({
            "game_id": -1,
            "subframe": -1,
            "error": (
                "internal_accounting_mismatch:"
                f"selected={len(rows)},accounted={accounted_rows}"
            ),
        })
        hard_failed_rows = len(hard_failures)
    elif excluded_rows:
        status = "PASS_WITH_EXCLUSIONS"
    else:
        status = "PASS"

    manifest = {
        "source": str(args.input),
        "requested_rows": args.rows,
        "selected_rows": len(rows),
        "built_rows": built_rows,
        "failed_rows": hard_failed_rows,
        "hard_failed_rows": hard_failed_rows,
        "excluded_rows": excluded_rows,
        "total_candidates": total_candidates,
        "mean_candidates": None if built_rows == 0 else total_candidates / built_rows,
        "min_candidates": None if not all_counts else min(all_counts),
        "max_candidates": None if not all_counts else max(all_counts),
        "backend": args.backend,
        "workers": args.workers,
        "max_states": args.max_states if args.backend == "reference" else args.fast_max_states,
        "reference_fallback_max_states": args.max_states,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "seconds": elapsed,
        "rows_per_second": None if elapsed == 0 else len(rows) / elapsed,
        "candidate_contract": (
            "expert-selected hold branch; unique landing geometries; "
            + (
                "fast geometry-only reachability with path-sensitive reference "
                "fallback on missing expert target"
                if args.backend == "fast"
                else "path-sensitive reference reachability deduped by geometry"
            )
        ),
        "reachability_contract": {
            "game": "tetrio",
            "entry_raise_rows_vs_generic": int(TETRIO_ENTRY_RAISE_ROWS),
            "rotation_system": "TETR.IO SRS+/180 production ruleset",
            "landing_identity": "(piece,x,y,rotation)",
            "exception_policy": "reference_confirmed_expert_unreachable_v1",
        },
        "exclusions": exclusions,
        "hard_failures": hard_failures,
        "failures": hard_failures,
        "shards": shard_stats,
        "status": status,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Built rows       : {built_rows:,}")
    print(f"Excluded rows    : {excluded_rows:,}")
    print(f"Hard failures    : {hard_failed_rows:,}")
    print(f"Candidates       : {total_candidates:,}")
    print(f"Mean candidates  : {manifest['mean_candidates']:.2f}" if built_rows else "Mean candidates  : n/a")
    print(f"Build time       : {elapsed:.2f}s")
    print(f"Throughput       : {manifest['rows_per_second']:.2f} rows/s")
    print(f"Manifest         : {out / 'manifest.json'}")
    if exclusions:
        print("Reference-confirmed exclusions:")
        for item in exclusions:
            print(
                f"  {item['game_id']}/{item['subframe']} "
                f"{item['error']}"
            )
    if hard_failures:
        print("Hard failures:")
        for item in hard_failures[:20]:
            print(
                f"  {item['game_id']}/{item['subframe']} "
                f"{item['error']}"
            )
    print(f"Result           : {manifest['status']}")



if __name__ == "__main__":
    main()
