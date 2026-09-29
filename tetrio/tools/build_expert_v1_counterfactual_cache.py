from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Iterable

import numpy as np

from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetrio.network.encoding import (
    EMPTY_PIECE_ID,
    PIECES,
    pack_board,
    piece_id,
    unpack_boards,
)
from tetrio.network.cache import load_shard, shard_paths
from tetrio.reachability import enumerate_tetrio_reachable_placements
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.types import PieceState


ID_TO_PIECE = {i: p for i, p in enumerate(PIECES)}


@dataclass(frozen=True)
class CounterfactualResult:
    ok: bool
    error: str | None
    used_reference_fallback: bool
    audited_reference: bool
    candidate_board_packed: np.ndarray | None
    candidate_piece: np.ndarray | None
    candidate_rotation: np.ndarray | None
    candidate_x: np.ndarray | None
    candidate_y: np.ndarray | None
    candidate_use_hold: np.ndarray | None
    candidate_lines: np.ndarray | None
    candidate_holes: np.ndarray | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the *opposite Hold branch* sidecar for Expert-v1. "
            "The existing Expert-v0 cache remains the expert branch; this avoids "
            "duplicating it in a second unified cache."
        )
    )
    p.add_argument(
        "--v0-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\train_full_fast_s8192"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Prefix rows to build; 0 means every row in the V0 cache.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 2)),
    )
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument(
        "--reference-audit-every",
        type=int,
        default=5_000,
        help=(
            "With fast backend, compare the complete counterfactual candidate "
            "set to the path-sensitive reference every N global rows. 0 disables."
        ),
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete matching sidecar shards already in --output-dir.",
    )
    p.add_argument(
        "--reuse-dir",
        type=Path,
        default=None,
        help=(
            "Optional older sidecar cache to seed a larger build. Only complete "
            "shards whose row identities match the current V0 cache are copied."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _piece_name(pid: int) -> str | None:
    if int(pid) == EMPTY_PIECE_ID:
        return None
    return ID_TO_PIECE[int(pid)]


def _reference_landings(
    board: np.ndarray,
    piece: str,
    max_states: int,
) -> list[PieceState]:
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
    return [
        p.landing_state
        for p in sorted(
            best.values(),
            key=lambda p: (
                p.landing_state.rotation % 4,
                p.landing_state.x,
                p.landing_state.y,
                len(p.path),
            ),
        )
    ]


def _geometry_set(states: Iterable[PieceState]) -> set[tuple]:
    return {s.geometry_key() for s in states}


def holes_for_boards(boards: np.ndarray) -> np.ndarray:
    """Count covered empty cells on 40x10 boards, top row first."""
    arr = np.asarray(boards) != 0
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.shape[1:] != (40, 10):
        raise ValueError(f"Expected [N,40,10], got {arr.shape}")
    seen = np.maximum.accumulate(arr, axis=1)
    holes = ((~arr) & seen).sum(axis=(1, 2))
    return holes.astype(np.uint8, copy=False)


def holes_for_packed(
    packed: np.ndarray,
    *,
    chunk: int = 100_000,
) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty(len(packed), dtype=np.uint8)
    for start in range(0, len(packed), int(chunk)):
        part = unpack_boards(packed[start:start + int(chunk)]).reshape(-1, 40, 10)
        out[start:start + len(part)] = holes_for_boards(part)
    return out


def other_branch_piece(
    *,
    active_id: int,
    hold_id: int,
    preview_ids: np.ndarray,
    expert_use_hold: int,
) -> tuple[str, int]:
    """Return (piece, counterfactual_use_hold)."""
    active = _piece_name(active_id)
    hold = _piece_name(hold_id)
    preview0 = _piece_name(int(preview_ids[0]))

    if active is None:
        raise RuntimeError("active piece is empty")

    if int(expert_use_hold) == 1:
        # Expert used Hold, so the counterfactual branch is simply no-hold.
        return active, 0

    # Expert did not Hold; counterfactual branch uses Hold.
    selected = hold if hold is not None else preview0
    if selected is None:
        raise RuntimeError("hold-empty counterfactual has no preview[0]")
    return selected, 1


def _build_one(item) -> CounterfactualResult:
    (
        board_packed,
        active_id,
        hold_id,
        preview_ids,
        expert_use_hold,
        backend,
        fast_max_states,
        reference_max_states,
        do_audit,
    ) = item

    try:
        board = unpack_boards(
            np.asarray(board_packed, dtype=np.uint8)[None, :]
        )[0].reshape(40, 10).astype(np.uint8, copy=False)
        selected_piece, cf_use_hold = other_branch_piece(
            active_id=int(active_id),
            hold_id=int(hold_id),
            preview_ids=np.asarray(preview_ids),
            expert_use_hold=int(expert_use_hold),
        )

        fallback = False
        audited = False

        if backend == "reference":
            landings = _reference_landings(
                board,
                selected_piece,
                int(reference_max_states),
            )
        else:
            landings = enumerate_tetrio_reachable_geometries_fast(
                board,
                selected_piece,
                max_states=int(fast_max_states),
            )

            if not landings:
                reference = _reference_landings(
                    board,
                    selected_piece,
                    int(reference_max_states),
                )
                if reference:
                    landings = reference
                    fallback = True

            if do_audit:
                reference = _reference_landings(
                    board,
                    selected_piece,
                    int(reference_max_states),
                )
                audited = True
                fast_keys = _geometry_set(landings)
                ref_keys = _geometry_set(reference)
                if fast_keys != ref_keys:
                    only_fast = sorted(fast_keys - ref_keys)[:8]
                    only_ref = sorted(ref_keys - fast_keys)[:8]
                    raise RuntimeError(
                        "fast/reference counterfactual parity failure: "
                        f"piece={selected_piece} fast={len(fast_keys)} "
                        f"reference={len(ref_keys)} "
                        f"only_fast={only_fast} only_ref={only_ref}"
                    )

        boards = []
        pieces = []
        rotations = []
        xs = []
        ys = []
        use_hold = []
        lines = []

        for landing in landings:
            locked = lock_piece(board, landing, TETRIO_MOVEMENT)
            after, cleared = clear_lines(locked, TETRIO_MOVEMENT)
            boards.append(pack_board(after))
            pieces.append(piece_id(landing.piece))
            rotations.append(int(landing.rotation) % 4)
            xs.append(int(landing.x))
            ys.append(int(landing.y))
            use_hold.append(cf_use_hold)
            lines.append(int(cleared))

        if boards:
            board_arr = np.stack(boards).astype(np.uint8)
            hole_arr = holes_for_packed(board_arr)
        else:
            board_arr = np.empty((0, 50), dtype=np.uint8)
            hole_arr = np.empty((0,), dtype=np.uint8)

        return CounterfactualResult(
            ok=True,
            error=None,
            used_reference_fallback=fallback,
            audited_reference=audited,
            candidate_board_packed=board_arr,
            candidate_piece=np.asarray(pieces, dtype=np.uint8),
            candidate_rotation=np.asarray(rotations, dtype=np.uint8),
            candidate_x=np.asarray(xs, dtype=np.int8),
            candidate_y=np.asarray(ys, dtype=np.int8),
            candidate_use_hold=np.asarray(use_hold, dtype=np.uint8),
            candidate_lines=np.asarray(lines, dtype=np.uint8),
            candidate_holes=hole_arr,
        )
    except Exception as exc:
        return CounterfactualResult(
            ok=False,
            error=f"{type(exc).__name__}:{exc}",
            used_reference_fallback=False,
            audited_reference=False,
            candidate_board_packed=None,
            candidate_piece=None,
            candidate_rotation=None,
            candidate_x=None,
            candidate_y=None,
            candidate_use_hold=None,
            candidate_lines=None,
            candidate_holes=None,
        )


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sidecar(
    path: Path,
    *,
    v0: dict[str, np.ndarray],
    row_count: int,
    results: list[CounterfactualResult],
) -> dict:
    counts = np.asarray(
        [len(r.candidate_piece) for r in results],
        dtype=np.int32,
    )
    offsets = np.zeros(row_count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)

    # Structural labels for the already-cached expert branch are tiny sidecar
    # metadata, not a duplicate of the large candidate board cache.
    v0_offsets = v0["candidate_offsets"].astype(np.int64, copy=False)
    selected_candidate_count = int(v0_offsets[row_count])
    selected_holes = holes_for_packed(
        v0["candidate_board_packed"][:selected_candidate_count]
    )
    state_holes = holes_for_packed(v0["state_board_packed"][:row_count])

    def cat(name, dtype):
        arrays = [getattr(r, name) for r in results]
        if not arrays or sum(len(x) for x in arrays) == 0:
            shape = (0, 50) if name == "candidate_board_packed" else (0,)
            return np.empty(shape, dtype=dtype)
        return np.concatenate(arrays).astype(dtype, copy=False)

    np.savez(
        path,
        candidate_offsets=offsets,
        candidate_board_packed=cat("candidate_board_packed", np.uint8),
        candidate_piece=cat("candidate_piece", np.uint8),
        candidate_rotation=cat("candidate_rotation", np.uint8),
        candidate_x=cat("candidate_x", np.int8),
        candidate_y=cat("candidate_y", np.int8),
        candidate_use_hold=cat("candidate_use_hold", np.uint8),
        candidate_lines=cat("candidate_lines", np.uint8),
        candidate_holes=cat("candidate_holes", np.uint8),
        selected_candidate_holes=selected_holes,
        holes_before=state_holes,
        game_id=np.asarray(v0["game_id"][:row_count], dtype=np.int64),
        subframe=np.asarray(v0["subframe"][:row_count], dtype=np.int64),
    )
    return {
        "rows": row_count,
        "counterfactual_candidates": int(counts.sum()),
        "selected_candidates": selected_candidate_count,
        "min_counterfactual_candidates": int(counts.min()) if row_count else 0,
        "max_counterfactual_candidates": int(counts.max()) if row_count else 0,
    }



def _existing_sidecar_stats(
    path: Path,
    *,
    v0: dict[str, np.ndarray],
    expected_rows: int,
) -> dict | None:
    """Validate a reusable sidecar shard against the authoritative V0 rows."""
    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as d:
            required = {
                "candidate_offsets",
                "candidate_board_packed",
                "candidate_piece",
                "candidate_holes",
                "selected_candidate_holes",
                "holes_before",
                "game_id",
                "subframe",
            }
            if not required.issubset(d.files):
                return None

            rows = int(d["holes_before"].shape[0])
            if rows != int(expected_rows):
                return None
            if int(d["candidate_offsets"].shape[0]) != rows + 1:
                return None

            candidate_count = int(d["candidate_offsets"][-1])
            if int(d["candidate_piece"].shape[0]) != candidate_count:
                return None
            if int(d["candidate_board_packed"].shape[0]) != candidate_count:
                return None
            if int(d["candidate_holes"].shape[0]) != candidate_count:
                return None

            expected_game = np.asarray(v0["game_id"][:rows], dtype=np.int64)
            expected_sub = np.asarray(v0["subframe"][:rows], dtype=np.int64)
            if not np.array_equal(
                np.asarray(d["game_id"], dtype=np.int64),
                expected_game,
            ):
                return None
            if not np.array_equal(
                np.asarray(d["subframe"], dtype=np.int64),
                expected_sub,
            ):
                return None

            selected_candidates = int(
                np.asarray(v0["candidate_offsets"], dtype=np.int64)[rows]
            )
            if int(d["selected_candidate_holes"].shape[0]) != selected_candidates:
                return None

            counts = np.diff(
                np.asarray(d["candidate_offsets"], dtype=np.int64)
            )
            return {
                "rows": rows,
                "counterfactual_candidates": candidate_count,
                "selected_candidates": selected_candidates,
                "min_counterfactual_candidates": (
                    int(counts.min()) if rows else 0
                ),
                "max_counterfactual_candidates": (
                    int(counts.max()) if rows else 0
                ),
            }
    except Exception:
        return None


def _try_reuse_sidecar(
    *,
    output_path: Path,
    reuse_dir: Path | None,
    v0: dict[str, np.ndarray],
    expected_rows: int,
    resume: bool,
) -> tuple[dict | None, str | None]:
    if resume:
        stats = _existing_sidecar_stats(
            output_path,
            v0=v0,
            expected_rows=expected_rows,
        )
        if stats is not None:
            return stats, "resume"

    if reuse_dir is not None:
        source = reuse_dir / output_path.name
        stats = _existing_sidecar_stats(
            source,
            v0=v0,
            expected_rows=expected_rows,
        )
        if stats is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_path)
            return stats, "reuse-dir"

    return None, None


def main() -> None:
    args = parse_args()

    manifest_path = args.v0_cache / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"V0 manifest not found: {manifest_path}")
    v0_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if v0_manifest.get("status") not in ("PASS", "PASS_WITH_EXCLUSIONS"):
        raise SystemExit(
            f"V0 cache status must be PASS/PASS_WITH_EXCLUSIONS; "
            f"got {v0_manifest.get('status')!r}"
        )
    if args.rows < 0:
        raise SystemExit("--rows must be >= 0")
    if args.reference_audit_every < 0:
        raise SystemExit("--reference-audit-every must be >= 0")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("shard_*.npz")) + [out / "manifest.json"]
    existing = [p for p in existing if p.exists()]
    if args.overwrite:
        for p in existing:
            p.unlink()
    elif existing and not args.resume:
        raise SystemExit(
            f"Output already contains Expert-v1 sidecar data: {out}\n"
            "Use --resume (default) or --overwrite."
        )

    paths = shard_paths(args.v0_cache)
    total_available = int(v0_manifest.get("built_rows", 0))
    requested = total_available if args.rows == 0 else min(args.rows, total_available)

    print("=" * 104)
    print("TETR.IO EXPERT V1 — COUNTERFACTUAL HOLD-BRANCH SIDECAR")
    print("=" * 104)
    print(f"V0 cache     : {args.v0_cache}")
    print(f"Rows         : {requested:,}/{total_available:,}")
    print(f"Backend      : {args.backend}")
    print(f"Workers      : {args.workers}")
    print(f"Ref audit    : every {args.reference_audit_every:,} row(s)")
    print(f"Resume       : {args.resume}")
    print(f"Reuse dir    : {args.reuse_dir}")
    print()

    global_row = 0
    built_rows = 0
    fresh_rows = 0
    reused_rows = 0
    total_cf_candidates = 0
    total_selected_candidates = 0
    reference_audits = 0
    reference_fallbacks = 0
    failures = []
    shard_stats = []
    started = time.perf_counter()

    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(max_workers=args.workers)

    try:
        for v0_path in paths:
            if built_rows >= requested:
                break

            v0 = load_shard(v0_path)
            available = int(v0["expert_index"].shape[0])
            take = min(available, requested - built_rows)
            if take <= 0:
                break

            out_path = out / v0_path.name
            reused_stats, reuse_kind = _try_reuse_sidecar(
                output_path=out_path,
                reuse_dir=args.reuse_dir,
                v0=v0,
                expected_rows=take,
                resume=args.resume,
            )
            if reused_stats is not None:
                shard_stats.append(
                    {
                        "file": out_path.name,
                        **reused_stats,
                        "reused": True,
                        "reuse_kind": reuse_kind,
                    }
                )
                built_rows += take
                reused_rows += take
                global_row += take
                total_cf_candidates += reused_stats["counterfactual_candidates"]
                total_selected_candidates += reused_stats["selected_candidates"]
                print(
                    f"  {reuse_kind.upper()} {out_path.name}: "
                    f"rows={take:,} built={built_rows:,}/{requested:,} "
                    f"cf_candidates={total_cf_candidates:,}"
                )
                continue

            work = []
            for i in range(take):
                do_audit = (
                    args.backend == "fast"
                    and args.reference_audit_every > 0
                    and global_row % args.reference_audit_every == 0
                )
                work.append(
                    (
                        v0["state_board_packed"][i],
                        int(v0["state_active"][i]),
                        int(v0["state_hold"][i]),
                        v0["state_preview"][i],
                        int(v0["use_hold"][i]),
                        args.backend,
                        args.fast_max_states,
                        args.reference_max_states,
                        do_audit,
                    )
                )
                global_row += 1

            if pool is None:
                results = list(map(_build_one, work))
            else:
                results = list(
                    pool.map(
                        _build_one,
                        work,
                        chunksize=max(1, len(work) // (args.workers * 16)),
                    )
                )

            bad = [
                (i, r)
                for i, r in enumerate(results)
                if not r.ok
            ]
            if bad:
                for i, r in bad[:20]:
                    failures.append(
                        {
                            "shard": v0_path.name,
                            "row": i,
                            "game_id": int(v0["game_id"][i]),
                            "subframe": int(v0["subframe"][i]),
                            "error": r.error,
                        }
                    )
                break

            reference_audits += sum(int(r.audited_reference) for r in results)
            reference_fallbacks += sum(
                int(r.used_reference_fallback) for r in results
            )

            stats = _write_sidecar(
                out_path,
                v0=v0,
                row_count=take,
                results=results,
            )
            shard_stats.append({"file": out_path.name, **stats})
            built_rows += take
            fresh_rows += take
            total_cf_candidates += stats["counterfactual_candidates"]
            total_selected_candidates += stats["selected_candidates"]

            elapsed = time.perf_counter() - started
            fresh_rate = fresh_rows / max(elapsed, 1e-9)
            print(
                f"  built={built_rows:,}/{requested:,} "
                f"fresh={fresh_rows:,} reused={reused_rows:,} "
                f"cf_candidates={total_cf_candidates:,} "
                f"audits(new)={reference_audits} "
                f"fallbacks(new)={reference_fallbacks} "
                f"fresh_rate={fresh_rate:.1f} rows/s"
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    elapsed = time.perf_counter() - started
    status = "PASS" if not failures and built_rows == requested else "FAIL"
    manifest = {
        "format": "tetrio_expert_v1_counterfactual_sidecar_v1",
        "status": status,
        "v0_cache": str(args.v0_cache),
        "v0_manifest_sha256": _manifest_hash(manifest_path),
        "available_v0_rows": total_available,
        "requested_rows": args.rows,
        "built_rows": built_rows,
        "fresh_rows": fresh_rows,
        "reused_rows": reused_rows,
        "reuse_dir": None if args.reuse_dir is None else str(args.reuse_dir),
        "selected_candidates_reused": total_selected_candidates,
        "counterfactual_candidates": total_cf_candidates,
        "joint_mean_candidates": (
            None if built_rows == 0
            else (total_selected_candidates + total_cf_candidates) / built_rows
        ),
        "backend": args.backend,
        "fast_max_states": args.fast_max_states,
        "reference_max_states": args.reference_max_states,
        "reference_audit_every": args.reference_audit_every,
        "reference_audits": reference_audits,
        "reference_fallbacks": reference_fallbacks,
        "seconds": elapsed,
        "rows_per_second": (
            None if elapsed == 0 else fresh_rows / elapsed
        ),
        "rows_per_second_scope": "fresh rows only",
        "candidate_contract": (
            "V0 expert-selected branch reused + opposite Hold branch sidecar; "
            "both branches compete jointly in Expert-v1"
        ),
        "structural_metadata": (
            "holes_before + holes for every selected/counterfactual candidate"
        ),
        "failures": failures,
        "shards": shard_stats,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Built rows     : {built_rows:,}")
    print(f"Joint mean K   : {manifest['joint_mean_candidates']:.2f}" if built_rows else "Joint mean K   : n/a")
    print(f"Ref audits     : {reference_audits}")
    print(f"Ref fallbacks  : {reference_fallbacks}")
    print(f"Time           : {elapsed:.2f}s")
    print(f"Manifest       : {out / 'manifest.json'}")
    print(f"Result         : {status}")
    if failures:
        print("Failures:")
        for item in failures[:20]:
            print(
                f"  {item['shard']} row={item['row']} "
                f"{item['game_id']}/{item['subframe']} {item['error']}"
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
