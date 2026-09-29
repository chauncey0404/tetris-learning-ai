from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from tetrio.future.features import FEATURE_NAMES
from tetrio.future.lookahead import (
    FutureCandidateInput,
    FutureFeatureConfig,
    build_row_future_features,
    packed_board_to_array,
)
from tetrio.network.cache_v1 import (
    compact_batches_from_pair,
    load_pair,
    sidecar_paths,
)
from tetrio.network.checkpoint import load_expert_v1
from tetrio.network.encoding import (
    EMPTY_PIECE_ID,
    PIECES,
    torch_dense_candidate_batch,
    torch_dense_state_batch,
)


ID_TO_PIECE = {i: p for i, p in enumerate(PIECES)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build Expert-v1.1 Top-K future-feature sidecar from the frozen "
            "Expert-v1 joint HOLD/NO-HOLD scorer."
        )
    )
    p.add_argument(
        "--v0-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\train_full_fast_s8192"),
    )
    p.add_argument("--cf-cache", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_joint_100k.pt"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--rows",
        type=int,
        default=0,
        help="0 = all rows available in the counterfactual cache.",
    )
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse already-completed future shards in --output-dir. This is "
            "safe across the old exact-T cache because exact-only columns are "
            "neutralized by the V1.1 model."
        ),
    )
    p.add_argument(
        "--reuse-dir",
        type=Path,
        default=None,
        help=(
            "Optional smaller completed future cache used to seed a larger "
            "build. Complete row-identity-matching shards are copied; a partial "
            "last shard is automatically rejected and rebuilt."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _piece_name(pid: int) -> str | None:
    if int(pid) == EMPTY_PIECE_ID:
        return None
    return ID_TO_PIECE[int(pid)]


def select_inference_shortlist(
    scores: np.ndarray,
    use_hold: np.ndarray,
    *,
    top_overall: int,
    top_per_branch: int,
) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32)
    use_hold = np.asarray(use_hold, dtype=bool)
    selected: set[int] = set()

    order = np.argsort(-scores, kind="stable")
    selected.update(int(i) for i in order[: max(0, int(top_overall))])

    for branch in (False, True):
        idx = np.flatnonzero(use_hold == branch)
        if idx.size:
            branch_order = idx[np.argsort(-scores[idx], kind="stable")]
            selected.update(
                int(i)
                for i in branch_order[: max(0, int(top_per_branch))]
            )

    return sorted(selected, key=lambda i: (-float(scores[i]), int(i)))


def _score_compact_batch(model, batch, device: torch.device) -> np.ndarray:
    def to(a):
        t = torch.from_numpy(a)
        return t.to(
            device=device,
            non_blocking=device.type == "cuda",
        )

    state = torch_dense_state_batch(
        to(batch.state_board_packed),
        to(batch.state_active),
        to(batch.state_hold),
        to(batch.state_preview),
    )
    candidates = torch_dense_candidate_batch(
        to(batch.candidate_board_packed),
        to(batch.candidate_piece),
        to(batch.candidate_rotation),
        to(batch.candidate_x),
        to(batch.candidate_y),
        to(batch.candidate_use_hold),
        to(batch.candidate_lines),
    )
    owner = to(batch.candidate_owner).long()

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=(
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        enabled=device.type == "cuda",
    ):
        scores = model.forward_flat(
            state=state,
            candidates=candidates,
            candidate_owner=owner,
        )
    return scores.float().cpu().numpy()


def _row_task(task):
    (
        board_before_packed,
        active,
        hold,
        preview,
        candidate_payload,
        config_dict,
    ) = task

    board_before = packed_board_to_array(board_before_packed)
    candidates = tuple(
        FutureCandidateInput(
            board_after=packed_board_to_array(item["board"]),
            piece=item["piece"],
            rotation=item["rotation"],
            x=item["x"],
            y=item["y"],
            use_hold=item["use_hold"],
            lines=item["lines"],
        )
        for item in candidate_payload
    )
    config = FutureFeatureConfig(**config_dict)
    return build_row_future_features(
        board_before=board_before,
        active=active,
        hold=hold,
        preview=tuple(preview),
        candidates=candidates,
        config=config,
    )



def _existing_future_shard_stats(
    path: Path,
    *,
    expected_rows: int,
    expected_game_id: np.ndarray | None = None,
    expected_subframe: np.ndarray | None = None,
) -> tuple[int, int, int] | None:
    """Return (rows, candidates, recall_hits) for a complete reusable shard."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as d:
            required = {
                "candidate_offsets",
                "base_score",
                "features",
                "expert_local",
                "expert_in_shortlist",
            }
            if not required.issubset(d.files):
                return None
            rows = int(d["expert_local"].shape[0])
            if rows != int(expected_rows):
                return None
            candidates = int(d["base_score"].shape[0])
            recall_hits = int(np.asarray(d["expert_in_shortlist"], dtype=np.uint8).sum())
            if int(d["candidate_offsets"][-1]) != candidates:
                return None
            if expected_game_id is not None:
                if "game_id" not in d.files or not np.array_equal(
                    np.asarray(d["game_id"], dtype=np.int64),
                    np.asarray(expected_game_id[:rows], dtype=np.int64),
                ):
                    return None
            if expected_subframe is not None:
                if "subframe" not in d.files or not np.array_equal(
                    np.asarray(d["subframe"], dtype=np.int64),
                    np.asarray(expected_subframe[:rows], dtype=np.int64),
                ):
                    return None
            return rows, candidates, recall_hits
    except Exception:
        return None



def _try_reuse_future_shard(
    *,
    output_path: Path,
    reuse_dir: Path | None,
    expected_rows: int,
    expected_game_id: np.ndarray,
    expected_subframe: np.ndarray,
    resume: bool,
) -> tuple[tuple[int, int, int] | None, str | None]:
    if resume:
        stats = _existing_future_shard_stats(
            output_path,
            expected_rows=expected_rows,
            expected_game_id=expected_game_id,
            expected_subframe=expected_subframe,
        )
        if stats is not None:
            return stats, "resume"

    if reuse_dir is not None:
        source = reuse_dir / output_path.name
        stats = _existing_future_shard_stats(
            source,
            expected_rows=expected_rows,
            expected_game_id=expected_game_id,
            expected_subframe=expected_subframe,
        )
        if stats is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_path)
            return stats, "reuse-dir"

    return None, None


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.rows < 0:
        raise SystemExit("--rows must be >= 0")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("shard_*.npz")) + [out / "manifest.json"]
    existing = [p for p in existing if p.exists()]
    if args.overwrite:
        for p in existing:
            p.unlink()
    elif existing and not args.resume:
        raise SystemExit(
            f"Output already contains files: {out}\n"
            "Use --resume (default) or --overwrite."
        )

    device = torch.device(args.device)
    model, ckpt = load_expert_v1(args.checkpoint, device=device)
    model.eval()

    cf_paths = sidecar_paths(args.cf_cache)
    available = 0
    for p in cf_paths:
        with np.load(p, allow_pickle=False) as d:
            available += int(d["holes_before"].shape[0])
    requested = available if args.rows == 0 else min(args.rows, available)

    config = FutureFeatureConfig(
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        tactical_preview_depth=2,
        exact_immediate_t=False,
    )
    config_dict = {
        "fast_max_states": config.fast_max_states,
        "reference_max_states": config.reference_max_states,
        "tactical_preview_depth": config.tactical_preview_depth,
        "exact_immediate_t": config.exact_immediate_t,
    }

    print("=" * 108)
    print("TETR.IO EXPERT V1.1 — FUTURE FEATURE CACHE")
    print("=" * 108)
    print(f"V1 checkpoint : {args.checkpoint} epoch={ckpt.get('epoch')}")
    print(f"V0 cache      : {args.v0_cache}")
    print(f"CF cache      : {args.cf_cache}")
    print(f"Rows          : {requested:,}/{available:,}")
    print(f"Shortlist     : top{args.top_overall} overall + top{args.top_per_branch}/branch")
    print(f"Workers       : {args.workers}")
    print(f"Device        : {device}")
    print("Tactical mode : FAST PROXY (no per-candidate reference BFS)")
    print(f"Resume        : {args.resume}")
    print(f"Reuse dir     : {args.reuse_dir}")
    print()

    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    built = 0
    newly_computed = 0
    total_candidates = 0
    recall_hits = 0
    shard_manifest = []
    started = time.perf_counter()

    try:
        for cf_path in cf_paths:
            if built >= requested:
                break

            # Read only the CF row count first so a completed shard can be
            # resumed without loading the much larger paired V0 cache.
            with np.load(cf_path, allow_pickle=False) as cf_probe:
                cf_rows_available = int(cf_probe["holes_before"].shape[0])
                cf_game_id = np.asarray(cf_probe["game_id"], dtype=np.int64)
                cf_subframe = np.asarray(cf_probe["subframe"], dtype=np.int64)

            shard_rows = min(
                cf_rows_available,
                requested - built,
            )
            if shard_rows <= 0:
                break

            out_path = out / cf_path.name
            reused, reuse_kind = _try_reuse_future_shard(
                output_path=out_path,
                reuse_dir=args.reuse_dir,
                expected_rows=shard_rows,
                expected_game_id=cf_game_id,
                expected_subframe=cf_subframe,
                resume=args.resume,
            )
            if reused is not None:
                rows_here, cand_here, recall_here = reused
                built += rows_here
                total_candidates += cand_here
                recall_hits += recall_here
                print(
                    f"  {reuse_kind.upper()} {cf_path.name}: rows={rows_here:,} "
                    f"candidates={cand_here:,} | total={built:,}/{requested:,} "
                    f"recall={recall_hits/max(1,built):.4f} "
                    "rate=pending ETA=pending"
                )
                shard_manifest.append(
                    {
                        "file": cf_path.name,
                        "rows": rows_here,
                        "stored_candidates": cand_here,
                        "resumed": True,
                        "reuse_kind": reuse_kind,
                    }
                )
                continue

            v0, cf = load_pair(args.v0_cache, cf_path)

            all_base = []
            all_features = []
            all_hold = []
            all_inf = []
            offsets = [0]
            expert_local_out = []
            expert_hold_out = []
            expert_in_shortlist = []
            game_ids = []
            subframes = []

            row_base = 0
            for batch in compact_batches_from_pair(
                v0=v0,
                cf=cf,
                batch_size=args.batch_size,
                rng=None,
            ):
                if row_base >= shard_rows:
                    break

                batch_rows = min(
                    len(batch.expert_index),
                    shard_rows - row_base,
                )
                if batch_rows <= 0:
                    break

                flat_scores = _score_compact_batch(model, batch, device)
                owner = np.asarray(batch.candidate_owner)
                local = np.asarray(batch.candidate_local)

                tasks = []
                row_meta = []

                for b in range(batch_rows):
                    flat_idx = np.flatnonzero(owner == b)
                    flat_idx = flat_idx[np.argsort(local[flat_idx], kind="stable")]

                    scores = flat_scores[flat_idx]
                    holds = np.asarray(batch.candidate_use_hold[flat_idx], dtype=bool)
                    inference = select_inference_shortlist(
                        scores,
                        holds,
                        top_overall=args.top_overall,
                        top_per_branch=args.top_per_branch,
                    )

                    expert_joint = int(batch.expert_index[b])
                    stored = list(inference)
                    forced = expert_joint not in stored
                    if forced:
                        stored.append(expert_joint)

                    inference_set = set(inference)
                    expert_stored = stored.index(expert_joint)

                    payload = []
                    stored_scores = []
                    stored_holds = []
                    stored_inf = []

                    local_to_flat = {
                        int(local[fi]): int(fi)
                        for fi in flat_idx
                    }

                    for joint_local in stored:
                        fi = local_to_flat[int(joint_local)]
                        pid = int(batch.candidate_piece[fi])
                        payload.append(
                            {
                                "board": np.asarray(
                                    batch.candidate_board_packed[fi],
                                    dtype=np.uint8,
                                ),
                                "piece": ID_TO_PIECE[pid],
                                "rotation": int(batch.candidate_rotation[fi]),
                                "x": int(batch.candidate_x[fi]),
                                "y": int(batch.candidate_y[fi]),
                                "use_hold": bool(batch.candidate_use_hold[fi]),
                                "lines": int(batch.candidate_lines[fi]),
                            }
                        )
                        stored_scores.append(float(flat_scores[fi]))
                        stored_holds.append(int(batch.candidate_use_hold[fi]))
                        stored_inf.append(int(joint_local in inference_set))

                    active = _piece_name(int(batch.state_active[b]))
                    hold = _piece_name(int(batch.state_hold[b]))
                    preview = tuple(
                        ID_TO_PIECE[int(x)]
                        for x in batch.state_preview[b]
                        if int(x) != EMPTY_PIECE_ID
                    )
                    if active is None:
                        raise RuntimeError("empty active piece in V1.1 cache")

                    tasks.append(
                        (
                            np.asarray(batch.state_board_packed[b], dtype=np.uint8),
                            active,
                            hold,
                            preview,
                            payload,
                            config_dict,
                        )
                    )
                    row_meta.append(
                        (
                            stored_scores,
                            stored_holds,
                            stored_inf,
                            expert_stored,
                            int(batch.expert_use_hold[b]),
                            not forced,
                            row_base + b,
                        )
                    )

                if pool is None:
                    features_list = list(map(_row_task, tasks))
                else:
                    features_list = list(
                        pool.map(
                            _row_task,
                            tasks,
                            chunksize=max(1, len(tasks) // max(1, args.workers * 4)),
                        )
                    )

                for features, meta in zip(features_list, row_meta):
                    (
                        stored_scores,
                        stored_holds,
                        stored_inf,
                        expert_stored,
                        expert_hold,
                        in_shortlist,
                        shard_row,
                    ) = meta
                    n = len(stored_scores)
                    all_base.extend(stored_scores)
                    all_hold.extend(stored_holds)
                    all_inf.extend(stored_inf)
                    all_features.append(features)
                    offsets.append(offsets[-1] + n)
                    expert_local_out.append(expert_stored)
                    expert_hold_out.append(expert_hold)
                    expert_in_shortlist.append(int(in_shortlist))
                    recall_hits += int(in_shortlist)
                    game_ids.append(int(v0["game_id"][shard_row]))
                    subframes.append(int(v0["subframe"][shard_row]))

                row_base += batch_rows

                processed = built + row_base
                computed_now = newly_computed + row_base
                elapsed = time.perf_counter() - started
                rate = computed_now / max(elapsed, 1e-9)
                eta = (requested - processed) / max(rate, 1e-9)
                print(
                    f"    {cf_path.name} {row_base:,}/{shard_rows:,} | "
                    f"total={processed:,}/{requested:,} "
                    f"rate={rate:.1f} rows/s ETA={_fmt_eta(eta)}",
                    flush=True,
                )

            features_flat = (
                np.concatenate(all_features, axis=0)
                if all_features
                else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
            )

            np.savez(
                out_path,
                candidate_offsets=np.asarray(offsets, dtype=np.int64),
                base_score=np.asarray(all_base, dtype=np.float32),
                features=np.asarray(features_flat, dtype=np.float32),
                candidate_use_hold=np.asarray(all_hold, dtype=np.uint8),
                inference_mask=np.asarray(all_inf, dtype=np.uint8),
                expert_local=np.asarray(expert_local_out, dtype=np.int16),
                expert_use_hold=np.asarray(expert_hold_out, dtype=np.uint8),
                expert_in_shortlist=np.asarray(expert_in_shortlist, dtype=np.uint8),
                game_id=np.asarray(game_ids, dtype=np.int64),
                subframe=np.asarray(subframes, dtype=np.int64),
            )

            rows_here = len(expert_local_out)
            cand_here = len(all_base)
            built += rows_here
            newly_computed += rows_here
            total_candidates += cand_here
            elapsed = time.perf_counter() - started
            print(
                f"  {cf_path.name}: rows={rows_here:,} candidates={cand_here:,} | "
                f"total={built:,}/{requested:,} recall={recall_hits/max(1,built):.4f} "
                f"rate={built/max(elapsed,1e-9):.1f} rows/s"
            )
            shard_manifest.append(
                {
                    "file": cf_path.name,
                    "rows": rows_here,
                    "stored_candidates": cand_here,
                }
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    elapsed = time.perf_counter() - started
    manifest = {
        "format": "tetrio_expert_v1_1_future_cache_v1",
        "status": "PASS" if built == requested else "FAIL",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "v0_cache": str(args.v0_cache),
        "cf_cache": str(args.cf_cache),
        "available_rows": available,
        "requested_rows": args.rows,
        "built_rows": built,
        "stored_candidates": total_candidates,
        "mean_stored_candidates": (
            total_candidates / built if built else None
        ),
        "shortlist_recall": (
            recall_hits / built if built else None
        ),
        "top_overall": args.top_overall,
        "top_per_branch": args.top_per_branch,
        "feature_names": list(FEATURE_NAMES),
        "feature_config": config_dict,
        "tactical_mode": "fast_proxy_bulk_v2",
        "model_uses_exact_t_features": False,
        "resume_enabled": args.resume,
        "reuse_dir": None if args.reuse_dir is None else str(args.reuse_dir),
        "seconds": elapsed,
        "rows_per_second": built / elapsed if elapsed else None,
        "shards": shard_manifest,
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Built rows       : {built:,}")
    print(f"Shortlist recall : {manifest['shortlist_recall']:.4f}" if built else "Shortlist recall : n/a")
    print(f"Mean stored K    : {manifest['mean_stored_candidates']:.2f}" if built else "Mean stored K    : n/a")
    print(f"Elapsed          : {elapsed:.2f}s")
    print(f"Manifest         : {out/'manifest.json'}")
    print(f"Result           : {manifest['status']}")
    if manifest["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
