from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from tetris_ai.learning.early_stopping import EarlyStoppingTracker
from tetris_ai.learning.listwise import (
    candidate_ranking_metrics,
    masked_candidate_cross_entropy,
)
from tetrio.network.cache_prepare import (
    ExpertV0CacheBuildSpec,
    ensure_expert_v0_cache,
)
from tetrio.network.cache import (
    ExpertV0Batch,
    ExpertV0CompactBatch,
    batches_from_shard,
    compact_batches_from_shard,
    load_shard,
    shard_paths,
)
from tetrio.network.encoding import (
    torch_dense_candidate_batch,
    torch_dense_state_batch,
)
from tetrio.network.model import TetrioExpertV0Network


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TETR.IO Expert v0 candidate ranker.")
    p.add_argument(
        "--train-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\train_full_fast_s8192"),
        help="Prepared train cache. Reused when PASS; auto-built when missing.",
    )
    p.add_argument(
        "--val-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\val_10k"),
        help="Prepared validation cache. Reused when PASS; auto-built when missing.",
    )
    p.add_argument(
        "--train-source",
        type=Path,
        default=Path(r"data\tetrio\expert\top_players_s1_train.parquet"),
        help="Source parquet used only when the train cache must be built.",
    )
    p.add_argument(
        "--val-source",
        type=Path,
        default=Path(r"data\tetrio\expert\top_players_s1_val.parquet"),
        help="Source parquet used only when the validation cache must be built.",
    )
    p.add_argument(
        "--train-cache-rows",
        type=int,
        default=0,
        help="Rows to auto-build for train cache; 0 means all eligible source rows.",
    )
    p.add_argument(
        "--val-cache-rows",
        type=int,
        default=10_000,
        help="Rows to auto-build for validation cache.",
    )
    p.add_argument(
        "--cache-backend",
        choices=("reference", "fast"),
        default="fast",
        help="Backend used only when a missing/incomplete cache must be built.",
    )
    p.add_argument(
        "--cache-workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 2)),
    )
    p.add_argument(
        "--cache-shard-size",
        type=int,
        default=8192,
    )
    p.add_argument(
        "--cache-seed",
        type=int,
        default=20260906,
    )
    p.add_argument(
        "--cache-fast-max-states",
        type=int,
        default=10_000,
    )
    p.add_argument(
        "--auto-build-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before training: reuse PASS caches; otherwise build/rebuild them "
            "and continue automatically."
        ),
    )
    p.add_argument("--output", type=Path, default=Path(r"models\tetrio_expert_v0.pt"))
    p.add_argument("--metrics", type=Path, default=Path(r"artifacts\tetrio\expert_v0_training.json"))
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum epochs. Early stopping can finish the run sooner.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hold-loss-weight", type=float, default=0.25)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--pipeline",
        choices=("dense", "compact_gpu"),
        default="compact_gpu",
        help=(
            "compact_gpu transfers packed cache tensors and expands them on GPU; "
            "dense preserves the old CPU-dense path."
        ),
    )
    p.add_argument(
        "--prefetch-shards",
        type=int,
        default=4,
        help="Background shard preparation threads. 0 disables prefetch.",
    )
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin compact/dense CPU tensors before CUDA copies.",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help=(
            "Stop after this many consecutive validation epochs without a "
            "meaningful Val Top-1 improvement. 0 disables early stopping."
        ),
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.001,
        help=(
            "Minimum Val Top-1 gain required to reset patience. "
            "0.001 = 0.1 percentage point."
        ),
    )
    p.add_argument(
        "--early-stop-warmup-epochs",
        type=int,
        default=3,
        help="Do not allow early stopping before this epoch.",
    )
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_from_numpy(
    array: np.ndarray,
    *,
    device: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if device.type == "cuda" and pin_memory:
        tensor = tensor.pin_memory()
    return tensor.to(
        device=device,
        non_blocking=(device.type == "cuda" and pin_memory),
    )


def to_device_dense(
    batch: ExpertV0Batch,
    device: torch.device,
    pin_memory: bool,
):
    state = _tensor_from_numpy(batch.state, device=device, pin_memory=pin_memory)
    candidates = _tensor_from_numpy(batch.candidates, device=device, pin_memory=pin_memory)
    mask = _tensor_from_numpy(batch.candidate_mask, device=device, pin_memory=pin_memory)
    target = _tensor_from_numpy(batch.expert_index, device=device, pin_memory=pin_memory)
    hold = _tensor_from_numpy(batch.use_hold, device=device, pin_memory=pin_memory)
    return state, candidates, mask, target, hold


def to_device_compact(
    batch: ExpertV0CompactBatch,
    device: torch.device,
    pin_memory: bool,
):
    transfer = lambda a: _tensor_from_numpy(a, device=device, pin_memory=pin_memory)
    return {
        "state_board_packed": transfer(batch.state_board_packed),
        "state_active": transfer(batch.state_active),
        "state_hold": transfer(batch.state_hold),
        "state_preview": transfer(batch.state_preview),
        "candidate_board_packed": transfer(batch.candidate_board_packed),
        "candidate_piece": transfer(batch.candidate_piece),
        "candidate_rotation": transfer(batch.candidate_rotation),
        "candidate_x": transfer(batch.candidate_x),
        "candidate_y": transfer(batch.candidate_y),
        "candidate_use_hold": transfer(batch.candidate_use_hold),
        "candidate_lines": transfer(batch.candidate_lines),
        "candidate_owner": transfer(batch.candidate_owner),
        "candidate_local": transfer(batch.candidate_local),
        "candidate_counts": transfer(batch.candidate_counts),
        "expert_index": transfer(batch.expert_index),
        "use_hold": transfer(batch.use_hold),
        "max_candidates": batch.max_candidates,
    }


def cache_rows(cache_dir: Path) -> int:
    total = 0
    for path in shard_paths(cache_dir):
        with np.load(path, allow_pickle=False) as data:
            total += int(data["expert_index"].shape[0])
    return total


def _prepare_shard(
    path: Path,
    *,
    batch_size: int,
    seed: int | None,
    pipeline: str,
):
    data = load_shard(path)
    rng = None if seed is None else np.random.default_rng(seed)
    if pipeline == "compact_gpu":
        return list(
            compact_batches_from_shard(
                data,
                batch_size=batch_size,
                rng=rng,
            )
        )
    return list(
        batches_from_shard(
            data,
            batch_size=batch_size,
            rng=rng,
        )
    )


def _iter_prepared_batches(
    *,
    paths: list[Path],
    batch_size: int,
    training: bool,
    epoch_seed: int,
    pipeline: str,
    prefetch_shards: int,
) -> Iterator[ExpertV0Batch | ExpertV0CompactBatch]:
    seeds = [
        (epoch_seed + 1_000_003 * i) if training else None
        for i in range(len(paths))
    ]

    if prefetch_shards <= 0:
        for path, seed in zip(paths, seeds):
            yield from _prepare_shard(
                path,
                batch_size=batch_size,
                seed=seed,
                pipeline=pipeline,
            )
        return

    worker_count = max(1, int(prefetch_shards))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        pending: deque[Future] = deque()
        next_index = 0

        def submit_one(index: int) -> None:
            pending.append(
                pool.submit(
                    _prepare_shard,
                    paths[index],
                    batch_size=batch_size,
                    seed=seeds[index],
                    pipeline=pipeline,
                )
            )

        while next_index < len(paths) and len(pending) < worker_count:
            submit_one(next_index)
            next_index += 1

        while pending:
            batches = pending.popleft().result()
            if next_index < len(paths):
                submit_one(next_index)
                next_index += 1
            yield from batches


def _compact_forward(
    *,
    model,
    batch: ExpertV0CompactBatch,
    device: torch.device,
    pin_memory: bool,
):
    packed = to_device_compact(batch, device, pin_memory)

    state = torch_dense_state_batch(
        packed["state_board_packed"],
        packed["state_active"],
        packed["state_hold"],
        packed["state_preview"],
    )
    candidates = torch_dense_candidate_batch(
        packed["candidate_board_packed"],
        packed["candidate_piece"],
        packed["candidate_rotation"],
        packed["candidate_x"],
        packed["candidate_y"],
        packed["candidate_use_hold"],
        packed["candidate_lines"],
    )

    flat_scores, hold_logit = model.forward_flat(
        state=state,
        candidates=candidates,
        candidate_owner=packed["candidate_owner"],
    )

    b = state.shape[0]
    max_k = int(packed["max_candidates"])
    scores = flat_scores.new_full(
        (b, max_k),
        torch.finfo(flat_scores.dtype).min,
    )
    scores = scores.index_put(
        (
            packed["candidate_owner"].to(dtype=torch.long),
            packed["candidate_local"].to(dtype=torch.long),
        ),
        flat_scores,
    )

    arange_k = torch.arange(max_k, device=device).unsqueeze(0)
    mask = arange_k < packed["candidate_counts"].to(dtype=torch.long).unsqueeze(1)

    return (
        scores,
        hold_logit,
        mask,
        packed["expert_index"].to(dtype=torch.long),
        packed["use_hold"].to(dtype=torch.float32),
    )


def run_epoch(
    *,
    model,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    optimizer,
    scaler,
    amp_dtype,
    training: bool,
    epoch_seed: int,
    hold_loss_weight: float,
    label_smoothing: float,
    grad_clip: float,
    pipeline: str,
    prefetch_shards: int,
    pin_memory: bool,
) -> dict[str, float]:
    paths = shard_paths(cache_dir)
    rng = np.random.default_rng(epoch_seed) if training else None
    if training:
        rng.shuffle(paths)
        model.train()
    else:
        model.eval()

    sums = {
        "loss": 0.0,
        "candidate_loss": 0.0,
        "hold_loss": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "mrr": 0.0,
        "hold_acc": 0.0,
        "rows": 0.0,
    }

    started = time.perf_counter()
    batches = _iter_prepared_batches(
        paths=paths,
        batch_size=batch_size,
        training=training,
        epoch_seed=epoch_seed,
        pipeline=pipeline,
        prefetch_shards=prefetch_shards,
    )

    for batch in batches:
        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            if pipeline == "compact_gpu":
                scores, hold_logit, mask, target, hold = _compact_forward(
                    model=model,
                    batch=batch,
                    device=device,
                    pin_memory=pin_memory,
                )
            else:
                state, candidates, mask, target, hold = to_device_dense(
                    batch,
                    device,
                    pin_memory,
                )
                scores, hold_logit = model(state=state, candidates=candidates)

            b = target.shape[0]
            candidate_loss = masked_candidate_cross_entropy(
                scores,
                target,
                mask,
                label_smoothing=label_smoothing,
            )
            hold_loss = F.binary_cross_entropy_with_logits(hold_logit, hold)
            loss = candidate_loss + hold_loss_weight * hold_loss

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        with torch.no_grad():
            metrics = candidate_ranking_metrics(scores, target, mask)
            hold_acc = ((hold_logit >= 0) == (hold >= 0.5)).float().mean()

        sums["loss"] += float(loss.detach()) * b
        sums["candidate_loss"] += float(candidate_loss.detach()) * b
        sums["hold_loss"] += float(hold_loss.detach()) * b
        sums["top1"] += float(metrics["top1"]) * b
        sums["top3"] += float(metrics["top3"]) * b
        sums["mrr"] += float(metrics["mrr"]) * b
        sums["hold_acc"] += float(hold_acc) * b
        sums["rows"] += b

    rows = sums.pop("rows")
    elapsed = time.perf_counter() - started
    result = {key: value / rows for key, value in sums.items()}
    result["rows_per_second"] = rows / elapsed if elapsed else 0.0
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    if args.prefetch_shards < 0:
        raise SystemExit("--prefetch-shards must be >= 0")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.early_stop_patience < 0:
        raise SystemExit("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        raise SystemExit("--early-stop-min-delta must be >= 0")
    if args.early_stop_warmup_epochs < 0:
        raise SystemExit("--early-stop-warmup-epochs must be >= 0")

    # Cache preparation is part of the formal training entry point:
    # completed PASS caches are reused; missing/partial caches are built first.
    ensure_expert_v0_cache(
        ExpertV0CacheBuildSpec(
            cache_dir=args.train_cache,
            source=args.train_source,
            rows=args.train_cache_rows,
            backend=args.cache_backend,
            workers=args.cache_workers,
            shard_size=args.cache_shard_size,
            seed=args.cache_seed,
            fast_max_states=args.cache_fast_max_states,
        ),
        auto_build=args.auto_build_cache,
        label="train cache",
    )
    ensure_expert_v0_cache(
        ExpertV0CacheBuildSpec(
            cache_dir=args.val_cache,
            source=args.val_source,
            rows=args.val_cache_rows,
            backend=args.cache_backend,
            workers=args.cache_workers,
            shard_size=args.cache_shard_size,
            seed=args.cache_seed,
            fast_max_states=args.cache_fast_max_states,
        ),
        auto_build=args.auto_build_cache,
        label="validation cache",
    )

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    train_rows = cache_rows(args.train_cache)
    val_rows = cache_rows(args.val_cache)
    if train_rows == 0 or val_rows == 0:
        raise SystemExit("Train/validation cache is empty")

    model = TetrioExpertV0Network().to(device)
    model_for_save = model
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        scaler = None
        amp_name = "bf16"
    elif device.type == "cuda":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        amp_name = "fp16"
    else:
        amp_dtype = torch.float32
        scaler = None
        amp_name = "fp32"

    early_stop = EarlyStoppingTracker(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="max",
    )

    print("=" * 96)
    print("TETR.IO EXPERT V0 — IMITATION PRETRAINING")
    print("=" * 96)
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"AMP          : {amp_name}")
    print(f"Train rows   : {train_rows:,}")
    print(f"Val rows     : {val_rows:,}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Max epochs   : {args.epochs}")
    print(f"Pipeline     : {args.pipeline}")
    print(f"Prefetch     : {args.prefetch_shards} shard(s)")
    print(f"Pin memory   : {args.pin_memory}")
    print(f"Compile      : {args.compile}")
    print(
        f"Auto cache   : {args.auto_build_cache} "
        f"(backend={args.cache_backend}, shard={args.cache_shard_size})"
    )
    if early_stop.enabled:
        print(
            f"Early stop   : patience={args.early_stop_patience}, "
            f"min_delta={args.early_stop_min_delta:.4f}, "
            f"warmup={args.early_stop_warmup_epochs}"
        )
    else:
        print("Early stop   : disabled")
    print()

    history: list[dict[str, Any]] = []
    best_top1 = -1.0
    best_epoch = 0
    stopped_early = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    started = time.perf_counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train = run_epoch(
            model=model,
            cache_dir=args.train_cache,
            device=device,
            batch_size=args.batch_size,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            training=True,
            epoch_seed=args.seed + epoch,
            hold_loss_weight=args.hold_loss_weight,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            pipeline=args.pipeline,
            prefetch_shards=args.prefetch_shards,
            pin_memory=args.pin_memory,
        )
        with torch.inference_mode():
            val = run_epoch(
                model=model,
                cache_dir=args.val_cache,
                device=device,
                batch_size=args.batch_size,
                optimizer=None,
                scaler=None,
                amp_dtype=amp_dtype,
                training=False,
                epoch_seed=args.seed,
                hold_loss_weight=args.hold_loss_weight,
                label_smoothing=0.0,
                grad_clip=args.grad_clip,
                pipeline=args.pipeline,
                prefetch_shards=args.prefetch_shards,
                pin_memory=args.pin_memory,
            )
        elapsed = time.perf_counter() - t0
        row = {"epoch": epoch, "seconds": elapsed, "train": train, "val": val}
        history.append(row)

        # Save the numerically best checkpoint even if the improvement is too
        # small to reset early-stop patience.
        if val["top1"] > best_top1:
            best_top1 = val["top1"]
            best_epoch = epoch
            torch.save(
                {
                    "format": "tetrio_expert_v0",
                    "epoch": epoch,
                    "model_state_dict": model_for_save.state_dict(),
                    "config": vars(args),
                    "metrics": row,
                    "state_contract": "400 board + active + hold + first 5 preview",
                    "candidate_contract": (
                        "expert-selected hold branch; unique reachable landing geometry; "
                        "after-board + placement metadata"
                    ),
                },
                args.output,
            )

        should_stop = False
        if early_stop.enabled:
            if epoch < args.early_stop_warmup_epochs:
                # Warmup observations do not consume patience.  Anchor the
                # tracker to the best seen value once warmup completes.
                pass
            elif epoch == args.early_stop_warmup_epochs:
                early_stop.anchor_value = best_top1
                early_stop.anchor_epoch = best_epoch
                early_stop.bad_epochs = 0
            else:
                should_stop = early_stop.update(val["top1"], epoch)

        patience_text = ""
        if early_stop.enabled and epoch >= args.early_stop_warmup_epochs:
            patience_text = (
                f" | patience={early_stop.bad_epochs}/{early_stop.patience}"
            )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train top1={train['top1']:.4f} top3={train['top3']:.4f} "
            f"hold={train['hold_acc']:.4f} {train['rows_per_second']:.0f} rows/s | "
            f"val top1={val['top1']:.4f} top3={val['top3']:.4f} "
            f"mrr={val['mrr']:.4f} hold={val['hold_acc']:.4f} | "
            f"{elapsed:.1f}s{patience_text}"
        )

        if should_stop:
            stopped_early = True
            stop_epoch = epoch
            stop_reason = (
                f"Val Top-1 failed to improve by at least "
                f"{args.early_stop_min_delta:.4f} for "
                f"{args.early_stop_patience} consecutive epoch(s)"
            )
            print()
            print(f"Early stopping at epoch {epoch}: {stop_reason}.")
            print(
                f"Best checkpoint remains epoch {best_epoch} "
                f"(Val Top-1={best_top1:.4f})."
            )
            break

    total_s = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    summary = {
        "format": "tetrio_expert_v0_training",
        "train_rows": train_rows,
        "val_rows": val_rows,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": amp_name,
        "pipeline": args.pipeline,
        "prefetch_shards": args.prefetch_shards,
        "pin_memory": args.pin_memory,
        "peak_cuda_memory_bytes": peak_memory,
        "best_epoch": best_epoch,
        "best_val_top1": best_top1,
        "requested_max_epochs": args.epochs,
        "completed_epochs": len(history),
        "early_stopping": {
            "enabled": early_stop.enabled,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "warmup_epochs": args.early_stop_warmup_epochs,
            "stopped_early": stopped_early,
            "stop_epoch": stop_epoch,
            "stop_reason": stop_reason,
        },
        "seconds": total_s,
        "history": history,
        "checkpoint": str(args.output),
    }
    args.metrics.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Completed     : {len(history)}/{args.epochs} epoch(s)")
    print(f"Best epoch   : {best_epoch}")
    print(f"Best val top1: {best_top1:.4f}")
    print(f"Early stopped: {stopped_early}")
    if peak_memory is not None:
        print(f"Peak CUDA mem: {peak_memory / (1024**3):.2f} GiB")
    print(f"Checkpoint   : {args.output}")
    print(f"Metrics      : {args.metrics}")
    print("Status       : RESEARCH BASELINE (not Champion)")


if __name__ == "__main__":
    main()
