from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from tetrio.network.cache_v1 import (
    ExpertV1CompactBatch,
    compact_batches_from_pair,
    load_pair,
    sidecar_paths,
)
from tetrio.network.checkpoint import load_expert_v0
from tetrio.network.encoding import (
    torch_dense_candidate_batch,
    torch_dense_state_batch,
)
from tetrio.network.model_v1 import TetrioExpertV1Network
from tetris_ai.learning.early_stopping import EarlyStoppingTracker
from tetris_ai.learning.listwise import candidate_ranking_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train TETR.IO Expert-v1: unified HOLD/NO-HOLD candidate ranking "
            "with conservative structural-quality supervision."
        )
    )
    p.add_argument(
        "--train-v0-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\train_full_fast_s8192"),
    )
    p.add_argument("--train-cf-cache", type=Path, required=True)
    p.add_argument(
        "--val-v0-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\val_10k"),
    )
    p.add_argument("--val-cf-cache", type=Path, required=True)
    p.add_argument(
        "--init-v0",
        type=Path,
        default=Path(r"models\tetrio_expert_v0_full.pt"),
        help="Warm-start V1 scorer from the frozen Expert-v0 scorer.",
    )
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_joint_100k.pt"),
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_joint_100k_training.json"),
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help=(
            "Pilot default. Joint candidates roughly double V0 candidate work; "
            "after the pilot, benchmark 4096/8192 for the full corpus."
        ),
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument(
        "--risky-expert-row-weight",
        type=float,
        default=0.35,
        help=(
            "Downweight expert labels that create a new hole even though a "
            "no-new-hole candidate exists. This reduces blind copying of "
            "possible misdrops / missing battle-context labels without "
            "declaring a different move to be the new ground truth."
        ),
    )
    p.add_argument(
        "--structure-loss-weight",
        type=float,
        default=0.03,
        help=(
            "Small margin auxiliary used only when the expert placement itself "
            "does not add holes."
        ),
    )
    p.add_argument("--structure-margin", type=float, default=0.25)

    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--device", default="cuda")
    p.add_argument("--prefetch-shards", type=int, default=4)
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--early-stop-min-delta", type=float, default=0.001)
    p.add_argument("--early-stop-warmup-epochs", type=int, default=3)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to(
    array: np.ndarray,
    *,
    device: torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    t = torch.from_numpy(array)
    if device.type == "cuda" and pin_memory:
        t = t.pin_memory()
    return t.to(
        device=device,
        non_blocking=(device.type == "cuda" and pin_memory),
    )


def cache_rows(sidecar_dir: Path) -> int:
    total = 0
    for p in sidecar_paths(sidecar_dir):
        with np.load(p, allow_pickle=False) as d:
            total += int(d["holes_before"].shape[0])
    return total


def _prepare_pair(
    *,
    v0_dir: Path,
    cf_path: Path,
    batch_size: int,
    seed: int | None,
):
    v0, cf = load_pair(v0_dir, cf_path)
    rng = None if seed is None else np.random.default_rng(seed)
    return list(
        compact_batches_from_pair(
            v0=v0,
            cf=cf,
            batch_size=batch_size,
            rng=rng,
        )
    )


def _iter_batches(
    *,
    v0_dir: Path,
    cf_dir: Path,
    batch_size: int,
    training: bool,
    epoch_seed: int,
    prefetch_shards: int,
) -> Iterator[ExpertV1CompactBatch]:
    paths = sidecar_paths(cf_dir)
    rng = np.random.default_rng(epoch_seed) if training else None
    if training:
        rng.shuffle(paths)

    seeds = [
        epoch_seed + i * 1_000_003 if training else None
        for i in range(len(paths))
    ]

    if prefetch_shards <= 0:
        for p, seed in zip(paths, seeds):
            yield from _prepare_pair(
                v0_dir=v0_dir,
                cf_path=p,
                batch_size=batch_size,
                seed=seed,
            )
        return

    workers = max(1, int(prefetch_shards))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[Future] = deque()
        next_i = 0

        def submit(i: int):
            pending.append(
                pool.submit(
                    _prepare_pair,
                    v0_dir=v0_dir,
                    cf_path=paths[i],
                    batch_size=batch_size,
                    seed=seeds[i],
                )
            )

        while next_i < len(paths) and len(pending) < workers:
            submit(next_i)
            next_i += 1

        while pending:
            batches = pending.popleft().result()
            if next_i < len(paths):
                submit(next_i)
                next_i += 1
            yield from batches


def weighted_masked_cross_entropy(
    scores: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    row_weight: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    mask = mask.bool()
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    logp = F.log_softmax(masked, dim=1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)

    if label_smoothing:
        valid_sum = logp.masked_fill(~mask, 0.0).sum(dim=1)
        valid_n = mask.sum(dim=1).to(scores.dtype)
        smooth = -valid_sum / valid_n
        per_row = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per_row = nll

    denom = row_weight.sum().clamp_min(1e-8)
    return (per_row * row_weight).sum() / denom


def structure_margin_loss(
    *,
    scores: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    holes: torch.Tensor,
    holes_before: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_holes = holes.gather(1, target[:, None]).squeeze(1)
    masked_holes = holes.masked_fill(~mask, 255)
    min_holes = masked_holes.min(dim=1).values

    # Potential label noise/context aliasing:
    # expert itself adds a hole while some legal candidate does not.
    expert_risky = (
        (expert_holes > holes_before)
        & (min_holes <= holes_before)
    )

    # Conservative structural supervision:
    # only when the expert label itself is structurally safe do we add a
    # margin against candidates that create new holes.
    safe_expert_row = expert_holes <= holes_before
    risky_candidate = (
        mask
        & (holes > holes_before[:, None])
        & safe_expert_row[:, None]
    )

    expert_score = scores.gather(1, target[:, None])
    pair_loss = F.relu(scores - expert_score + float(margin))
    if risky_candidate.any():
        loss = pair_loss[risky_candidate].mean()
    else:
        loss = scores.sum() * 0.0

    return loss, expert_risky, min_holes


def _forward_compact(
    *,
    model: TetrioExpertV1Network,
    batch: ExpertV1CompactBatch,
    device: torch.device,
    pin_memory: bool,
):
    tr = lambda a: _to(a, device=device, pin_memory=pin_memory)
    state = torch_dense_state_batch(
        tr(batch.state_board_packed),
        tr(batch.state_active),
        tr(batch.state_hold),
        tr(batch.state_preview),
    )
    candidates = torch_dense_candidate_batch(
        tr(batch.candidate_board_packed),
        tr(batch.candidate_piece),
        tr(batch.candidate_rotation),
        tr(batch.candidate_x),
        tr(batch.candidate_y),
        tr(batch.candidate_use_hold),
        tr(batch.candidate_lines),
    )
    owner = tr(batch.candidate_owner).long()
    local = tr(batch.candidate_local).long()
    counts = tr(batch.candidate_counts).long()
    target = tr(batch.expert_index).long()
    expert_use_hold = tr(batch.expert_use_hold).bool()
    holes_before = tr(batch.holes_before).long()
    flat_candidate_hold = tr(batch.candidate_use_hold).bool()
    flat_holes = tr(batch.candidate_holes).long()

    flat_scores = model.forward_flat(
        state=state,
        candidates=candidates,
        candidate_owner=owner,
    )
    b = state.shape[0]
    k = int(batch.max_candidates)

    scores = flat_scores.new_full(
        (b, k),
        torch.finfo(flat_scores.dtype).min,
    )
    scores[owner, local] = flat_scores

    mask = (
        torch.arange(k, device=device)[None, :]
        < counts[:, None]
    )

    candidate_hold = torch.zeros(
        (b, k),
        device=device,
        dtype=torch.bool,
    )
    candidate_hold[owner, local] = flat_candidate_hold

    candidate_holes = torch.full(
        (b, k),
        255,
        device=device,
        dtype=torch.long,
    )
    candidate_holes[owner, local] = flat_holes

    return (
        scores,
        mask,
        target,
        expert_use_hold,
        candidate_hold,
        candidate_holes,
        holes_before,
    )


def run_epoch(
    *,
    model: TetrioExpertV1Network,
    v0_dir: Path,
    cf_dir: Path,
    device: torch.device,
    batch_size: int,
    optimizer,
    amp_dtype,
    training: bool,
    epoch_seed: int,
    risky_expert_row_weight: float,
    structure_loss_weight: float,
    structure_margin: float,
    label_smoothing: float,
    grad_clip: float,
    prefetch_shards: int,
    pin_memory: bool,
) -> dict[str, float]:
    model.train(training)
    sums = {
        "loss": 0.0,
        "candidate_loss": 0.0,
        "structure_loss": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "mrr": 0.0,
        "branch_acc": 0.0,
        "expert_risky_rate": 0.0,
        "top1_hole_creation_rate": 0.0,
        "top1_avoidable_hole_rate": 0.0,
        "mean_top1_hole_delta": 0.0,
        "rows": 0.0,
    }
    started = time.perf_counter()

    for batch in _iter_batches(
        v0_dir=v0_dir,
        cf_dir=cf_dir,
        batch_size=batch_size,
        training=training,
        epoch_seed=epoch_seed,
        prefetch_shards=prefetch_shards,
    ):
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            (
                scores,
                mask,
                target,
                expert_use_hold,
                candidate_hold,
                candidate_holes,
                holes_before,
            ) = _forward_compact(
                model=model,
                batch=batch,
                device=device,
                pin_memory=pin_memory,
            )

            structure_loss, expert_risky, min_holes = structure_margin_loss(
                scores=scores,
                mask=mask,
                target=target,
                holes=candidate_holes,
                holes_before=holes_before,
                margin=structure_margin,
            )

            row_weight = torch.ones(
                target.shape[0],
                device=device,
                dtype=scores.dtype,
            )
            row_weight = torch.where(
                expert_risky,
                torch.full_like(
                    row_weight,
                    float(risky_expert_row_weight),
                ),
                row_weight,
            )

            candidate_loss = weighted_masked_cross_entropy(
                scores,
                target,
                mask,
                row_weight,
                label_smoothing=label_smoothing if training else 0.0,
            )
            loss = (
                candidate_loss
                + float(structure_loss_weight) * structure_loss
            )

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
            )
            optimizer.step()

        with torch.no_grad():
            metrics = candidate_ranking_metrics(scores, target, mask)
            masked = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min,
            )
            top1_idx = masked.argmax(dim=1)
            pred_hold = candidate_hold.gather(
                1,
                top1_idx[:, None],
            ).squeeze(1)
            branch_acc = (pred_hold == expert_use_hold).float().mean()

            top1_holes = candidate_holes.gather(
                1,
                top1_idx[:, None],
            ).squeeze(1)
            hole_delta = top1_holes - holes_before
            creates = hole_delta > 0
            avoidable = creates & (min_holes <= holes_before)

        b = target.shape[0]
        sums["loss"] += float(loss.detach()) * b
        sums["candidate_loss"] += float(candidate_loss.detach()) * b
        sums["structure_loss"] += float(structure_loss.detach()) * b
        sums["top1"] += float(metrics["top1"]) * b
        sums["top3"] += float(metrics["top3"]) * b
        sums["mrr"] += float(metrics["mrr"]) * b
        sums["branch_acc"] += float(branch_acc) * b
        sums["expert_risky_rate"] += float(expert_risky.float().mean()) * b
        sums["top1_hole_creation_rate"] += float(creates.float().mean()) * b
        sums["top1_avoidable_hole_rate"] += float(avoidable.float().mean()) * b
        sums["mean_top1_hole_delta"] += float(hole_delta.float().mean()) * b
        sums["rows"] += b

    rows = sums.pop("rows")
    elapsed = time.perf_counter() - started
    result = {k: v / rows for k, v in sums.items()}
    result["rows_per_second"] = rows / elapsed if elapsed else 0.0
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not 0.0 < args.risky_expert_row_weight <= 1.0:
        raise SystemExit("--risky-expert-row-weight must be in (0,1]")
    if args.structure_loss_weight < 0.0:
        raise SystemExit("--structure-loss-weight must be >= 0")
    if args.structure_margin < 0.0:
        raise SystemExit("--structure-margin must be >= 0")

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    train_rows = cache_rows(args.train_cf_cache)
    val_rows = cache_rows(args.val_cf_cache)
    if not train_rows or not val_rows:
        raise SystemExit("Train/validation sidecar cache is empty")

    model = TetrioExpertV1Network().to(device)
    initialized_from = None
    if not args.from_scratch:
        v0, v0_ckpt = load_expert_v0(
            args.init_v0,
            device=device,
        )
        model.scorer.load_state_dict(
            v0.scorer.state_dict(),
            strict=True,
        )
        initialized_from = str(args.init_v0)
        del v0
        print(
            f"Warm start   : V0 scorer, epoch={v0_ckpt.get('epoch')} "
            f"from {args.init_v0}"
        )
    else:
        print("Warm start   : disabled (from scratch)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        amp_name = "bf16"
    elif device.type == "cuda":
        amp_dtype = torch.float16
        amp_name = "fp16"
    else:
        amp_dtype = torch.float32
        amp_name = "fp32"

    early = EarlyStoppingTracker(
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        mode="max",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("TETR.IO EXPERT V1 — JOINT HOLD/NO-HOLD RANKING")
    print("=" * 100)
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"AMP          : {amp_name}")
    print(f"Train rows   : {train_rows:,}")
    print(f"Val rows     : {val_rows:,}")
    print(f"Batch        : {args.batch_size}")
    print(f"LR           : {args.lr}")
    print(f"Risk row wt  : {args.risky_expert_row_weight}")
    print(f"Struct wt    : {args.structure_loss_weight}")
    print(f"Struct margin: {args.structure_margin}")
    print()

    best_top1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    stopped = False
    started = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model=model,
            v0_dir=args.train_v0_cache,
            cf_dir=args.train_cf_cache,
            device=device,
            batch_size=args.batch_size,
            optimizer=optimizer,
            amp_dtype=amp_dtype,
            training=True,
            epoch_seed=args.seed + epoch,
            risky_expert_row_weight=args.risky_expert_row_weight,
            structure_loss_weight=args.structure_loss_weight,
            structure_margin=args.structure_margin,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            prefetch_shards=args.prefetch_shards,
            pin_memory=args.pin_memory,
        )
        with torch.inference_mode():
            val = run_epoch(
                model=model,
                v0_dir=args.val_v0_cache,
                cf_dir=args.val_cf_cache,
                device=device,
                batch_size=args.batch_size,
                optimizer=None,
                amp_dtype=amp_dtype,
                training=False,
                epoch_seed=args.seed,
                risky_expert_row_weight=args.risky_expert_row_weight,
                structure_loss_weight=args.structure_loss_weight,
                structure_margin=args.structure_margin,
                label_smoothing=0.0,
                grad_clip=args.grad_clip,
                prefetch_shards=args.prefetch_shards,
                pin_memory=args.pin_memory,
            )

        row = {"epoch": epoch, "train": train, "val": val}
        history.append(row)

        if val["top1"] > best_top1:
            best_top1 = val["top1"]
            best_epoch = epoch
            torch.save(
                {
                    "format": "tetrio_expert_v1",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "metrics": row,
                    "initialized_from": initialized_from,
                    "state_contract": (
                        "400 board + active + hold + first 5 preview"
                    ),
                    "candidate_contract": (
                        "joint no-hold + hold reachable candidates; "
                        "after-board + piece/rotation/x/y/use_hold/lines"
                    ),
                    "quality_contract": (
                        "downweight structurally risky expert labels; "
                        "safe-expert margin vs hole-creating alternatives"
                    ),
                },
                args.output,
            )

        should_stop = False
        if early.enabled:
            if epoch < args.early_stop_warmup_epochs:
                pass
            elif epoch == args.early_stop_warmup_epochs:
                early.anchor_value = best_top1
                early.anchor_epoch = best_epoch
                early.bad_epochs = 0
            else:
                should_stop = early.update(val["top1"], epoch)

        patience = ""
        if early.enabled and epoch >= args.early_stop_warmup_epochs:
            patience = f" patience={early.bad_epochs}/{early.patience}"

        print(
            f"E{epoch:02d}/{args.epochs} | "
            f"train top1={train['top1']:.4f} branch={train['branch_acc']:.4f} "
            f"avoid={train['top1_avoidable_hole_rate']:.4f} "
            f"{train['rows_per_second']:.0f} rows/s | "
            f"val top1={val['top1']:.4f} top3={val['top3']:.4f} "
            f"branch={val['branch_acc']:.4f} "
            f"avoid={val['top1_avoidable_hole_rate']:.4f} "
            f"expertRisk={val['expert_risky_rate']:.4f}{patience}"
        )

        if should_stop:
            stopped = True
            print(
                f"Early stop at epoch {epoch}; best epoch {best_epoch} "
                f"Val Top1={best_top1:.4f}"
            )
            break

    elapsed = time.perf_counter() - started
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    summary = {
        "format": "tetrio_expert_v1_training",
        "status": "RESEARCH PILOT (not Champion)",
        "train_rows": train_rows,
        "val_rows": val_rows,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "amp": amp_name,
        "best_epoch": best_epoch,
        "best_val_top1": best_top1,
        "completed_epochs": len(history),
        "stopped_early": stopped,
        "seconds": elapsed,
        "peak_cuda_memory_bytes": peak,
        "checkpoint": str(args.output),
        "history": history,
    }
    args.metrics.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Best epoch   : {best_epoch}")
    print(f"Best Val Top1: {best_top1:.4f}")
    if peak is not None:
        print(f"Peak CUDA mem: {peak/(1024**3):.2f} GiB")
    print(f"Checkpoint   : {args.output}")
    print(f"Metrics      : {args.metrics}")
    print("Status       : RESEARCH PILOT (not Champion)")


if __name__ == "__main__":
    main()
