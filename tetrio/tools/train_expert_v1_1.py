from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from tetrio.future.dominance import dominance_margin_loss
from tetrio.future.features import feature_index
from tetrio.network.checkpoint import load_expert_v1
from tetrio.network.future_cache import (
    batches_from_future_shard,
    future_shard_paths,
    load_future_shard,
)
from tetrio.network.model_v1_1 import TetrioExpertV11Network


HOLES = feature_index("holes_after")
HOLE_DELTA = feature_index("hole_delta")
T_DESTROYED = feature_index("t_opportunity_destroyed")
T_DEFERRED = feature_index("t_cashout_deferred")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train Expert-v1.1 future residual reranker on a frozen Expert-v1 "
            "base scorer."
        )
    )
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument(
        "--init-v1",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_joint_100k.pt"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_1_future_100k.pt"),
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_1_future_100k_training.json"),
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.01)
    p.add_argument("--risky-expert-row-weight", type=float, default=0.35)
    p.add_argument("--dominance-weight", type=float, default=0.05)
    p.add_argument("--dominance-margin", type=float, default=0.20)
    p.add_argument("--residual-l2-weight", type=float, default=0.002)
    p.add_argument("--max-adjustment", type=float, default=2.0)
    p.add_argument("--reranker-hidden-size", type=int, default=96)
    p.add_argument(
        "--max-top1-drop",
        type=float,
        default=0.015,
        help=(
            "Checkpoint is eligible for quality selection only if validation "
            "Top1 stays within this absolute drop from epoch-0 V1 baseline."
        ),
    )
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument(
        "--recovery-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional self-generated V1 recovery pairs from "
            "collect_expert_v1_recovery_states. These are conservative "
            "dominance pairs, not oracle action labels."
        ),
    )
    p.add_argument("--recovery-weight", type=float, default=0.25)
    p.add_argument("--recovery-batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to(a, device):
    return torch.from_numpy(a).to(
        device=device,
        non_blocking=device.type == "cuda",
    )


def _weighted_ce(
    scores,
    target,
    mask,
    row_weight,
    label_smoothing,
):
    masked = scores.masked_fill(
        ~mask,
        torch.finfo(scores.dtype).min,
    )
    logp = F.log_softmax(masked, dim=1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)

    if label_smoothing:
        valid_sum = logp.masked_fill(~mask, 0.0).sum(dim=1)
        valid_n = mask.sum(dim=1).to(scores.dtype)
        smooth = -valid_sum / valid_n
        per = (
            (1.0 - label_smoothing) * nll
            + label_smoothing * smooth
        )
    else:
        per = nll

    return (per * row_weight).sum() / row_weight.sum().clamp_min(1e-8)


def _metrics(
    final_scores,
    base_scores,
    raw_features,
    candidate_hold,
    inference_mask,
    expert_local,
    expert_hold,
    expert_in_shortlist,
):
    masked = final_scores.masked_fill(
        ~inference_mask,
        torch.finfo(final_scores.dtype).min,
    )
    order = torch.argsort(masked, dim=1, descending=True)
    pred = order[:, 0]

    recall = expert_in_shortlist.bool()
    correct = (pred == expert_local) & recall
    top3 = (
        (order[:, : min(3, order.shape[1])] == expert_local[:, None]).any(dim=1)
        & recall
    )

    rank_match = order == expert_local[:, None]
    rank = rank_match.float().argmax(dim=1) + 1
    mrr = torch.where(
        recall,
        1.0 / rank.float(),
        torch.zeros_like(rank, dtype=torch.float32),
    )

    pred_hold = candidate_hold.gather(1, pred[:, None]).squeeze(1).bool()
    branch_acc = (pred_hold == expert_hold.bool()).float().mean()

    rows = torch.arange(final_scores.shape[0], device=final_scores.device)
    chosen_holes = raw_features[rows, pred, HOLES]
    chosen_delta = raw_features[rows, pred, HOLE_DELTA]
    before_holes = chosen_holes - chosen_delta

    inf_holes = raw_features[..., HOLES].masked_fill(~inference_mask, 255.0)
    min_holes = inf_holes.min(dim=1).values
    avoidable = (chosen_delta > 0) & (min_holes <= before_holes)

    t_destroyed = raw_features[rows, pred, T_DESTROYED] > 0.5
    t_deferred = raw_features[rows, pred, T_DEFERRED] > 0.5

    return {
        "top1": float(correct.float().mean().item()),
        "top3": float(top3.float().mean().item()),
        "mrr": float(mrr.mean().item()),
        "branch_acc": float(branch_acc.item()),
        "shortlist_recall": float(recall.float().mean().item()),
        "avoidable_rate": float(avoidable.float().mean().item()),
        "t_destroyed_rate": float(t_destroyed.float().mean().item()),
        "t_cashout_deferred_rate": float(t_deferred.float().mean().item()),
    }



def load_recovery_pairs(path: Path | None):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Recovery JSONL not found: {path}")

    winner_base = []
    loser_base = []
    winner_feat = []
    loser_feat = []
    winner_hold = []
    loser_hold = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            winner = row["dominant_recovery"]
            loser = row["base_top1"]

            winner_base.append(float(winner["score"]))
            loser_base.append(float(loser["score"]))
            winner_feat.append(winner["future_features"])
            loser_feat.append(loser["future_features"])
            winner_hold.append(int(bool(winner.get("use_hold", False))))
            loser_hold.append(int(bool(loser.get("use_hold", False))))

    if not winner_base:
        return None

    return {
        "winner_base": np.asarray(winner_base, dtype=np.float32),
        "loser_base": np.asarray(loser_base, dtype=np.float32),
        "winner_feat": np.asarray(winner_feat, dtype=np.float32),
        "loser_feat": np.asarray(loser_feat, dtype=np.float32),
        "winner_hold": np.asarray(winner_hold, dtype=np.uint8),
        "loser_hold": np.asarray(loser_hold, dtype=np.uint8),
    }


def run_recovery_epoch(
    *,
    model,
    recovery,
    device,
    optimizer,
    batch_size,
    seed,
    margin,
    recovery_weight,
    residual_l2_weight,
):
    if recovery is None:
        return {
            "pairs": 0,
            "loss": 0.0,
            "pair_acc": 0.0,
        }

    n = len(recovery["winner_base"])
    order = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)

    total_loss = 0.0
    total_acc = 0.0
    seen = 0

    model.reranker.train(True)

    for start in range(0, n, int(batch_size)):
        idx = order[start:start + int(batch_size)]
        b = len(idx)

        base = np.stack(
            (
                recovery["winner_base"][idx],
                recovery["loser_base"][idx],
            ),
            axis=1,
        )
        feat = np.stack(
            (
                recovery["winner_feat"][idx],
                recovery["loser_feat"][idx],
            ),
            axis=1,
        )
        hold = np.stack(
            (
                recovery["winner_hold"][idx],
                recovery["loser_hold"][idx],
            ),
            axis=1,
        )

        base_t = _to(base.astype(np.float32), device).float()
        feat_t = _to(feat.astype(np.float32), device).float()
        hold_t = _to(hold.astype(np.uint8), device).bool()
        mask = torch.ones((b, 2), device=device, dtype=torch.bool)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=(
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            enabled=device.type == "cuda",
        ):
            final, residual = model.final_scores(
                base_scores=base_t,
                raw_features=feat_t,
                candidate_use_hold=hold_t,
                mask=mask,
            )
            pair = torch.relu(
                final[:, 1] - final[:, 0] + float(margin)
            ).mean()
            res_l2 = residual.pow(2).mean()
            loss = (
                float(recovery_weight) * pair
                + float(residual_l2_weight) * res_l2
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.reranker.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            acc = (final[:, 0] > final[:, 1]).float().mean()

        total_loss += float(loss.detach()) * b
        total_acc += float(acc) * b
        seen += b

    return {
        "pairs": seen,
        "loss": total_loss / max(1, seen),
        "pair_acc": total_acc / max(1, seen),
    }


def run_epoch(
    *,
    model,
    cache_dir,
    device,
    batch_size,
    optimizer,
    training,
    seed,
    risky_expert_row_weight,
    dominance_weight,
    dominance_margin,
    residual_l2_weight,
    label_smoothing,
):
    model.reranker.train(training)
    sums = {
        "loss": 0.0,
        "ce": 0.0,
        "dominance": 0.0,
        "residual_l2": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "mrr": 0.0,
        "branch_acc": 0.0,
        "shortlist_recall": 0.0,
        "avoidable_rate": 0.0,
        "t_destroyed_rate": 0.0,
        "t_cashout_deferred_rate": 0.0,
        "expert_risky_rate": 0.0,
        "dominance_pairs": 0.0,
        "rows": 0.0,
    }
    started = time.perf_counter()

    paths = future_shard_paths(cache_dir)
    rng = np.random.default_rng(seed) if training else None
    if training:
        rng.shuffle(paths)

    for path in paths:
        data = load_future_shard(path)
        for batch in batches_from_future_shard(
            data,
            batch_size=batch_size,
            rng=rng if training else None,
        ):
            base = _to(batch.base_scores, device).float()
            features = _to(batch.raw_features, device).float()
            hold = _to(batch.candidate_use_hold, device).bool()
            valid = _to(batch.valid_mask.astype(np.uint8), device).bool()
            inference = _to(batch.inference_mask.astype(np.uint8), device).bool()
            target = _to(batch.expert_local, device).long()
            expert_hold = _to(batch.expert_use_hold, device).bool()
            recall = _to(batch.expert_in_shortlist, device).bool()

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.bfloat16
                    if device.type == "cuda" and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
                enabled=device.type == "cuda",
            ):
                final, residual = model.final_scores(
                    base_scores=base,
                    raw_features=features,
                    candidate_use_hold=hold,
                    mask=valid,
                )

                rows = torch.arange(base.shape[0], device=device)
                expert_holes = features[rows, target, HOLES]
                expert_delta = features[rows, target, HOLE_DELTA]
                before_holes = expert_holes - expert_delta

                inf_holes = features[..., HOLES].masked_fill(~inference, 255.0)
                min_holes = inf_holes.min(dim=1).values
                expert_risky = (
                    (expert_delta > 0)
                    & (min_holes <= before_holes)
                )

                # Reranker cannot recover an expert candidate that the frozen
                # base shortlist excluded.  Do not train on impossible rows.
                row_weight = recall.to(final.dtype)
                row_weight = torch.where(
                    expert_risky & recall,
                    torch.full_like(
                        row_weight,
                        float(risky_expert_row_weight),
                    ),
                    row_weight,
                )

                ce = _weighted_ce(
                    final,
                    target,
                    valid,
                    row_weight,
                    label_smoothing if training else 0.0,
                )
                dom, pair_count = dominance_margin_loss(
                    final,
                    features,
                    inference,
                    margin=dominance_margin,
                )
                res_l2 = (
                    residual.masked_select(valid).pow(2).mean()
                    if valid.any()
                    else residual.sum() * 0.0
                )
                loss = (
                    ce
                    + float(dominance_weight) * dom
                    + float(residual_l2_weight) * res_l2
                )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.reranker.parameters(),
                    1.0,
                )
                optimizer.step()

            met = _metrics(
                final,
                base,
                features,
                hold,
                inference,
                target,
                expert_hold,
                recall,
            )

            b = base.shape[0]
            sums["loss"] += float(loss.detach()) * b
            sums["ce"] += float(ce.detach()) * b
            sums["dominance"] += float(dom.detach()) * b
            sums["residual_l2"] += float(res_l2.detach()) * b
            for key, value in met.items():
                sums[key] += float(value) * b
            sums["expert_risky_rate"] += float(expert_risky.float().mean()) * b
            sums["dominance_pairs"] += pair_count
            sums["rows"] += b

    rows = sums.pop("rows")
    elapsed = time.perf_counter() - started
    out = {
        k: (v / rows if k != "dominance_pairs" else v)
        for k, v in sums.items()
    }
    out["rows_per_second"] = rows / elapsed if elapsed else 0.0
    return out


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    v1, v1_ckpt = load_expert_v1(args.init_v1, device=device)
    model = TetrioExpertV11Network(
        reranker_hidden_size=args.reranker_hidden_size,
        max_adjustment=args.max_adjustment,
    ).to(device)
    model.scorer.load_state_dict(v1.scorer.state_dict(), strict=True)
    model.freeze_scorer()
    del v1

    optimizer = torch.optim.AdamW(
        model.reranker.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    recovery = load_recovery_pairs(args.recovery_jsonl)
    recovery_count = (
        0 if recovery is None else len(recovery["winner_base"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("TETR.IO EXPERT V1.1 — FUTURE RESIDUAL RERANKER")
    print("=" * 104)
    print(f"Base V1      : {args.init_v1} epoch={v1_ckpt.get('epoch')}")
    print(f"Train cache  : {args.train_cache}")
    print(f"Val cache    : {args.val_cache}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"Batch        : {args.batch_size}")
    print(f"Max adjust   : ±{args.max_adjustment}")
    print(f"Recovery     : {recovery_count:,} pair(s)")
    print(f"Recovery wt  : {args.recovery_weight}")
    print()

    with torch.inference_mode():
        baseline = run_epoch(
            model=model,
            cache_dir=args.val_cache,
            device=device,
            batch_size=args.batch_size,
            optimizer=None,
            training=False,
            seed=args.seed,
            risky_expert_row_weight=args.risky_expert_row_weight,
            dominance_weight=args.dominance_weight,
            dominance_margin=args.dominance_margin,
            residual_l2_weight=args.residual_l2_weight,
            label_smoothing=0.0,
        )

    print(
        "E00 V1 BASE | "
        f"top1={baseline['top1']:.4f} top3={baseline['top3']:.4f} "
        f"branch={baseline['branch_acc']:.4f} "
        f"recall={baseline['shortlist_recall']:.4f} "
        f"avoid={baseline['avoidable_rate']:.4f} "
        f"Tdestroy={baseline['t_destroyed_rate']:.4f} "
        f"Tdefer={baseline['t_cashout_deferred_rate']:.4f}"
    )

    top1_floor = baseline["top1"] - float(args.max_top1_drop)
    best_quality = float("inf")
    best_epoch = 0
    history = [{"epoch": 0, "val": baseline, "kind": "frozen_v1_baseline"}]
    started = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model=model,
            cache_dir=args.train_cache,
            device=device,
            batch_size=args.batch_size,
            optimizer=optimizer,
            training=True,
            seed=args.seed + epoch,
            risky_expert_row_weight=args.risky_expert_row_weight,
            dominance_weight=args.dominance_weight,
            dominance_margin=args.dominance_margin,
            residual_l2_weight=args.residual_l2_weight,
            label_smoothing=args.label_smoothing,
        )
        recovery_metrics = run_recovery_epoch(
            model=model,
            recovery=recovery,
            device=device,
            optimizer=optimizer,
            batch_size=args.recovery_batch_size,
            seed=args.seed + 100_000 + epoch,
            margin=args.dominance_margin,
            recovery_weight=args.recovery_weight,
            residual_l2_weight=args.residual_l2_weight,
        )

        with torch.inference_mode():
            val = run_epoch(
                model=model,
                cache_dir=args.val_cache,
                device=device,
                batch_size=args.batch_size,
                optimizer=None,
                training=False,
                seed=args.seed,
                risky_expert_row_weight=args.risky_expert_row_weight,
                dominance_weight=args.dominance_weight,
                dominance_margin=args.dominance_margin,
                residual_l2_weight=args.residual_l2_weight,
                label_smoothing=0.0,
            )

        eligible = val["top1"] >= top1_floor
        quality = (
            val["avoidable_rate"]
            + 0.50 * val["t_destroyed_rate"]
            + 0.25 * val["t_cashout_deferred_rate"]
        )
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "val": val,
                "recovery": recovery_metrics,
                "eligible": eligible,
                "quality_cost": quality,
            }
        )

        if eligible and quality < best_quality:
            best_quality = quality
            best_epoch = epoch
            torch.save(
                {
                    "format": "tetrio_expert_v1_1",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "metrics": history[-1],
                    "base_v1": str(args.init_v1),
                    "base_v1_epoch": v1_ckpt.get("epoch"),
                    "selection_contract": (
                        f"Val Top1 >= V1 baseline-{args.max_top1_drop}; "
                        "then minimize avoidable + 0.5*Tdestroy + 0.25*Tdefer"
                    ),
                    "status": "RESEARCH PILOT (not Champion)",
                },
                args.output,
            )

        print(
            f"E{epoch:02d}/{args.epochs} | "
            f"train top1={train['top1']:.4f} avoid={train['avoidable_rate']:.4f} "
            f"Tdestroy={train['t_destroyed_rate']:.4f} "
            f"{train['rows_per_second']:.0f} rows/s | "
            f"val top1={val['top1']:.4f} top3={val['top3']:.4f} "
            f"branch={val['branch_acc']:.4f} avoid={val['avoidable_rate']:.4f} "
            f"Tdestroy={val['t_destroyed_rate']:.4f} "
            f"Tdefer={val['t_cashout_deferred_rate']:.4f} "
            f"recoveryAcc={recovery_metrics['pair_acc']:.3f} "
            f"{'ELIGIBLE' if eligible else 'TOP1_GUARD_FAIL'}"
        )

    if best_epoch == 0:
        raise SystemExit(
            "No V1.1 epoch passed the Top1 guard. Keep V1 baseline; "
            "do not promote this pilot."
        )

    elapsed = time.perf_counter() - started
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    summary = {
        "format": "tetrio_expert_v1_1_training",
        "status": "RESEARCH PILOT (not Champion)",
        "base_v1": str(args.init_v1),
        "base_v1_epoch": v1_ckpt.get("epoch"),
        "baseline": baseline,
        "top1_floor": top1_floor,
        "best_epoch": best_epoch,
        "best_quality_cost": best_quality,
        "checkpoint": str(args.output),
        "recovery_jsonl": (
            None if args.recovery_jsonl is None else str(args.recovery_jsonl)
        ),
        "recovery_pairs": recovery_count,
        "seconds": elapsed,
        "peak_cuda_memory_bytes": peak,
        "history": history,
    }
    args.metrics.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"Best epoch   : {best_epoch}")
    print(f"Quality cost : {best_quality:.6f}")
    if peak is not None:
        print(f"Peak CUDA mem: {peak/(1024**3):.2f} GiB")
    print(f"Checkpoint   : {args.output}")
    print(f"Metrics      : {args.metrics}")
    print("Status       : RESEARCH PILOT (not Champion)")


if __name__ == "__main__":
    main()
