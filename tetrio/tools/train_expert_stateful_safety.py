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
from tetrio.network.checkpoint import load_expert_v1_1
from tetrio.network.future_cache import future_shard_paths, load_future_shard
from tetrio.network.model_v1_2 import TetrioExpertV12Network
from tetrio.network.stateful_cache import (
    batches_from_stateful_pair,
    load_stateful_shard,
)
from tetrio.stateful.features import STATEFUL_FEATURE_NAMES
from tetrio.stateful.safety import (
    add_count_dict,
    change_diagnostics,
    classify_expert_vs_baseline,
    rates_from_counts,
    relation_counts,
    relation_row_weights,
    unsafe_expert_preservation_loss,
)


HOLES = feature_index("holes_after")
HOLE_DELTA = feature_index("hole_delta")
T_DESTROYED = feature_index("t_opportunity_destroyed")
T_DEFERRED = feature_index("t_cashout_deferred")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the safety-constrained causal state residual on frozen "
            "Expert V1.1 500K."
        )
    )
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--train-state-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--val-state-cache", type=Path, required=True)
    p.add_argument(
        "--init-v1-1",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_1_future_500k.pt"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"models\tetrio_expert_stateful_safety_100k.pt"),
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            r"artifacts\tetrio\expert_stateful_safety_100k_training.json"
        ),
    )

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.01)

    # Frozen before the experiment. These weights encode the intended contract:
    # learn strongly when expert structurally dominates the baseline, weakly
    # when relation is ambiguous, and never imitate a structurally dominated
    # expert target.
    p.add_argument("--agreement-row-weight", type=float, default=0.10)
    p.add_argument("--safe-row-weight", type=float, default=1.00)
    p.add_argument("--unsafe-row-weight", type=float, default=0.00)
    p.add_argument("--ambiguous-row-weight", type=float, default=0.20)

    p.add_argument("--safety-pair-weight", type=float, default=0.50)
    p.add_argument("--safety-margin", type=float, default=0.10)
    p.add_argument("--dominance-weight", type=float, default=0.10)
    p.add_argument("--dominance-margin", type=float, default=0.20)
    p.add_argument("--residual-l2-weight", type=float, default=0.003)

    p.add_argument("--state-hidden-size", type=int, default=64)
    p.add_argument("--state-max-adjustment", type=float, default=1.0)

    # Predeclared promotion contract. Both imitation and structural quality must
    # improve over the exact ZERO-state / frozen V1.1 baseline.
    p.add_argument("--min-top1-improvement", type=float, default=0.0020)
    p.add_argument("--min-quality-improvement", type=float, default=0.0005)
    p.add_argument("--max-unsafe-change-rate", type=float, default=0.0050)

    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument(
        "--recovery-jsonl",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_recovery_states.jsonl"),
        help=(
            "Provenance only. Frozen V1.1 already learned these events; the "
            "historical recovery JSONL has no audited battle state."
        ),
    )
    p.add_argument("--recovery-weight", type=float, default=0.25)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to(a, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(a).to(
        device=device,
        non_blocking=device.type == "cuda",
    )


def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=(
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        ),
        enabled=device.type == "cuda",
    )


def _weighted_ce(
    scores: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    row_weight: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
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
            (1.0 - float(label_smoothing)) * nll
            + float(label_smoothing) * smooth
        )
    else:
        per = nll

    denom = row_weight.sum()
    if float(denom.detach().item()) <= 0.0:
        return scores.sum() * 0.0
    return (per * row_weight).sum() / denom


def _metrics(
    final_scores: torch.Tensor,
    raw_features: torch.Tensor,
    candidate_hold: torch.Tensor,
    inference_mask: torch.Tensor,
    expert_local: torch.Tensor,
    expert_hold: torch.Tensor,
    expert_in_shortlist: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    masked = final_scores.masked_fill(
        ~inference_mask,
        torch.finfo(final_scores.dtype).min,
    )
    order = torch.argsort(masked, dim=1, descending=True)
    pred = order[:, 0]
    recall = expert_in_shortlist.bool()

    correct = (pred == expert_local) & recall
    top3 = (
        (
            order[:, : min(3, order.shape[1])]
            == expert_local[:, None]
        ).any(dim=1)
        & recall
    )
    rank_match = order == expert_local[:, None]
    rank = rank_match.float().argmax(dim=1) + 1
    mrr = torch.where(
        recall,
        1.0 / rank.float(),
        torch.zeros_like(rank, dtype=torch.float32),
    )

    pred_hold = candidate_hold.gather(
        1,
        pred[:, None],
    ).squeeze(1).bool()
    branch_acc = (pred_hold == expert_hold.bool()).float().mean()

    rows = torch.arange(
        final_scores.shape[0],
        device=final_scores.device,
    )
    chosen_holes = raw_features[rows, pred, HOLES]
    chosen_delta = raw_features[rows, pred, HOLE_DELTA]
    before_holes = chosen_holes - chosen_delta

    inf_holes = raw_features[..., HOLES].masked_fill(
        ~inference_mask,
        255.0,
    )
    min_holes = inf_holes.min(dim=1).values
    avoidable = (chosen_delta > 0) & (min_holes <= before_holes)

    t_destroyed = raw_features[rows, pred, T_DESTROYED] > 0.5
    t_deferred = raw_features[rows, pred, T_DEFERRED] > 0.5

    return {
        "top1": float(correct.float().mean()),
        "top3": float(top3.float().mean()),
        "mrr": float(mrr.mean()),
        "branch_acc": float(branch_acc),
        "shortlist_recall": float(recall.float().mean()),
        "avoidable_rate": float(avoidable.float().mean()),
        "t_destroyed_rate": float(t_destroyed.float().mean()),
        "t_cashout_deferred_rate": float(t_deferred.float().mean()),
    }, pred


def quality_cost(metrics: dict[str, float]) -> float:
    return float(
        metrics["avoidable_rate"]
        + 0.50 * metrics["t_destroyed_rate"]
        + 0.25 * metrics["t_cashout_deferred_rate"]
    )


def count_recovery(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _pair_paths(
    future_dir: Path,
    state_dir: Path,
) -> list[tuple[Path, Path]]:
    pairs = []
    for future_path in future_shard_paths(future_dir):
        state_path = state_dir / future_path.name
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Missing matching state shard: {state_path}"
            )
        pairs.append((future_path, state_path))
    return pairs


def run_training_epoch(
    *,
    model: TetrioExpertV12Network,
    future_dir: Path,
    state_dir: Path,
    device: torch.device,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    model.stateful.train(True)
    pairs = _pair_paths(future_dir, state_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)

    sums = {
        "loss": 0.0,
        "ce": 0.0,
        "safety_pair": 0.0,
        "dominance": 0.0,
        "residual_l2": 0.0,
        "rows": 0.0,
        "effective_row_weight": 0.0,
        "unsafe_pair_constraints": 0,
        "dominance_pairs": 0,
    }
    relation_total: dict[str, int] = {}
    started = time.perf_counter()

    for future_path, state_path in pairs:
        future = load_future_shard(future_path)
        state = load_stateful_shard(state_path)

        for batch in batches_from_stateful_pair(
            future=future,
            stateful=state,
            batch_size=batch_size,
            rng=rng,
        ):
            f = batch.future
            base = _to(f.base_scores, device).float()
            feat = _to(f.raw_features, device).float()
            hold = _to(f.candidate_use_hold, device).bool()
            valid = _to(f.valid_mask.astype(np.uint8), device).bool()
            inf = _to(
                f.inference_mask.astype(np.uint8),
                device,
            ).bool()
            target = _to(f.expert_local, device).long()
            recall = _to(f.expert_in_shortlist, device).bool()
            battle = _to(batch.battle_state, device).float()

            with torch.no_grad(), _autocast(device):
                v11, _ = model.base.final_scores(
                    base_scores=base,
                    raw_features=feat,
                    candidate_use_hold=hold,
                    mask=valid,
                )
                relation = classify_expert_vs_baseline(
                    baseline_scores=v11,
                    raw_features=feat,
                    inference_mask=inf,
                    expert_local=target,
                    expert_in_shortlist=recall,
                )

            add_count_dict(
                relation_total,
                relation_counts(relation),
            )

            row_weight = relation_row_weights(
                relation,
                agreement_weight=args.agreement_row_weight,
                safe_weight=args.safe_row_weight,
                unsafe_weight=args.unsafe_row_weight,
                ambiguous_weight=args.ambiguous_row_weight,
                dtype=v11.dtype,
            )

            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                residual = model.stateful(
                    v11_scores=v11,
                    raw_future_features=feat,
                    candidate_use_hold=hold,
                    battle_state=battle,
                    mask=valid,
                )
                final = (v11 + residual).masked_fill(
                    ~valid,
                    torch.finfo(v11.dtype).min,
                )

                ce = _weighted_ce(
                    final,
                    target,
                    valid,
                    row_weight,
                    args.label_smoothing,
                )

                safety_pair, unsafe_pairs = (
                    unsafe_expert_preservation_loss(
                        final_scores=final,
                        expert_local=target,
                        relation=relation,
                        margin=args.safety_margin,
                    )
                )

                dominance, dom_pairs = dominance_margin_loss(
                    final,
                    feat,
                    inf,
                    margin=args.dominance_margin,
                )

                if valid.any():
                    residual_l2 = (
                        residual.masked_select(valid).pow(2).mean()
                    )
                else:
                    residual_l2 = residual.sum() * 0.0

                loss = (
                    ce
                    + float(args.safety_pair_weight) * safety_pair
                    + float(args.dominance_weight) * dominance
                    + float(args.residual_l2_weight) * residual_l2
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.stateful.parameters(),
                1.0,
            )
            optimizer.step()

            b = int(base.shape[0])
            sums["loss"] += float(loss.detach()) * b
            sums["ce"] += float(ce.detach()) * b
            sums["safety_pair"] += float(safety_pair.detach()) * b
            sums["dominance"] += float(dominance.detach()) * b
            sums["residual_l2"] += float(residual_l2.detach()) * b
            sums["rows"] += b
            sums["effective_row_weight"] += float(
                row_weight.detach().sum().item()
            )
            sums["unsafe_pair_constraints"] += int(unsafe_pairs)
            sums["dominance_pairs"] += int(dom_pairs)

    rows = max(1.0, float(sums.pop("rows")))
    elapsed = time.perf_counter() - started
    out = {
        "loss": sums["loss"] / rows,
        "ce": sums["ce"] / rows,
        "safety_pair": sums["safety_pair"] / rows,
        "dominance": sums["dominance"] / rows,
        "residual_l2": sums["residual_l2"] / rows,
        "effective_row_weight_per_row": (
            sums["effective_row_weight"] / rows
        ),
        "unsafe_pair_constraints": int(
            sums["unsafe_pair_constraints"]
        ),
        "dominance_pairs": int(sums["dominance_pairs"]),
        "rows_per_second": rows / max(elapsed, 1e-9),
        "relation_counts": relation_total,
        "relation_rates": rates_from_counts(relation_total),
    }
    return out


def evaluate_safety_ablation(
    *,
    model: TetrioExpertV12Network,
    future_dir: Path,
    state_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()

    metric_keys = (
        "top1",
        "top3",
        "mrr",
        "branch_acc",
        "shortlist_recall",
        "avoidable_rate",
        "t_destroyed_rate",
        "t_cashout_deferred_rate",
    )
    true_sums = {k: 0.0 for k in metric_keys}
    zero_sums = {k: 0.0 for k in metric_keys}

    rows_total = 0
    max_true_residual = 0.0
    max_zero_residual = 0.0
    max_zero_score_delta = 0.0

    expert_relation_total: dict[str, int] = {}
    change_total: dict[str, int] = {}

    for future_path, state_path in _pair_paths(
        future_dir,
        state_dir,
    ):
        future = load_future_shard(future_path)
        state = load_stateful_shard(state_path)

        for batch in batches_from_stateful_pair(
            future=future,
            stateful=state,
            batch_size=batch_size,
            rng=None,
        ):
            f = batch.future
            base = _to(f.base_scores, device).float()
            feat = _to(f.raw_features, device).float()
            hold = _to(f.candidate_use_hold, device).bool()
            valid = _to(f.valid_mask.astype(np.uint8), device).bool()
            inf = _to(
                f.inference_mask.astype(np.uint8),
                device,
            ).bool()
            target = _to(f.expert_local, device).long()
            expert_hold = _to(f.expert_use_hold, device).bool()
            recall = _to(f.expert_in_shortlist, device).bool()
            battle = _to(batch.battle_state, device).float()
            zero_battle = torch.zeros_like(battle)

            with torch.inference_mode(), _autocast(device):
                v11, _ = model.base.final_scores(
                    base_scores=base,
                    raw_features=feat,
                    candidate_use_hold=hold,
                    mask=valid,
                )
                relation = classify_expert_vs_baseline(
                    baseline_scores=v11,
                    raw_features=feat,
                    inference_mask=inf,
                    expert_local=target,
                    expert_in_shortlist=recall,
                )

                r_true = model.stateful(
                    v11_scores=v11,
                    raw_future_features=feat,
                    candidate_use_hold=hold,
                    battle_state=battle,
                    mask=valid,
                )
                r_zero = model.stateful(
                    v11_scores=v11,
                    raw_future_features=feat,
                    candidate_use_hold=hold,
                    battle_state=zero_battle,
                    mask=valid,
                )

                true_final = (v11 + r_true).masked_fill(
                    ~valid,
                    torch.finfo(v11.dtype).min,
                )
                zero_final = (v11 + r_zero).masked_fill(
                    ~valid,
                    torch.finfo(v11.dtype).min,
                )

            true_m, true_pred = _metrics(
                true_final,
                feat,
                hold,
                inf,
                target,
                expert_hold,
                recall,
            )
            zero_m, zero_pred = _metrics(
                zero_final,
                feat,
                hold,
                inf,
                target,
                expert_hold,
                recall,
            )

            b = int(base.shape[0])
            for key in metric_keys:
                true_sums[key] += true_m[key] * b
                zero_sums[key] += zero_m[key] * b

            add_count_dict(
                expert_relation_total,
                relation_counts(relation),
            )
            add_count_dict(
                change_total,
                change_diagnostics(
                    baseline_pred=zero_pred,
                    true_pred=true_pred,
                    expert_local=target,
                    expert_in_shortlist=recall,
                    raw_features=feat,
                    inference_mask=inf,
                ),
            )

            rows_total += b
            max_true_residual = max(
                max_true_residual,
                float(r_true.abs().max().item()),
            )
            max_zero_residual = max(
                max_zero_residual,
                float(r_zero.abs().max().item()),
            )
            max_zero_score_delta = max(
                max_zero_score_delta,
                float(
                    (zero_final - v11)
                    .masked_select(valid)
                    .abs()
                    .max()
                    .item()
                ),
            )

    true = {
        key: value / rows_total
        for key, value in true_sums.items()
    }
    zero = {
        key: value / rows_total
        for key, value in zero_sums.items()
    }
    change_rates = rates_from_counts(change_total)
    relation_rates = rates_from_counts(expert_relation_total)

    return {
        "true": true,
        "zero": zero,
        "true_quality_cost": quality_cost(true),
        "zero_quality_cost": quality_cost(zero),
        "quality_delta_true_minus_zero": (
            quality_cost(true) - quality_cost(zero)
        ),
        "top1_delta_true_minus_zero": (
            true["top1"] - zero["top1"]
        ),
        "decision_change_rate": change_rates.get(
            "changed_rate",
            0.0,
        ),
        "unsafe_change_rate": change_rates.get(
            "unsafe_change_rate",
            0.0,
        ),
        "safe_change_rate": change_rates.get(
            "safe_change_rate",
            0.0,
        ),
        "ambiguous_change_rate": change_rates.get(
            "ambiguous_change_rate",
            0.0,
        ),
        "max_true_residual": max_true_residual,
        "max_zero_residual": max_zero_residual,
        "max_zero_score_delta_vs_v11": max_zero_score_delta,
        "expert_relation_counts": expert_relation_total,
        "expert_relation_rates": relation_rates,
        "change_counts": change_total,
        "change_rates": change_rates,
        "rows": rows_total,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    base, base_ckpt = load_expert_v1_1(
        args.init_v1_1,
        device=device,
    )
    base_cfg = base_ckpt.get("config", {})

    model = TetrioExpertV12Network(
        v11_reranker_hidden_size=int(
            base_cfg.get("reranker_hidden_size", 96)
        ),
        v11_max_adjustment=float(
            base_cfg.get("max_adjustment", 2.0)
        ),
        state_hidden_size=args.state_hidden_size,
        state_max_adjustment=args.state_max_adjustment,
    ).to(device)
    model.base.load_state_dict(base.state_dict(), strict=True)
    model.freeze_base()
    del base

    optimizer = torch.optim.AdamW(
        model.stateful.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    recovery_count = count_recovery(args.recovery_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("TETR.IO EXPERT — SAFETY-CONSTRAINED CAUSAL STATE RESIDUAL")
    print("=" * 112)
    print(
        f"Base V1.1     : {args.init_v1_1} "
        f"epoch={base_ckpt.get('epoch')}"
    )
    print(f"Train future  : {args.train_cache}")
    print(f"Train state   : {args.train_state_cache}")
    print(f"Val future    : {args.val_cache}")
    print(f"Val state     : {args.val_state_cache}")
    print(f"State fields  : {', '.join(STATEFUL_FEATURE_NAMES)}")
    print(f"State adjust  : ±{args.state_max_adjustment}")
    print(
        "Row weights   : "
        f"agree={args.agreement_row_weight} "
        f"safe={args.safe_row_weight} "
        f"unsafe={args.unsafe_row_weight} "
        f"ambiguous={args.ambiguous_row_weight}"
    )
    print(
        "Safety loss   : "
        f"weight={args.safety_pair_weight} "
        f"margin={args.safety_margin}"
    )
    print(
        "Global dom    : "
        f"weight={args.dominance_weight} "
        f"margin={args.dominance_margin}"
    )
    print(
        f"Recovery      : {recovery_count:,} pair(s), "
        f"weight={args.recovery_weight} "
        "(frozen-base provenance only)"
    )
    print(
        "Promotion     : "
        f"ΔTop1>={args.min_top1_improvement:+.4f}, "
        f"ΔQ<=-{args.min_quality_improvement:.4f}, "
        f"unsafe-change<={args.max_unsafe_change_rate:.4f}"
    )
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(device)}")
    print()

    e00 = evaluate_safety_ablation(
        model=model,
        future_dir=args.val_cache,
        state_dir=args.val_state_cache,
        device=device,
        batch_size=args.batch_size,
    )
    if (
        e00["max_true_residual"] != 0.0
        or e00["max_zero_residual"] != 0.0
        or e00["decision_change_rate"] != 0.0
    ):
        raise SystemExit(f"E00 PARITY FAIL: {e00}")

    print(
        "E00 V1.1 PARITY PASS | "
        f"top1={e00['zero']['top1']:.4f} "
        f"top3={e00['zero']['top3']:.4f} "
        f"branch={e00['zero']['branch_acc']:.4f} "
        f"avoid={e00['zero']['avoidable_rate']:.4f} "
        f"Tdestroy={e00['zero']['t_destroyed_rate']:.4f} "
        f"Tdefer={e00['zero']['t_cashout_deferred_rate']:.4f} "
        "maxResidual=0 decisionChange=0"
    )

    rel = e00["expert_relation_counts"]
    print(
        "VAL relation   | "
        f"agree={rel.get('agreement_rows', 0):,} "
        f"safe={rel.get('safe_expert_rows', 0):,} "
        f"unsafe={rel.get('unsafe_expert_rows', 0):,} "
        f"ambig={rel.get('ambiguous_rows', 0):,} "
        f"missing={rel.get('missing_expert_rows', 0):,}"
    )
    print()

    baseline_quality = e00["zero_quality_cost"]
    required_quality = (
        baseline_quality - args.min_quality_improvement
    )
    required_top1 = (
        e00["zero"]["top1"] + args.min_top1_improvement
    )

    best_epoch = 0
    best_quality = float("inf")
    best_top1 = float("-inf")
    history = [
        {
            "epoch": 0,
            "kind": "frozen_v1_1_e00",
            "ablation": e00,
        }
    ]

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs + 1):
        train = run_training_epoch(
            model=model,
            future_dir=args.train_cache,
            state_dir=args.train_state_cache,
            device=device,
            batch_size=args.batch_size,
            optimizer=optimizer,
            seed=args.seed + epoch,
            args=args,
        )
        ab = evaluate_safety_ablation(
            model=model,
            future_dir=args.val_cache,
            state_dir=args.val_state_cache,
            device=device,
            batch_size=args.batch_size,
        )

        eligible = (
            ab["true"]["top1"] >= required_top1
            and ab["true_quality_cost"] <= required_quality
            and ab["decision_change_rate"] > 0.0
            and ab["unsafe_change_rate"]
            <= args.max_unsafe_change_rate
            and ab["max_zero_residual"] == 0.0
            and ab["max_zero_score_delta_vs_v11"] == 0.0
        )

        item = {
            "epoch": epoch,
            "train": train,
            "ablation": ab,
            "eligible": bool(eligible),
            "quality_cost": ab["true_quality_cost"],
        }
        history.append(item)

        candidate_better = (
            best_epoch == 0
            or ab["true_quality_cost"] < best_quality - 1e-12
            or (
                abs(ab["true_quality_cost"] - best_quality) <= 1e-12
                and ab["true"]["top1"] > best_top1
            )
        )

        if eligible and candidate_better:
            best_epoch = epoch
            best_quality = ab["true_quality_cost"]
            best_top1 = ab["true"]["top1"]

            torch.save(
                {
                    "format": "tetrio_expert_v1_2_stateful",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "base_v1_1": str(args.init_v1_1),
                    "base_v1_1_epoch": base_ckpt.get("epoch"),
                    "base_v1_1_config": base_cfg,
                    "state_feature_names": list(
                        STATEFUL_FEATURE_NAMES
                    ),
                    "training_objective": (
                        "safety_constrained_stateful"
                    ),
                    "safety_contract": {
                        "agreement_row_weight": (
                            args.agreement_row_weight
                        ),
                        "safe_row_weight": args.safe_row_weight,
                        "unsafe_row_weight": args.unsafe_row_weight,
                        "ambiguous_row_weight": (
                            args.ambiguous_row_weight
                        ),
                        "safety_pair_weight": (
                            args.safety_pair_weight
                        ),
                        "safety_margin": args.safety_margin,
                        "dominance_weight": args.dominance_weight,
                        "dominance_margin": args.dominance_margin,
                        "min_top1_improvement": (
                            args.min_top1_improvement
                        ),
                        "min_quality_improvement": (
                            args.min_quality_improvement
                        ),
                        "max_unsafe_change_rate": (
                            args.max_unsafe_change_rate
                        ),
                    },
                    "metrics": item,
                    "status": "RESEARCH PILOT (not Champion)",
                },
                args.output,
            )

        tr = train["relation_counts"]
        cr = ab["change_rates"]
        print(
            f"E{epoch:02d}/{args.epochs} | "
            f"train {train['rows_per_second']:.0f} rows/s "
            f"A/S/U/?="
            f"{tr.get('agreement_rows', 0):,}/"
            f"{tr.get('safe_expert_rows', 0):,}/"
            f"{tr.get('unsafe_expert_rows', 0):,}/"
            f"{tr.get('ambiguous_rows', 0):,} | "
            f"TRUE top1={ab['true']['top1']:.4f} "
            f"top3={ab['true']['top3']:.4f} "
            f"branch={ab['true']['branch_acc']:.4f} "
            f"avoid={ab['true']['avoidable_rate']:.4f} "
            f"Tdestroy={ab['true']['t_destroyed_rate']:.4f} "
            f"Tdefer={ab['true']['t_cashout_deferred_rate']:.4f} "
            f"Q={ab['true_quality_cost']:.6f} | "
            f"ZERO top1={ab['zero']['top1']:.4f} "
            f"Q={ab['zero_quality_cost']:.6f} | "
            f"change={ab['decision_change_rate']:.4f} "
            f"safechg={ab['safe_change_rate']:.4f} "
            f"unsafechg={ab['unsafe_change_rate']:.4f} "
            f"toExpert={cr.get('changed_to_expert_rate', 0.0):.4f} "
            f"{'ELIGIBLE' if eligible else 'NO_PROMOTION'}"
        )

    elapsed = time.perf_counter() - started
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    status = (
        "RESEARCH PILOT (not Champion)"
        if best_epoch
        else "NON_PROMOTION"
    )

    summary = {
        "format": "tetrio_expert_stateful_safety_training",
        "status": status,
        "base_v1_1": str(args.init_v1_1),
        "base_v1_1_epoch": base_ckpt.get("epoch"),
        "e00": e00,
        "baseline_quality_cost": baseline_quality,
        "required_quality_cost": required_quality,
        "required_top1": required_top1,
        "max_unsafe_change_rate": args.max_unsafe_change_rate,
        "best_epoch": best_epoch,
        "best_quality_cost": (
            best_quality if best_epoch else None
        ),
        "best_top1": best_top1 if best_epoch else None,
        "checkpoint": str(args.output) if best_epoch else None,
        "recovery_jsonl": (
            str(args.recovery_jsonl)
            if args.recovery_jsonl
            else None
        ),
        "recovery_pairs": recovery_count,
        "recovery_weight_requested": args.recovery_weight,
        "recovery_stateful_loss_enabled": False,
        "seconds": elapsed,
        "peak_cuda_memory_bytes": peak,
        "config": vars(args),
        "history": history,
    }
    args.metrics.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Best epoch    : "
        f"{best_epoch if best_epoch else 'NONE'}"
    )
    print(f"Baseline Q    : {baseline_quality:.6f}")
    print(f"Required Q    : <= {required_quality:.6f}")
    print(f"Baseline Top1 : {e00['zero']['top1']:.6f}")
    print(f"Required Top1 : >= {required_top1:.6f}")
    print(
        f"Max unsafechg : <= "
        f"{args.max_unsafe_change_rate:.6f}"
    )
    if best_epoch:
        print(f"Best Q        : {best_quality:.6f}")
        print(f"Best Top1     : {best_top1:.6f}")
    if peak is not None:
        print(
            f"Peak CUDA mem : "
            f"{peak / (1024 ** 3):.2f} GiB"
        )
    print(
        f"Checkpoint    : "
        f"{args.output if best_epoch else 'NOT WRITTEN'}"
    )
    print(f"Metrics       : {args.metrics}")
    print(f"Status        : {status}")


if __name__ == "__main__":
    main()
