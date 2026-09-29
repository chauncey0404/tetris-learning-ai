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


HOLES = feature_index("holes_after")
HOLE_DELTA = feature_index("hole_delta")
T_DESTROYED = feature_index("t_opportunity_destroyed")
T_DEFERRED = feature_index("t_cashout_deferred")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train V1.2A causal state interaction residual")
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
        default=Path(r"models\tetrio_expert_v1_2_stateful_100k.pt"),
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_2_stateful_100k_training.json"),
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.01)
    p.add_argument("--risky-expert-row-weight", type=float, default=0.35)
    p.add_argument("--dominance-weight", type=float, default=0.05)
    p.add_argument("--dominance-margin", type=float, default=0.20)
    p.add_argument("--residual-l2-weight", type=float, default=0.003)
    p.add_argument("--state-hidden-size", type=int, default=64)
    p.add_argument("--state-max-adjustment", type=float, default=1.0)
    p.add_argument("--max-top1-drop", type=float, default=0.005)
    p.add_argument("--min-quality-improvement", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument(
        "--recovery-jsonl",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_recovery_states.jsonl"),
        help=(
            "Recorded for experiment provenance only. The frozen V1.1 500K "
            "base already learned these recovery pairs; V1.2A does not invent "
            "battle state for old recovery events."
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


def _to(a, device):
    return torch.from_numpy(a).to(device=device, non_blocking=device.type == "cuda")


def _weighted_ce(scores, target, mask, row_weight, label_smoothing):
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    logp = F.log_softmax(masked, dim=1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    if label_smoothing:
        valid_sum = logp.masked_fill(~mask, 0.0).sum(dim=1)
        valid_n = mask.sum(dim=1).to(scores.dtype)
        smooth = -valid_sum / valid_n
        per = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        per = nll
    return (per * row_weight).sum() / row_weight.sum().clamp_min(1e-8)


def _metrics(final_scores, raw_features, candidate_hold, inference_mask, expert_local, expert_hold, expert_in_shortlist):
    masked = final_scores.masked_fill(~inference_mask, torch.finfo(final_scores.dtype).min)
    order = torch.argsort(masked, dim=1, descending=True)
    pred = order[:, 0]
    recall = expert_in_shortlist.bool()
    correct = (pred == expert_local) & recall
    top3 = ((order[:, :min(3, order.shape[1])] == expert_local[:, None]).any(dim=1) & recall)
    rank_match = order == expert_local[:, None]
    rank = rank_match.float().argmax(dim=1) + 1
    mrr = torch.where(recall, 1.0 / rank.float(), torch.zeros_like(rank, dtype=torch.float32))
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
        "top1": float(correct.float().mean()),
        "top3": float(top3.float().mean()),
        "mrr": float(mrr.mean()),
        "branch_acc": float(branch_acc),
        "shortlist_recall": float(recall.float().mean()),
        "avoidable_rate": float(avoidable.float().mean()),
        "t_destroyed_rate": float(t_destroyed.float().mean()),
        "t_cashout_deferred_rate": float(t_deferred.float().mean()),
    }, pred


def quality_cost(m: dict) -> float:
    return float(m["avoidable_rate"] + 0.50*m["t_destroyed_rate"] + 0.25*m["t_cashout_deferred_rate"])


def count_recovery(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _pair_paths(future_dir: Path, state_dir: Path):
    future_paths = future_shard_paths(future_dir)
    pairs = []
    for fp in future_paths:
        sp = state_dir / fp.name
        if not sp.is_file():
            raise FileNotFoundError(f"Missing matching V1.2 state shard: {sp}")
        pairs.append((fp, sp))
    return pairs


def run_training_epoch(*, model, future_dir, state_dir, device, batch_size, optimizer, seed, args):
    model.stateful.train(True)
    pairs = _pair_paths(future_dir, state_dir)
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    sums = {"loss":0.0,"ce":0.0,"dominance":0.0,"residual_l2":0.0,"rows":0.0}
    started = time.perf_counter()
    for fp, sp in pairs:
        future = load_future_shard(fp)
        state = load_stateful_shard(sp)
        for batch in batches_from_stateful_pair(future=future, stateful=state, batch_size=batch_size, rng=rng):
            f = batch.future
            base = _to(f.base_scores, device).float()
            feat = _to(f.raw_features, device).float()
            hold = _to(f.candidate_use_hold, device).bool()
            valid = _to(f.valid_mask.astype(np.uint8), device).bool()
            inf = _to(f.inference_mask.astype(np.uint8), device).bool()
            target = _to(f.expert_local, device).long()
            recall = _to(f.expert_in_shortlist, device).bool()
            battle = _to(batch.battle_state, device).float()

            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=(torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16),
                enabled=device.type == "cuda",
            ):
                v11, _ = model.base.final_scores(
                    base_scores=base,
                    raw_features=feat,
                    candidate_use_hold=hold,
                    mask=valid,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=(torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16),
                enabled=device.type == "cuda",
            ):
                residual = model.stateful(
                    v11_scores=v11,
                    raw_future_features=feat,
                    candidate_use_hold=hold,
                    battle_state=battle,
                    mask=valid,
                )
                final = (v11 + residual).masked_fill(~valid, torch.finfo(v11.dtype).min)
                rows = torch.arange(base.shape[0], device=device)
                expert_holes = feat[rows, target, HOLES]
                expert_delta = feat[rows, target, HOLE_DELTA]
                before_holes = expert_holes - expert_delta
                inf_holes = feat[..., HOLES].masked_fill(~inf, 255.0)
                min_holes = inf_holes.min(dim=1).values
                expert_risky = (expert_delta > 0) & (min_holes <= before_holes)
                row_weight = recall.to(final.dtype)
                row_weight = torch.where(
                    expert_risky & recall,
                    torch.full_like(row_weight, float(args.risky_expert_row_weight)),
                    row_weight,
                )
                ce = _weighted_ce(final, target, valid, row_weight, args.label_smoothing)
                dom, _ = dominance_margin_loss(final, feat, inf, margin=args.dominance_margin)
                res_l2 = residual.masked_select(valid).pow(2).mean() if valid.any() else residual.sum()*0.0
                loss = ce + args.dominance_weight*dom + args.residual_l2_weight*res_l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.stateful.parameters(), 1.0)
            optimizer.step()
            b = base.shape[0]
            sums["loss"] += float(loss.detach())*b
            sums["ce"] += float(ce.detach())*b
            sums["dominance"] += float(dom.detach())*b
            sums["residual_l2"] += float(res_l2.detach())*b
            sums["rows"] += b
    rows = sums.pop("rows")
    elapsed = time.perf_counter()-started
    out = {k:v/rows for k,v in sums.items()}
    out["rows_per_second"] = rows/max(elapsed,1e-9)
    return out


def evaluate_ablation(*, model, future_dir, state_dir, device, batch_size):
    model.eval()
    true_sums = {k:0.0 for k in ("top1","top3","mrr","branch_acc","shortlist_recall","avoidable_rate","t_destroyed_rate","t_cashout_deferred_rate")}
    zero_sums = dict(true_sums)
    rows_total = 0
    changed = 0
    max_true_residual = 0.0
    max_zero_residual = 0.0
    max_e00_score_delta = 0.0
    for fp, sp in _pair_paths(future_dir, state_dir):
        future = load_future_shard(fp)
        state = load_stateful_shard(sp)
        for batch in batches_from_stateful_pair(future=future, stateful=state, batch_size=batch_size, rng=None):
            f = batch.future
            base = _to(f.base_scores, device).float()
            feat = _to(f.raw_features, device).float()
            hold = _to(f.candidate_use_hold, device).bool()
            valid = _to(f.valid_mask.astype(np.uint8), device).bool()
            inf = _to(f.inference_mask.astype(np.uint8), device).bool()
            target = _to(f.expert_local, device).long()
            expert_hold = _to(f.expert_use_hold, device).bool()
            recall = _to(f.expert_in_shortlist, device).bool()
            battle = _to(batch.battle_state, device).float()
            zero_battle = torch.zeros_like(battle)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=(torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16),
                enabled=device.type == "cuda",
            ):
                v11, _ = model.base.final_scores(
                    base_scores=base,
                    raw_features=feat,
                    candidate_use_hold=hold,
                    mask=valid,
                )
                r_true = model.stateful(
                    v11_scores=v11, raw_future_features=feat,
                    candidate_use_hold=hold, battle_state=battle, mask=valid,
                )
                r_zero = model.stateful(
                    v11_scores=v11, raw_future_features=feat,
                    candidate_use_hold=hold, battle_state=zero_battle, mask=valid,
                )
                true_final = (v11+r_true).masked_fill(~valid, torch.finfo(v11.dtype).min)
                zero_final = (v11+r_zero).masked_fill(~valid, torch.finfo(v11.dtype).min)
            true_m, true_pred = _metrics(true_final, feat, hold, inf, target, expert_hold, recall)
            zero_m, zero_pred = _metrics(zero_final, feat, hold, inf, target, expert_hold, recall)
            b = base.shape[0]
            for k in true_sums:
                true_sums[k] += true_m[k]*b
                zero_sums[k] += zero_m[k]*b
            changed += int((true_pred != zero_pred).sum().item())
            rows_total += b
            max_true_residual = max(max_true_residual, float(r_true.abs().max().item()))
            max_zero_residual = max(max_zero_residual, float(r_zero.abs().max().item()))
            max_e00_score_delta = max(max_e00_score_delta, float((zero_final-v11).masked_select(valid).abs().max().item()))
    true = {k:v/rows_total for k,v in true_sums.items()}
    zero = {k:v/rows_total for k,v in zero_sums.items()}
    return {
        "true": true,
        "zero": zero,
        "true_quality_cost": quality_cost(true),
        "zero_quality_cost": quality_cost(zero),
        "quality_delta_true_minus_zero": quality_cost(true)-quality_cost(zero),
        "top1_delta_true_minus_zero": true["top1"]-zero["top1"],
        "decision_change_rate": changed/rows_total,
        "max_true_residual": max_true_residual,
        "max_zero_residual": max_zero_residual,
        "max_zero_score_delta_vs_v11": max_e00_score_delta,
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

    base, base_ckpt = load_expert_v1_1(args.init_v1_1, device=device)
    base_cfg = base_ckpt.get("config", {})
    model = TetrioExpertV12Network(
        v11_reranker_hidden_size=int(base_cfg.get("reranker_hidden_size",96)),
        v11_max_adjustment=float(base_cfg.get("max_adjustment",2.0)),
        state_hidden_size=args.state_hidden_size,
        state_max_adjustment=args.state_max_adjustment,
    ).to(device)
    model.base.load_state_dict(base.state_dict(), strict=True)
    model.freeze_base()
    del base

    optimizer = torch.optim.AdamW(
        model.stateful.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    recovery_count = count_recovery(args.recovery_jsonl)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)

    print("="*108)
    print("TETR.IO EXPERT V1.2A — CAUSAL STATE INTERACTION RESIDUAL")
    print("="*108)
    print(f"Base V1.1    : {args.init_v1_1} epoch={base_ckpt.get('epoch')}")
    print(f"Train future : {args.train_cache}")
    print(f"Train state  : {args.train_state_cache}")
    print(f"Val future   : {args.val_cache}")
    print(f"Val state    : {args.val_state_cache}")
    print(f"State fields : {', '.join(STATEFUL_FEATURE_NAMES)}")
    print(f"State adjust : ±{args.state_max_adjustment}")
    print(f"Recovery     : {recovery_count:,} pair(s), weight={args.recovery_weight} (frozen-base provenance only)")
    print("Recovery V1.2 loss: DISABLED — historical recovery JSONL has no audited battle state")
    print(f"Device       : {device}")
    if device.type == "cuda": print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print()

    e00 = evaluate_ablation(
        model=model, future_dir=args.val_cache, state_dir=args.val_state_cache,
        device=device, batch_size=args.batch_size,
    )
    if e00["max_true_residual"] != 0.0 or e00["max_zero_residual"] != 0.0 or e00["decision_change_rate"] != 0.0:
        raise SystemExit(f"E00 PARITY FAIL: {e00}")
    print(
        "E00 V1.1 PARITY PASS | "
        f"top1={e00['zero']['top1']:.4f} top3={e00['zero']['top3']:.4f} "
        f"branch={e00['zero']['branch_acc']:.4f} avoid={e00['zero']['avoidable_rate']:.4f} "
        f"Tdestroy={e00['zero']['t_destroyed_rate']:.4f} Tdefer={e00['zero']['t_cashout_deferred_rate']:.4f} "
        "maxResidual=0 decisionChange=0"
    )

    baseline_quality = e00["zero_quality_cost"]
    top1_floor = e00["zero"]["top1"] - args.max_top1_drop
    required_quality = baseline_quality - args.min_quality_improvement
    best_quality = baseline_quality
    best_epoch = 0
    history = [{"epoch":0,"kind":"frozen_v1_1_e00","ablation":e00}]
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, args.epochs+1):
        train = run_training_epoch(
            model=model, future_dir=args.train_cache, state_dir=args.train_state_cache,
            device=device, batch_size=args.batch_size, optimizer=optimizer,
            seed=args.seed+epoch, args=args,
        )
        ab = evaluate_ablation(
            model=model, future_dir=args.val_cache, state_dir=args.val_state_cache,
            device=device, batch_size=args.batch_size,
        )
        eligible = (
            ab["true"]["top1"] >= top1_floor
            and ab["true_quality_cost"] <= required_quality
            and ab["decision_change_rate"] > 0.0
            and ab["max_zero_residual"] == 0.0
        )
        item = {
            "epoch":epoch,"train":train,"ablation":ab,"eligible":eligible,
            "quality_cost":ab["true_quality_cost"],
        }
        history.append(item)
        if eligible and ab["true_quality_cost"] < best_quality:
            best_quality = ab["true_quality_cost"]
            best_epoch = epoch
            torch.save(
                {
                    "format":"tetrio_expert_v1_2_stateful",
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "config":vars(args),
                    "base_v1_1":str(args.init_v1_1),
                    "base_v1_1_epoch":base_ckpt.get("epoch"),
                    "base_v1_1_config":base_cfg,
                    "state_feature_names":list(STATEFUL_FEATURE_NAMES),
                    "metrics":item,
                    "selection_contract":(
                        "TRUE-state Top1 >= V1.1 baseline-max_top1_drop; "
                        "TRUE quality improves baseline by min_quality_improvement; "
                        "decision_change_rate>0; ZERO-state residual stays exactly zero"
                    ),
                    "status":"RESEARCH PILOT (not Champion)",
                },
                args.output,
            )
        print(
            f"E{epoch:02d}/{args.epochs} | train {train['rows_per_second']:.0f} rows/s | "
            f"TRUE top1={ab['true']['top1']:.4f} top3={ab['true']['top3']:.4f} "
            f"branch={ab['true']['branch_acc']:.4f} avoid={ab['true']['avoidable_rate']:.4f} "
            f"Tdestroy={ab['true']['t_destroyed_rate']:.4f} Tdefer={ab['true']['t_cashout_deferred_rate']:.4f} "
            f"Q={ab['true_quality_cost']:.6f} | ZERO top1={ab['zero']['top1']:.4f} "
            f"Q={ab['zero_quality_cost']:.6f} | change={ab['decision_change_rate']:.4f} "
            f"{'ELIGIBLE' if eligible else 'NO_PROMOTION'}"
        )

    elapsed = time.perf_counter()-started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    status = "RESEARCH PILOT (not Champion)" if best_epoch else "NON_PROMOTION"
    summary = {
        "format":"tetrio_expert_v1_2_stateful_training",
        "status":status,
        "base_v1_1":str(args.init_v1_1),
        "base_v1_1_epoch":base_ckpt.get("epoch"),
        "e00":e00,
        "baseline_quality_cost":baseline_quality,
        "required_quality_cost":required_quality,
        "top1_floor":top1_floor,
        "best_epoch":best_epoch,
        "best_quality_cost":best_quality if best_epoch else None,
        "checkpoint":str(args.output) if best_epoch else None,
        "recovery_jsonl":str(args.recovery_jsonl) if args.recovery_jsonl else None,
        "recovery_pairs":recovery_count,
        "recovery_weight_requested":args.recovery_weight,
        "recovery_v1_2_loss_enabled":False,
        "seconds":elapsed,
        "peak_cuda_memory_bytes":peak,
        "history":history,
    }
    args.metrics.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print()
    print(f"Best epoch   : {best_epoch if best_epoch else 'NONE'}")
    print(f"Baseline Q   : {baseline_quality:.6f}")
    print(f"Required Q   : <= {required_quality:.6f}")
    if best_epoch: print(f"Best Q       : {best_quality:.6f}")
    if peak is not None: print(f"Peak CUDA mem: {peak/(1024**3):.2f} GiB")
    print(f"Checkpoint   : {args.output if best_epoch else 'NOT WRITTEN'}")
    print(f"Metrics      : {args.metrics}")
    print(f"Status       : {status}")


if __name__ == "__main__":
    main()
