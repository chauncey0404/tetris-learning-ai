from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch

from tetrio.network.checkpoint import load_expert_v1_1
from tetrio.network.future_cache import future_shard_paths, load_future_shard
from tetrio.network.model_v1_2 import TetrioExpertV12Network
from tetrio.network.stateful_cache import (
    batches_from_stateful_pair,
    load_stateful_shard,
)
from tetrio.stateful.features import STATEFUL_FEATURE_NAMES
from tetrio.tools.train_expert_v1_2 import (
    _metrics,
    _to,
    quality_cost,
    run_training_epoch,
    seed_all,
)


GROUPS = {
    "all": tuple(range(7)),
    "combo_btb": (0, 1),
    "previous_outcome": (2, 3, 4, 5, 6),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Diagnostic-only state identifiability audit. Retrains the original "
            "V1.2A 100K residual in memory, then compares TRUE, SHUFFLED and "
            "ZERO state under fixed validation rows. No checkpoint is promoted."
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
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument(
        "--min-true-shuffled-top1-gap",
        type=float,
        default=0.0020,
        help="Predeclared identifiability gap: TRUE - SHUFFLED Top1.",
    )
    p.add_argument(
        "--required-consecutive-epochs",
        type=int,
        default=3,
        help="Number of consecutive epochs that must pass the identifiability gap.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"artifacts\tetrio\state_identifiability_audit.json"
        ),
    )
    return p.parse_args()


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


def _pair_paths(
    future_dir: Path,
    state_dir: Path,
) -> list[tuple[Path, Path]]:
    pairs = []
    for fp in future_shard_paths(future_dir):
        sp = state_dir / fp.name
        if not sp.is_file():
            raise FileNotFoundError(f"Missing matching state shard: {sp}")
        pairs.append((fp, sp))
    return pairs


def _load_validation_state(
    state_dir: Path,
    future_dir: Path,
) -> np.ndarray:
    rows = []
    for fp, sp in _pair_paths(future_dir, state_dir):
        future = load_future_shard(fp)
        state = load_stateful_shard(sp)
        if int(future["expert_local"].shape[0]) != int(
            state["state_features"].shape[0]
        ):
            raise RuntimeError(f"row mismatch for {fp.name}")
        rows.append(
            np.asarray(state["state_features"], dtype=np.float32)
        )
    return np.ascontiguousarray(np.concatenate(rows, axis=0))


def deranged_permutation(n: int, seed: int) -> np.ndarray:
    if n < 2:
        raise ValueError("need at least two rows for shuffled-state audit")
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    fixed = np.flatnonzero(p == np.arange(n))
    # Pairwise swap fixed points; if one remains, swap it with any other row.
    for i in range(0, len(fixed) - 1, 2):
        a = int(fixed[i])
        b = int(fixed[i + 1])
        p[a], p[b] = p[b], p[a]
    if len(fixed) % 2:
        a = int(fixed[-1])
        b = 0 if a != 0 else 1
        p[a], p[b] = p[b], p[a]
    if np.any(p == np.arange(n)):
        raise RuntimeError("failed to construct derangement")
    return p


def _mask_group(
    state: torch.Tensor,
    group: str,
) -> torch.Tensor:
    if group == "all":
        return state
    keep = GROUPS[group]
    out = torch.zeros_like(state)
    out[:, list(keep)] = state[:, list(keep)]
    return out


def evaluate_modes(
    *,
    model: TetrioExpertV12Network,
    future_dir: Path,
    state_dir: Path,
    shuffled_all: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    model.eval()
    modes = (
        "zero",
        "true_all",
        "shuffled_all",
        "true_combo_btb",
        "shuffled_combo_btb",
        "true_previous_outcome",
        "shuffled_previous_outcome",
    )
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
    sums = {
        mode: {k: 0.0 for k in metric_keys}
        for mode in modes
    }
    changed = {mode: 0 for mode in modes if mode != "zero"}
    rows_total = 0
    global_offset = 0

    for fp, sp in _pair_paths(future_dir, state_dir):
        future = load_future_shard(fp)
        state = load_stateful_shard(sp)

        for batch in batches_from_stateful_pair(
            future=future,
            stateful=state,
            batch_size=batch_size,
            rng=None,
        ):
            f = batch.future
            b = int(f.base_scores.shape[0])
            base = _to(f.base_scores, device).float()
            feat = _to(f.raw_features, device).float()
            hold = _to(f.candidate_use_hold, device).bool()
            valid = _to(f.valid_mask.astype(np.uint8), device).bool()
            inf = _to(f.inference_mask.astype(np.uint8), device).bool()
            target = _to(f.expert_local, device).long()
            expert_hold = _to(f.expert_use_hold, device).bool()
            recall = _to(f.expert_in_shortlist, device).bool()

            true_state = _to(batch.battle_state, device).float()
            shuffled_state = torch.from_numpy(
                shuffled_all[global_offset:global_offset + b]
            ).to(device=device, dtype=torch.float32)
            global_offset += b

            battle_by_mode = {
                "zero": torch.zeros_like(true_state),
                "true_all": true_state,
                "shuffled_all": shuffled_state,
                "true_combo_btb": _mask_group(
                    true_state, "combo_btb"
                ),
                "shuffled_combo_btb": _mask_group(
                    shuffled_state, "combo_btb"
                ),
                "true_previous_outcome": _mask_group(
                    true_state, "previous_outcome"
                ),
                "shuffled_previous_outcome": _mask_group(
                    shuffled_state, "previous_outcome"
                ),
            }

            with torch.inference_mode(), _autocast(device):
                v11, _ = model.base.final_scores(
                    base_scores=base,
                    raw_features=feat,
                    candidate_use_hold=hold,
                    mask=valid,
                )

                zero_final = None
                zero_pred = None
                for mode in modes:
                    residual = model.stateful(
                        v11_scores=v11,
                        raw_future_features=feat,
                        candidate_use_hold=hold,
                        battle_state=battle_by_mode[mode],
                        mask=valid,
                    )
                    final = (v11 + residual).masked_fill(
                        ~valid,
                        torch.finfo(v11.dtype).min,
                    )
                    m, pred = _metrics(
                        final,
                        feat,
                        hold,
                        inf,
                        target,
                        expert_hold,
                        recall,
                    )
                    for key in metric_keys:
                        sums[mode][key] += float(m[key]) * b

                    if mode == "zero":
                        zero_final = final
                        zero_pred = pred
                    else:
                        changed[mode] += int(
                            (pred != zero_pred).sum().item()
                        )

            rows_total += b

    if global_offset != int(shuffled_all.shape[0]):
        raise RuntimeError(
            f"validation row traversal mismatch: "
            f"{global_offset} != {shuffled_all.shape[0]}"
        )

    out = {}
    for mode in modes:
        metrics = {
            k: sums[mode][k] / rows_total
            for k in metric_keys
        }
        out[mode] = {
            "metrics": metrics,
            "quality_cost": quality_cost(metrics),
            "decision_change_vs_zero": (
                0.0
                if mode == "zero"
                else changed[mode] / rows_total
            ),
        }

    for group in ("all", "combo_btb", "previous_outcome"):
        true_key = f"true_{group}"
        shuffled_key = f"shuffled_{group}"
        out[f"{group}_identifiability"] = {
            "top1_true_minus_shuffled": (
                out[true_key]["metrics"]["top1"]
                - out[shuffled_key]["metrics"]["top1"]
            ),
            "branch_true_minus_shuffled": (
                out[true_key]["metrics"]["branch_acc"]
                - out[shuffled_key]["metrics"]["branch_acc"]
            ),
            "quality_true_minus_shuffled": (
                out[true_key]["quality_cost"]
                - out[shuffled_key]["quality_cost"]
            ),
        }
    return out


def consecutive_pass(
    history: list[dict],
    *,
    group: str,
    gap: float,
    required: int,
) -> tuple[bool, list[int]]:
    key = f"{group}_identifiability"
    streak: list[int] = []
    best: list[int] = []
    for row in history:
        if int(row["epoch"]) == 0:
            continue
        if (
            row["eval"][key]["top1_true_minus_shuffled"]
            >= float(gap)
        ):
            streak.append(int(row["epoch"]))
            if len(streak) > len(best):
                best = list(streak)
        else:
            streak = []
    return len(best) >= int(required), best


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    seed_all(args.seed)
    random.seed(args.seed)
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

    val_state = _load_validation_state(
        args.val_state_cache,
        args.val_cache,
    )
    perm = deranged_permutation(
        int(val_state.shape[0]),
        args.seed + 100_003,
    )
    shuffled = np.ascontiguousarray(val_state[perm])

    print("=" * 112)
    print("TETR.IO STATE IDENTIFIABILITY AUDIT — TRUE vs SHUFFLED vs ZERO")
    print("=" * 112)
    print(f"Base          : {args.init_v1_1} epoch={base_ckpt.get('epoch')}")
    print(f"Train future  : {args.train_cache}")
    print(f"Train state   : {args.train_state_cache}")
    print(f"Val future    : {args.val_cache}")
    print(f"Val state     : {args.val_state_cache}")
    print(f"State fields  : {', '.join(STATEFUL_FEATURE_NAMES)}")
    print(
        "Primary gate  : TRUE_ALL - SHUFFLED_ALL Top1 "
        f">= {args.min_true_shuffled_top1_gap:+.4f} for "
        f"{args.required_consecutive_epochs} consecutive epochs"
    )
    print("Checkpoint    : NEVER WRITTEN (diagnostic only)")
    print()

    e00 = evaluate_modes(
        model=model,
        future_dir=args.val_cache,
        state_dir=args.val_state_cache,
        shuffled_all=shuffled,
        device=device,
        batch_size=args.batch_size,
    )
    history = [{"epoch": 0, "eval": e00}]
    print(
        "E00 | "
        f"ZERO={e00['zero']['metrics']['top1']:.4f} "
        f"TRUE={e00['true_all']['metrics']['top1']:.4f} "
        f"SHUF={e00['shuffled_all']['metrics']['top1']:.4f}"
    )

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
        ev = evaluate_modes(
            model=model,
            future_dir=args.val_cache,
            state_dir=args.val_state_cache,
            shuffled_all=shuffled,
            device=device,
            batch_size=args.batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train,
                "eval": ev,
            }
        )

        print(
            f"E{epoch:02d}/{args.epochs} | "
            f"train={train['rows_per_second']:.0f} rows/s | "
            f"ALL T/S/Z="
            f"{ev['true_all']['metrics']['top1']:.4f}/"
            f"{ev['shuffled_all']['metrics']['top1']:.4f}/"
            f"{ev['zero']['metrics']['top1']:.4f} "
            f"gap={ev['all_identifiability']['top1_true_minus_shuffled']:+.4f} | "
            f"COMBO gap="
            f"{ev['combo_btb_identifiability']['top1_true_minus_shuffled']:+.4f} | "
            f"PREV gap="
            f"{ev['previous_outcome_identifiability']['top1_true_minus_shuffled']:+.4f}"
        )

    group_status = {}
    for group in ("all", "combo_btb", "previous_outcome"):
        passed, streak = consecutive_pass(
            history,
            group=group,
            gap=args.min_true_shuffled_top1_gap,
            required=args.required_consecutive_epochs,
        )
        group_status[group] = {
            "passed": passed,
            "best_consecutive_passing_epochs": streak,
        }

    status = (
        "IDENTIFIABLE"
        if group_status["all"]["passed"]
        else "NOT_IDENTIFIABLE"
    )

    report = {
        "format": "tetrio_state_identifiability_audit",
        "status": status,
        "base": str(args.init_v1_1),
        "base_epoch": base_ckpt.get("epoch"),
        "seed": int(args.seed),
        "state_feature_names": list(STATEFUL_FEATURE_NAMES),
        "groups": {
            k: list(v) for k, v in GROUPS.items()
        },
        "shuffle": {
            "kind": "global deterministic derangement",
            "rows": int(val_state.shape[0]),
            "fixed_points": int(
                np.count_nonzero(perm == np.arange(len(perm)))
            ),
        },
        "gate": {
            "min_true_shuffled_top1_gap": (
                args.min_true_shuffled_top1_gap
            ),
            "required_consecutive_epochs": (
                args.required_consecutive_epochs
            ),
        },
        "group_status": group_status,
        "history": history,
        "checkpoint_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 112)
    print("IDENTIFIABILITY SUMMARY")
    print("=" * 112)
    for group, item in group_status.items():
        print(
            f"{group:18s}: "
            f"{'PASS' if item['passed'] else 'FAIL'} "
            f"streak={item['best_consecutive_passing_epochs']}"
        )
    print(f"Status : {status}")
    print(f"Report : {args.output}")


if __name__ == "__main__":
    main()
