from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tetrio.network.checkpoint import load_expert_v1_2
from tetrio.tools.train_expert_stateful_safety import (
    evaluate_safety_ablation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate TRUE/ZERO ablation and structural safety diagnostics "
            "for a trained stateful safety checkpoint."
        )
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--future-cache", type=Path, required=True)
    p.add_argument("--state-cache", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"artifacts\tetrio\expert_stateful_safety_ablation.json"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, ckpt = load_expert_v1_2(
        args.checkpoint,
        device=device,
    )
    ab = evaluate_safety_ablation(
        model=model,
        future_dir=args.future_cache,
        state_dir=args.state_cache,
        device=device,
        batch_size=args.batch_size,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "training_objective": ckpt.get("training_objective"),
        "ablation": ab,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    true = ab["true"]
    zero = ab["zero"]
    cr = ab["change_rates"]
    rel = ab["expert_relation_counts"]

    print("=" * 112)
    print("TETR.IO EXPERT — STATEFUL SAFETY TRUE/ZERO ABLATION")
    print("=" * 112)
    print(
        f"TRUE: top1={true['top1']:.4f} "
        f"top3={true['top3']:.4f} "
        f"branch={true['branch_acc']:.4f} "
        f"avoid={true['avoidable_rate']:.4f} "
        f"Tdestroy={true['t_destroyed_rate']:.4f} "
        f"Tdefer={true['t_cashout_deferred_rate']:.4f} "
        f"Q={ab['true_quality_cost']:.6f}"
    )
    print(
        f"ZERO: top1={zero['top1']:.4f} "
        f"top3={zero['top3']:.4f} "
        f"branch={zero['branch_acc']:.4f} "
        f"avoid={zero['avoidable_rate']:.4f} "
        f"Tdestroy={zero['t_destroyed_rate']:.4f} "
        f"Tdefer={zero['t_cashout_deferred_rate']:.4f} "
        f"Q={ab['zero_quality_cost']:.6f}"
    )
    print(
        f"ΔTop1 TRUE-ZERO : "
        f"{ab['top1_delta_true_minus_zero']:+.6f}"
    )
    print(
        f"ΔQ TRUE-ZERO    : "
        f"{ab['quality_delta_true_minus_zero']:+.6f} "
        "(negative is better)"
    )
    print(
        f"Decision change : {ab['decision_change_rate']:.4f} "
        f"safe={ab['safe_change_rate']:.4f} "
        f"unsafe={ab['unsafe_change_rate']:.4f} "
        f"ambiguous={ab['ambiguous_change_rate']:.4f}"
    )
    print(
        "Changed to expert: "
        f"{cr.get('changed_to_expert_rate', 0.0):.4f}"
    )
    print(
        "Expert relation   : "
        f"agree={rel.get('agreement_rows', 0):,} "
        f"safe={rel.get('safe_expert_rows', 0):,} "
        f"unsafe={rel.get('unsafe_expert_rows', 0):,} "
        f"ambig={rel.get('ambiguous_rows', 0):,}"
    )
    print(f"ZERO max residual: {ab['max_zero_residual']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
