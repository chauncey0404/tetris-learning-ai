from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tetrio.network.checkpoint import (
    load_expert_v1,
    load_expert_v1_1,
)
from tetrio.rollout.batched import (
    BatchedRolloutConfig,
    run_batched_v1,
    run_batched_v11,
)
from tetrio.tools.watch_expert_v0 import parse_seed_spec
from tetrio.tools.watch_expert_v1 import ExpertV1Rollout
from tetrio.tools.watch_expert_v1_1 import ExpertV11Rollout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Paired high-throughput comparison of Expert V1 and V1.1. "
            "CPU reachability is process-parallel and GPU scoring is batched "
            "across multiple deterministic seed trajectories."
        )
    )
    p.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_joint_100k.pt"),
    )
    p.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_1_future_20k.pt"),
    )
    p.add_argument("--seeds", default="9051-9070")
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--reference-audit-every", type=int, default=250)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--state-batch", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=512)
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument(
        "--parity-pieces",
        type=int,
        default=0,
        help=(
            "Before the A/B run, compare old sequential vs new batched "
            "placement trace for this many pieces on the first seed. "
            "0 disables."
        ),
    )
    p.add_argument(
        "--parity-only",
        action="store_true",
        help="Run parity checks and exit.",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_model_comparison_9051_9070.json"),
    )
    return p.parse_args()


def _config(args, *, max_pieces: int, collect_trace: bool) -> BatchedRolloutConfig:
    return BatchedRolloutConfig(
        max_pieces=max_pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        workers=args.workers,
        state_batch=args.state_batch,
        progress_every=args.progress_every,
        top_overall=args.top_overall,
        top_per_branch=args.top_per_branch,
        collect_trace=collect_trace,
    )


def _trace_key(item: dict) -> tuple:
    return (
        item["active"],
        item["hold_before"],
        bool(item["use_hold"]),
        item["branch_mode"],
        item["piece"],
        int(item["rotation"]),
        int(item["x"]),
        int(item["y"]),
        int(item["lines"]),
    )


def _sequential_trace_v1(model, device, seed: int, args, pieces: int) -> list[dict]:
    s = ExpertV1Rollout(
        model,
        device=device,
        seed=seed,
        max_pieces=pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        hold_threshold=0.5,
    )
    trace = []
    while not s.game_over and len(trace) < pieces:
        decision = s.pending_decision
        if decision is None:
            break
        chosen = decision.chosen
        trace.append(
            {
                "active": decision.active,
                "hold_before": decision.hold_before,
                "use_hold": bool(decision.use_hold),
                "branch_mode": decision.branch.mode,
                "piece": chosen.state.piece,
                "rotation": int(chosen.state.rotation) % 4,
                "x": int(chosen.state.x),
                "y": int(chosen.state.y),
                "lines": int(chosen.lines),
            }
        )
        if not s.step():
            break
    return trace


def _sequential_trace_v11(model, device, seed: int, args, pieces: int) -> list[dict]:
    s = ExpertV11Rollout(
        model,
        device=device,
        seed=seed,
        max_pieces=pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        hold_threshold=0.5,
        top_overall=args.top_overall,
        top_per_branch=args.top_per_branch,
    )
    trace = []
    while not s.game_over and len(trace) < pieces:
        decision = s.pending_decision
        if decision is None:
            break
        chosen = decision.chosen
        trace.append(
            {
                "active": decision.active,
                "hold_before": decision.hold_before,
                "use_hold": bool(decision.use_hold),
                "branch_mode": decision.branch.mode,
                "piece": chosen.state.piece,
                "rotation": int(chosen.state.rotation) % 4,
                "x": int(chosen.state.x),
                "y": int(chosen.state.y),
                "lines": int(chosen.lines),
            }
        )
        if not s.step():
            break
    return trace


def _check_trace(name: str, sequential: list[dict], batched: list[dict]) -> None:
    n = min(len(sequential), len(batched))
    for i in range(n):
        if _trace_key(sequential[i]) != _trace_key(batched[i]):
            raise RuntimeError(
                f"{name} parity failure at move {i+1}: "
                f"sequential={sequential[i]} batched={batched[i]}"
            )
    if len(sequential) != len(batched):
        raise RuntimeError(
            f"{name} parity length mismatch: "
            f"sequential={len(sequential)} batched={len(batched)}"
        )
    print(f"{name} parity PASS: {len(sequential)} move(s)")


def _run_parity(
    *,
    v1,
    v11,
    device,
    seeds: list[int],
    args,
    pieces: int,
) -> None:
    parity_seeds = [int(s) for s in seeds[: min(4, len(seeds))]]
    print("=" * 112)
    print("SEQUENTIAL ↔ BATCHED POLICY PARITY")
    print("=" * 112)
    print(f"Seeds  : {parity_seeds}")
    print(f"Pieces : {pieces}/seed")
    print(
        "Note   : up to 4 seeds are batched together so parity covers "
        "multi-state GPU batching, not only the one-state path."
    )
    print()

    seq_v1 = {
        seed: _sequential_trace_v1(v1, device, seed, args, pieces)
        for seed in parity_seeds
    }
    bat_v1_result = run_batched_v1(
        v1,
        seeds=parity_seeds,
        device=device,
        config=_config(args, max_pieces=pieces, collect_trace=True),
    )
    bat_v1 = {
        int(r["seed"]): r.get("trace", [])
        for r in bat_v1_result["results"]
    }
    for seed in parity_seeds:
        _check_trace(
            f"V1 seed={seed}",
            seq_v1[seed],
            bat_v1[seed],
        )

    seq_v11 = {
        seed: _sequential_trace_v11(v11, device, seed, args, pieces)
        for seed in parity_seeds
    }
    bat_v11_result = run_batched_v11(
        v11,
        seeds=parity_seeds,
        device=device,
        config=_config(args, max_pieces=pieces, collect_trace=True),
    )
    bat_v11 = {
        int(r["seed"]): r.get("trace", [])
        for r in bat_v11_result["results"]
    }
    for seed in parity_seeds:
        _check_trace(
            f"V1.1 seed={seed}",
            seq_v11[seed],
            bat_v11[seed],
        )


def _paired_summary(
    baseline: list[dict],
    candidate: list[dict],
) -> dict:
    by_seed_a = {int(r["seed"]): r for r in baseline}
    by_seed_b = {int(r["seed"]): r for r in candidate}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))

    piece_delta = np.asarray(
        [
            int(by_seed_b[s]["pieces"]) - int(by_seed_a[s]["pieces"])
            for s in seeds
        ],
        dtype=np.float64,
    )
    line_delta = np.asarray(
        [
            int(by_seed_b[s]["lines"]) - int(by_seed_a[s]["lines"])
            for s in seeds
        ],
        dtype=np.float64,
    )
    avoid_delta = np.asarray(
        [
            float(by_seed_b[s]["avoidable_hole_rate"])
            - float(by_seed_a[s]["avoidable_hole_rate"])
            for s in seeds
        ],
        dtype=np.float64,
    )
    max_hole_delta = np.asarray(
        [
            int(by_seed_b[s]["max_holes"]) - int(by_seed_a[s]["max_holes"])
            for s in seeds
        ],
        dtype=np.float64,
    )

    wins = int(np.sum(piece_delta > 0))
    ties = int(np.sum(piece_delta == 0))
    losses = int(np.sum(piece_delta < 0))

    return {
        "seeds": seeds,
        "wins_ties_losses_by_pieces": [wins, ties, losses],
        "mean_piece_delta": float(piece_delta.mean()) if len(piece_delta) else 0.0,
        "median_piece_delta": float(np.median(piece_delta)) if len(piece_delta) else 0.0,
        "mean_line_delta": float(line_delta.mean()) if len(line_delta) else 0.0,
        "mean_avoidable_rate_delta": (
            float(avoid_delta.mean()) if len(avoid_delta) else 0.0
        ),
        "mean_max_holes_delta": (
            float(max_hole_delta.mean()) if len(max_hole_delta) else 0.0
        ),
        "per_seed": [
            {
                "seed": int(s),
                "baseline_pieces": int(by_seed_a[s]["pieces"]),
                "candidate_pieces": int(by_seed_b[s]["pieces"]),
                "piece_delta": int(
                    by_seed_b[s]["pieces"] - by_seed_a[s]["pieces"]
                ),
                "baseline_lines": int(by_seed_a[s]["lines"]),
                "candidate_lines": int(by_seed_b[s]["lines"]),
                "line_delta": int(
                    by_seed_b[s]["lines"] - by_seed_a[s]["lines"]
                ),
                "baseline_avoidable_rate": float(
                    by_seed_a[s]["avoidable_hole_rate"]
                ),
                "candidate_avoidable_rate": float(
                    by_seed_b[s]["avoidable_hole_rate"]
                ),
                "baseline_max_holes": int(by_seed_a[s]["max_holes"]),
                "candidate_max_holes": int(by_seed_b[s]["max_holes"]),
            }
            for s in seeds
        ],
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    seeds = parse_seed_spec(args.seeds, 9051)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    v1, ckpt_v1 = load_expert_v1(
        args.baseline_checkpoint,
        device=device,
    )
    v11, ckpt_v11 = load_expert_v1_1(
        args.candidate_checkpoint,
        device=device,
    )

    print("=" * 112)
    print("TETR.IO EXPERT MODEL COMPARISON — STRICT PARITY + CPU PARALLEL")
    print("=" * 112)
    print(f"Baseline  : {args.baseline_checkpoint} epoch={ckpt_v1.get('epoch')}")
    print(f"Candidate : {args.candidate_checkpoint} epoch={ckpt_v11.get('epoch')}")
    print(f"Seeds     : {seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print(f"Workers   : {args.workers}")
    print(f"State batch: {args.state_batch}")
    print(f"Device    : {device}")
    if device.type == "cuda":
        print(f"GPU       : {torch.cuda.get_device_name(device)}")
    print()

    if args.parity_pieces > 0:
        _run_parity(
            v1=v1,
            v11=v11,
            device=device,
            seeds=seeds,
            args=args,
            pieces=int(args.parity_pieces),
        )
        print()
        if args.parity_only:
            return

    config = _config(
        args,
        max_pieces=args.max_pieces,
        collect_trace=False,
    )

    baseline = run_batched_v1(
        v1,
        seeds=seeds,
        device=device,
        config=config,
    )
    candidate = run_batched_v11(
        v11,
        seeds=seeds,
        device=device,
        config=config,
    )
    paired = _paired_summary(
        baseline["results"],
        candidate["results"],
    )

    report = {
        "format": "tetrio_expert_model_comparison",
        "status": "DEVELOPMENT A/B (not Champion qualification)",
        "baseline": {
            "checkpoint": str(args.baseline_checkpoint),
            "epoch": ckpt_v1.get("epoch"),
            **baseline,
        },
        "candidate": {
            "checkpoint": str(args.candidate_checkpoint),
            "epoch": ckpt_v11.get("epoch"),
            **candidate,
        },
        "paired": paired,
    }

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 112)
    print("PER-SEED PAIRED RESULTS")
    print("=" * 112)
    for row in paired["per_seed"]:
        print(
            f"Seed {row['seed']}: "
            f"V1={row['baseline_pieces']:,}p/{row['baseline_lines']:,}L "
            f"V1.1={row['candidate_pieces']:,}p/{row['candidate_lines']:,}L "
            f"ΔP={row['piece_delta']:+,} ΔL={row['line_delta']:+,} "
            f"avoid={row['baseline_avoidable_rate']:.4f}"
            f"→{row['candidate_avoidable_rate']:.4f} "
            f"maxHoles={row['baseline_max_holes']}"
            f"→{row['candidate_max_holes']}"
        )

    print()
    print("=" * 112)
    print("AGGREGATES")
    print("=" * 112)
    ba = baseline["aggregate"]
    ca = candidate["aggregate"]
    print(
        f"V1   : meanPieces={ba.get('mean_pieces', 0.0):.2f} "
        f"meanLines={ba.get('mean_lines', 0.0):.2f} "
        f"gameOvers={ba.get('game_overs', 0)} "
        f"avoid={ba.get('mean_avoidable_hole_rate', 0.0):.5f} "
        f"maxHoles={ba.get('mean_max_holes', 0.0):.2f}"
    )
    print(
        f"V1.1 : meanPieces={ca.get('mean_pieces', 0.0):.2f} "
        f"meanLines={ca.get('mean_lines', 0.0):.2f} "
        f"gameOvers={ca.get('game_overs', 0)} "
        f"avoid={ca.get('mean_avoidable_hole_rate', 0.0):.5f} "
        f"maxHoles={ca.get('mean_max_holes', 0.0):.2f} "
        f"Tdestroy={ca.get('mean_t_destroyed_rate', 0.0):.5f} "
        f"Tdefer={ca.get('mean_t_deferred_rate', 0.0):.5f}"
    )

    print()
    print("=" * 112)
    print("PAIRED SUMMARY")
    print("=" * 112)
    print(
        "Pieces W/T/L : "
        f"{paired['wins_ties_losses_by_pieces'][0]}/"
        f"{paired['wins_ties_losses_by_pieces'][1]}/"
        f"{paired['wins_ties_losses_by_pieces'][2]}"
    )
    print(f"Mean Δpieces : {paired['mean_piece_delta']:+.2f}")
    print(f"Median Δpieces: {paired['median_piece_delta']:+.2f}")
    print(f"Mean Δlines  : {paired['mean_line_delta']:+.2f}")
    print(
        "Mean Δavoidable rate: "
        f"{paired['mean_avoidable_rate_delta']:+.5f}"
    )
    print(
        f"Mean Δmax holes: {paired['mean_max_holes_delta']:+.2f}"
    )
    print()
    print(
        "Baseline runtime : "
        f"{baseline['runtime']['states_per_second']:.2f} states/s"
    )
    print(
        "Candidate runtime: "
        f"{candidate['runtime']['states_per_second']:.2f} states/s"
    )
    print(f"Report: {args.save_json}")


if __name__ == "__main__":
    main()
