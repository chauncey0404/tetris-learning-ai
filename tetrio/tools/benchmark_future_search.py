from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tetrio.network.checkpoint import load_expert_v1_1
from tetrio.rollout.batched import BatchedRolloutConfig, run_batched_v11
from tetrio.tools.watch_expert_v0 import parse_seed_spec


MODES = {
    "baseline": dict(
        adaptive_future_scheduling=False,
        future_search_cache=False,
        future_feature_memo=False,
        future_max_chunks_per_row=1,
    ),
    "cache_memo": dict(
        adaptive_future_scheduling=False,
        future_search_cache=True,
        future_feature_memo=True,
        future_max_chunks_per_row=1,
    ),
    "chunk2": dict(
        adaptive_future_scheduling=True,
        future_search_cache=True,
        future_feature_memo=True,
        future_max_chunks_per_row=2,
    ),
    "chunk4": dict(
        adaptive_future_scheduling=True,
        future_search_cache=True,
        future_feature_memo=True,
        future_max_chunks_per_row=4,
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Exact apples-to-apples Expert V1.1 future-search benchmark. "
            "All modes use the same checkpoint/seeds/horizon and must produce "
            "the exact same move trace."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_1_future_500k.pt"),
    )
    p.add_argument("--seeds", default="9091-9094")
    p.add_argument("--max-pieces", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--reference-audit-every", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--state-batch", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=512)
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument(
        "--modes",
        default="baseline,cache_memo,chunk2,chunk4",
        help="Comma-separated subset: baseline,cache_memo,chunk2,chunk4",
    )
    p.add_argument(
        "--save-json",
        type=Path,
        default=Path(r"artifacts\tetrio\future_search_performance_benchmark.json"),
    )
    return p.parse_args()


def trace_key(item: dict) -> tuple:
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


def assert_trace_equal(reference: dict, candidate: dict, name: str) -> None:
    ra = {int(x["seed"]): x for x in reference["results"]}
    rb = {int(x["seed"]): x for x in candidate["results"]}
    for seed in sorted(ra):
        a = ra[seed].get("trace", [])
        b = rb[seed].get("trace", [])
        if len(a) != len(b):
            raise RuntimeError(
                f"{name} trace length mismatch seed={seed}: {len(a)} != {len(b)}"
            )
        for move, (x, y) in enumerate(zip(a, b), start=1):
            if trace_key(x) != trace_key(y):
                raise RuntimeError(
                    f"{name} POLICY DIVERGENCE seed={seed} move={move}: "
                    f"reference={x} candidate={y}"
                )


def make_config(args, mode: str, pieces: int) -> BatchedRolloutConfig:
    return BatchedRolloutConfig(
        max_pieces=pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        workers=args.workers,
        state_batch=args.state_batch,
        progress_every=args.progress_every,
        top_overall=args.top_overall,
        top_per_branch=args.top_per_branch,
        collect_trace=True,
        **MODES[mode],
    )


def main() -> None:
    args = parse_args()
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    bad = [x for x in modes if x not in MODES]
    if bad:
        raise SystemExit(f"Unknown modes: {bad}")
    if not modes:
        raise SystemExit("No modes selected")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    seeds = parse_seed_spec(args.seeds, 9091)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    model, ckpt = load_expert_v1_1(args.checkpoint, device=device)

    print("=" * 112)
    print("TETR.IO V1.1 FUTURE SEARCH — EXACT PERFORMANCE BENCHMARK")
    print("=" * 112)
    print(f"Checkpoint : {args.checkpoint} epoch={ckpt.get('epoch')}")
    print(f"Seeds      : {seeds}")
    print(f"Pieces     : {args.max_pieces}/seed")
    print(f"Workers    : {args.workers}")
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(device)}")
    print(f"Modes      : {modes}")
    print()

    # Warm the neural/CUDA path before measuring any mode.
    print("Warm-up: 40 pieces")
    run_batched_v11(
        model,
        seeds=[int(seeds[0])],
        device=device,
        config=make_config(args, "baseline", 40),
    )
    print()

    runs = {}
    reference = None
    reference_name = None

    for mode in modes:
        print("=" * 112)
        print(f"MODE: {mode}")
        print("=" * 112)
        result = run_batched_v11(
            model,
            seeds=seeds,
            device=device,
            config=make_config(args, mode, int(args.max_pieces)),
        )

        if reference is None:
            reference = result
            reference_name = mode
        else:
            assert_trace_equal(reference, result, mode)
            print(
                f"TRACE PARITY PASS: {mode} == {reference_name} "
                f"for {len(seeds)} seed(s)"
            )

        rate = float(result["runtime"]["states_per_second"])
        runs[mode] = {
            "rate": rate,
            "runtime": result["runtime"],
            "aggregate": result["aggregate"],
            "results": result["results"],
            "trace_parity_pass": True,
            "trace_parity_to": reference_name,
        }
        print(f"MODE RESULT: {mode} = {rate:.2f} states/s")
        print()

    ref_rate = float(runs[modes[0]]["rate"])
    best_mode = max(modes, key=lambda x: float(runs[x]["rate"]))
    best_rate = float(runs[best_mode]["rate"])

    print("=" * 112)
    print("PERFORMANCE SUMMARY")
    print("=" * 112)
    for mode in modes:
        rate = float(runs[mode]["rate"])
        print(
            f"{mode:12s} {rate:8.2f} states/s "
            f"speedup_vs_{modes[0]}={rate/ref_rate:.3f}x"
        )
    print(f"Best mode    : {best_mode}")
    print(f"Best rate    : {best_rate:.2f} states/s")
    print(f"Best speedup : {best_rate/ref_rate:.3f}x vs {modes[0]}")

    report = {
        "checkpoint": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "seeds": seeds,
        "max_pieces": int(args.max_pieces),
        "workers": int(args.workers),
        "device": str(device),
        "reference_mode": reference_name,
        "modes": runs,
        "best_mode": best_mode,
        "best_rate": best_rate,
        "best_speedup_vs_reference": best_rate / ref_rate,
    }
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report       : {args.save_json}")


if __name__ == "__main__":
    main()
