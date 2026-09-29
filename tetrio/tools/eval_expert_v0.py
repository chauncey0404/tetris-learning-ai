from __future__ import annotations

import argparse
import json
import os
import pathlib
from pathlib import Path
import time

import torch

from tetrio.network.cache_prepare import (
    ExpertV0CacheBuildSpec,
    ensure_expert_v0_cache,
)
from tetrio.network.model import TetrioExpertV0Network
from tetrio.tools.train_expert_v0 import cache_rows, run_epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained TETR.IO Expert-v0 checkpoint on a held-out "
            "candidate cache without any optimizer updates."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v0_full.pt"),
    )
    p.add_argument(
        "--test-cache",
        type=Path,
        default=Path(r"data\tetrio\expert_v0\test_full_fast_s8192"),
    )
    p.add_argument(
        "--test-source",
        type=Path,
        default=Path(r"data\tetrio\expert\top_players_s1_test.parquet"),
    )
    p.add_argument(
        "--test-cache-rows",
        type=int,
        default=0,
        help="Rows to auto-build for test cache; 0 means all eligible rows.",
    )
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--pipeline",
        choices=("dense", "compact_gpu"),
        default="compact_gpu",
    )
    p.add_argument("--prefetch-shards", type=int, default=4)
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--auto-build-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--cache-backend",
        choices=("reference", "fast"),
        default="fast",
    )
    p.add_argument(
        "--cache-workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 2)),
    )
    p.add_argument("--cache-shard-size", type=int, default=8192)
    p.add_argument("--cache-seed", type=int, default=20260906)
    p.add_argument("--cache-fast-max-states", type=int, default=10_000)
    p.add_argument("--cache-reference-max-states", type=int, default=50_000)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v0_full_test.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    ensure_expert_v0_cache(
        ExpertV0CacheBuildSpec(
            cache_dir=args.test_cache,
            source=args.test_source,
            rows=args.test_cache_rows,
            backend=args.cache_backend,
            workers=args.cache_workers,
            shard_size=args.cache_shard_size,
            seed=args.cache_seed,
            fast_max_states=args.cache_fast_max_states,
        ),
        auto_build=args.auto_build_cache,
        label="held-out test cache",
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    # PyTorch 2.6+ defaults to the safer weights-only loader.  Expert-v0
    # checkpoints created on Windows contain pathlib.WindowsPath values in the
    # saved argparse config, so explicitly allowlist pathlib path classes.
    safe_path_globals = [
        pathlib.Path,
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    ]
    with torch.serialization.safe_globals(safe_path_globals):
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    if checkpoint.get("format") != "tetrio_expert_v0":
        raise SystemExit(
            f"Unexpected checkpoint format: {checkpoint.get('format')!r}"
        )

    model = TetrioExpertV0Network().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        amp_name = "bf16"
    elif device.type == "cuda":
        amp_dtype = torch.float16
        amp_name = "fp16"
    else:
        amp_dtype = torch.float32
        amp_name = "fp32"

    rows = cache_rows(args.test_cache)
    if rows <= 0:
        raise SystemExit("Held-out test cache is empty")

    print("=" * 96)
    print("TETR.IO EXPERT V0 — HELD-OUT TEST")
    print("=" * 96)
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Best epoch   : {checkpoint.get('epoch')}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"AMP          : {amp_name}")
    print(f"Test rows    : {rows:,}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Pipeline     : {args.pipeline}")
    print()

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        test = run_epoch(
            model=model,
            cache_dir=args.test_cache,
            device=device,
            batch_size=args.batch_size,
            optimizer=None,
            scaler=None,
            amp_dtype=amp_dtype,
            training=False,
            epoch_seed=0,
            hold_loss_weight=0.25,
            label_smoothing=0.0,
            grad_clip=1.0,
            pipeline=args.pipeline,
            prefetch_shards=args.prefetch_shards,
            pin_memory=args.pin_memory,
        )

    seconds = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    train_metrics = checkpoint.get("metrics") or {}
    best_val = (train_metrics.get("val") or {})
    val_top1 = best_val.get("top1")
    val_top3 = best_val.get("top3")
    val_mrr = best_val.get("mrr")
    val_hold = best_val.get("hold_acc")

    def gap(test_value, val_value):
        if val_value is None:
            return None
        return float(test_value) - float(val_value)

    report = {
        "format": "tetrio_expert_v0_heldout_test",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_cache": str(args.test_cache),
        "test_rows": rows,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": amp_name,
        "seconds": seconds,
        "peak_cuda_memory_bytes": peak_memory,
        "test": test,
        "checkpoint_validation": {
            "top1": val_top1,
            "top3": val_top3,
            "mrr": val_mrr,
            "hold_acc": val_hold,
        },
        "test_minus_validation": {
            "top1": gap(test["top1"], val_top1),
            "top3": gap(test["top3"], val_top3),
            "mrr": gap(test["mrr"], val_mrr),
            "hold_acc": gap(test["hold_acc"], val_hold),
        },
        "note": (
            "Held-out test is for final generalization measurement. "
            "Do not tune hyperparameters against this split; continue tuning on validation."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Top-1        : {test['top1']:.4f}")
    print(f"Top-3        : {test['top3']:.4f}")
    print(f"MRR          : {test['mrr']:.4f}")
    print(f"Hold acc     : {test['hold_acc']:.4f}")
    print(f"Throughput   : {test['rows_per_second']:.0f} rows/s")
    print(f"Elapsed      : {seconds:.1f}s")
    if val_top1 is not None:
        print()
        print("Test - checkpoint validation:")
        print(f"  Top-1      : {test['top1'] - float(val_top1):+.4f}")
        print(f"  Top-3      : {test['top3'] - float(val_top3):+.4f}")
        print(f"  MRR        : {test['mrr'] - float(val_mrr):+.4f}")
        print(f"  Hold acc   : {test['hold_acc'] - float(val_hold):+.4f}")
    if peak_memory is not None:
        print(f"Peak CUDA mem: {peak_memory / (1024**3):.2f} GiB")
    print(f"Report       : {args.output}")
    print("Status       : HELD-OUT GENERALIZATION RESULT")


if __name__ == "__main__":
    main()
