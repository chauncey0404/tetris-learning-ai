from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_ai.learning.ranking import (
    OfflineRankingCorpus,
    RankingAuxTrainer,
    evaluate_ranking_corpus,
)
from tetris_ai.model.q_network import ObservableSafeQNetwork


DEFAULT_CHECKPOINT = (
    "models/v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
)
DEFAULT_CORPUS = "data/v8_8_7_ranking_corpus_4761_4775.npz"


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "V8.8.7 offline ranking-aux preflight. "
            "No checkpoint is saved and no formal model is modified."
        )
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--output",
        default="artifacts/v8_8_7_ranking_preflight.json",
    )
    args = p.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1.")
    if args.weight <= 0.0:
        raise SystemExit("--weight must be > 0.")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be > 0.")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    corpus_path = PROJECT_ROOT / args.corpus

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("env_steps", -1)) != 31_200_000:
        raise RuntimeError(
            "Preflight base must be the formal 31.2M Champion."
        )

    base_model = ObservableSafeQNetwork().to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    corpus = OfflineRankingCorpus(corpus_path, device=device)

    before_train = evaluate_ranking_corpus(
        base_model,
        corpus,
        split="train",
    )
    before_val = evaluate_ranking_corpus(
        base_model,
        corpus,
        split="val",
    )

    # Work only on an in-memory copy. This is a plumbing/generalization
    # preflight, not a training checkpoint.
    model = copy.deepcopy(base_model)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4,
    )
    trainer = RankingAuxTrainer(
        model=model,
        optimizer=optimizer,
        corpus=corpus,
        split="train",
        batch_size=args.batch_size,
        weight=args.weight,
        temperature=args.temperature,
        seed=20260831,
    )

    metric_samples = []
    for step in range(1, args.steps + 1):
        metrics = trainer.step(
            collect_metrics=(
                step == 1
                or step == args.steps
                or step % max(1, args.steps // 5) == 0
            )
        )
        if metrics is not None:
            metric_samples.append(
                {
                    "step": step,
                    "loss": metrics.loss,
                    "pair_accuracy": metrics.pair_accuracy,
                    "valid_pairs": metrics.valid_pairs,
                    "q_span": metrics.q_span,
                }
            )
            print(
                f"ranking step {step:>4}/{args.steps} "
                f"loss={metrics.loss:.6f} "
                f"acc={metrics.pair_accuracy*100:5.1f}% "
                f"pairs={metrics.valid_pairs}"
            )

    model.eval()
    after_train = evaluate_ranking_corpus(
        model,
        corpus,
        split="train",
    )
    after_val = evaluate_ranking_corpus(
        model,
        corpus,
        split="val",
    )

    payload = {
        "status": "V8.8.7 RANKING AUX PREFLIGHT ONLY",
        "checkpoint": str(checkpoint_path),
        "corpus": str(corpus_path),
        "device": str(device),
        "steps": args.steps,
        "weight": args.weight,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "before": {
            "train": before_train,
            "val": before_val,
        },
        "after": {
            "train": after_train,
            "val": after_val,
        },
        "metric_samples": metric_samples,
        "important": (
            "This script never saves model weights. "
            "Validation results are diagnostic only."
        ),
    }

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("V8.8.7 RANKING AUX PREFLIGHT COMPLETE")
    print("=" * 96)
    print(
        "TRAIN pairwise:",
        f"{before_train['pairwise_accuracy']*100:.1f}% -> "
        f"{after_train['pairwise_accuracy']*100:.1f}%",
    )
    print(
        "VAL pairwise  :",
        f"{before_val['pairwise_accuracy']*100:.1f}% -> "
        f"{after_val['pairwise_accuracy']*100:.1f}%",
    )
    print("Ranking updates:", trainer.updates)
    print("No model checkpoint was written.")
    print("Report:", output)
    print("=" * 96)


if __name__ == "__main__":
    main()
