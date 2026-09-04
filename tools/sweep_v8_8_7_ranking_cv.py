from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tetris_ai.learning.ranking import (
    OfflineRankingCorpus,
    pairwise_logistic_ranking_loss,
    pairwise_ordering_accuracy_numpy,
)
from tetris_ai.model.q_network import ObservableSafeQNetwork
from tetris_ai.learning.cuda_graph import make_capturable_adamw

DEFAULT_CHECKPOINT = "models/v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
DEFAULT_CORPUS = "data/v8_8_7_ranking_corpus_4761_4775.npz"


@dataclass
class FoldRow:
    fold: int
    validation_seeds: str
    train_states: int
    validation_states: int
    weight: float
    updates: int
    baseline_train_accuracy: float
    final_train_accuracy: float
    delta_train_accuracy: float
    baseline_validation_accuracy: float
    final_validation_accuracy: float
    delta_validation_accuracy: float
    train_pairs: int
    validation_pairs: int
    last_raw_ranking_loss: float
    optimizer_resumed: bool


@dataclass
class SummaryRow:
    weight: float
    updates: int
    folds: int
    mean_baseline_validation_accuracy: float
    mean_final_validation_accuracy: float
    mean_delta_validation_accuracy: float
    median_delta_validation_accuracy: float
    worst_delta_validation_accuracy: float
    best_delta_validation_accuracy: float
    improved_folds: int
    tied_folds: int
    regressed_folds: int
    mean_delta_train_accuracy: float
    mean_train_validation_delta_gap: float
    conservative_pass: bool
    rank_score: float


def parse_float_list(text: str) -> list[float]:
    out, seen = [], set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0:
            raise ValueError("All weights must be > 0.")
        key = round(value, 12)
        if key not in seen:
            seen.add(key)
            out.append(value)
    if not out:
        raise ValueError("No weights parsed.")
    return out


def parse_int_list(text: str) -> list[int]:
    out, seen = [], set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("All update counts must be > 0.")
        if value not in seen:
            seen.add(value)
            out.append(value)
    if not out:
        raise ValueError("No update counts parsed.")
    return out


def cpu_clone_state_dict(state_dict: dict) -> dict:
    return {
        k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
        for k, v in state_dict.items()
    }


def make_seed_folds(unique_seeds: list[int], folds: int, sample_seed: int) -> list[list[int]]:
    if folds < 2:
        raise ValueError("--folds must be >= 2.")
    if len(unique_seeds) < folds:
        raise ValueError(f"Need at least {folds} unique seeds; corpus has {len(unique_seeds)}.")
    seeds = list(sorted(unique_seeds))
    random.Random(int(sample_seed)).shuffle(seeds)
    result = [[] for _ in range(folds)]
    for i, seed in enumerate(seeds):
        result[i % folds].append(int(seed))
    return result


def indices_for_seed_sets(seed_array: np.ndarray, validation_seeds: set[int], device: torch.device):
    val_mask = np.isin(seed_array, np.asarray(sorted(validation_seeds), dtype=np.int64))
    train_idx = torch.from_numpy(np.flatnonzero(~val_mask).astype(np.int64)).to(device)
    val_idx = torch.from_numpy(np.flatnonzero(val_mask).astype(np.int64)).to(device)
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise RuntimeError("Fold produced empty train/validation split.")
    return train_idx, val_idx


@torch.inference_mode()
def evaluate_indices(model, corpus: OfflineRankingCorpus, indices: torch.Tensor, batch_size: int = 512) -> dict:
    q_rows, target_rows, mask_rows = [], [], []
    for start in range(0, int(indices.numel()), int(batch_size)):
        idx = indices[start:start + int(batch_size)]
        q = model(
            state=corpus.state[idx],
            candidates=corpus.candidates[idx],
            rewards=corpus.rewards[idx],
            teacher_scores=corpus.teacher_scores[idx],
            teacher_ranks=corpus.teacher_ranks[idx],
        )
        q_rows.append(q.detach().cpu().numpy())
        target_rows.append(corpus.pair_targets[idx].detach().cpu().numpy())
        mask_rows.append(corpus.candidate_mask[idx].detach().cpu().numpy())
    q_np = np.concatenate(q_rows)
    t_np = np.concatenate(target_rows)
    m_np = np.concatenate(mask_rows)
    correct, total, accuracy = pairwise_ordering_accuracy_numpy(q_np, t_np, m_np)
    return {"states": int(indices.numel()), "correct": int(correct), "pairs": int(total), "accuracy": float(accuracy)}


def run_one_configuration(
    *, base_model_state, checkpoint_optimizer_state, corpus, train_indices, validation_indices,
    baseline_train, baseline_validation, fold, validation_seeds, weight, updates, batch_size,
    temperature, lr, weight_decay, grad_clip, sample_seed, device,
) -> FoldRow:
    model = ObservableSafeQNetwork().to(device)
    model.load_state_dict(base_model_state)
    model.train()

    optimizer, _, resumed = make_capturable_adamw(
        model=model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        checkpoint_optimizer_state=copy.deepcopy(checkpoint_optimizer_state),
        resume=True,
    )
    if not resumed:
        raise RuntimeError("CV requires resumable 31.2M optimizer state; refusing fallback.")

    # Same fold seed for every config: shorter runs are prefixes of longer runs.
    gen = torch.Generator(device=device)
    gen.manual_seed(int(sample_seed) + int(fold) * 100_003)
    last_loss = float("nan")

    for _ in range(int(updates)):
        positions = torch.randint(0, int(train_indices.numel()), (int(batch_size),), device=device, generator=gen)
        idx = train_indices[positions]
        q = model(
            state=corpus.state[idx],
            candidates=corpus.candidates[idx],
            rewards=corpus.rewards[idx],
            teacher_scores=corpus.teacher_scores[idx],
            teacher_ranks=corpus.teacher_ranks[idx],
        )
        raw_loss, _, valid_count = pairwise_logistic_ranking_loss(
            q,
            corpus.pair_targets[idx],
            candidate_mask=corpus.candidate_mask[idx],
            temperature=temperature,
        )
        if int(valid_count.detach().item()) <= 0:
            raise RuntimeError("Sampled ranking batch has zero valid pair labels.")
        optimizer.zero_grad(set_to_none=True)
        (raw_loss * float(weight)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        last_loss = float(raw_loss.detach().item())

    model.eval()
    final_train = evaluate_indices(model, corpus, train_indices)
    final_val = evaluate_indices(model, corpus, validation_indices)

    row = FoldRow(
        fold=int(fold), validation_seeds="|".join(map(str, sorted(validation_seeds))),
        train_states=int(train_indices.numel()), validation_states=int(validation_indices.numel()),
        weight=float(weight), updates=int(updates),
        baseline_train_accuracy=float(baseline_train["accuracy"]),
        final_train_accuracy=float(final_train["accuracy"]),
        delta_train_accuracy=float(final_train["accuracy"] - baseline_train["accuracy"]),
        baseline_validation_accuracy=float(baseline_validation["accuracy"]),
        final_validation_accuracy=float(final_val["accuracy"]),
        delta_validation_accuracy=float(final_val["accuracy"] - baseline_validation["accuracy"]),
        train_pairs=int(final_train["pairs"]), validation_pairs=int(final_val["pairs"]),
        last_raw_ranking_loss=float(last_loss), optimizer_resumed=bool(resumed),
    )
    del optimizer, model
    torch.cuda.empty_cache()
    return row


def summarize(rows: list[FoldRow]) -> list[SummaryRow]:
    groups = {}
    for row in rows:
        groups.setdefault((row.weight, row.updates), []).append(row)
    out = []
    for (weight, updates), items in groups.items():
        vd = np.asarray([x.delta_validation_accuracy for x in items], dtype=np.float64)
        td = np.asarray([x.delta_train_accuracy for x in items], dtype=np.float64)
        bv = np.asarray([x.baseline_validation_accuracy for x in items], dtype=np.float64)
        fv = np.asarray([x.final_validation_accuracy for x in items], dtype=np.float64)
        eps = 1e-12
        improved = int(np.count_nonzero(vd > eps))
        tied = int(np.count_nonzero(np.abs(vd) <= eps))
        regressed = int(len(items) - improved - tied)
        conservative_pass = bool(
            float(vd.mean()) > 0.0
            and improved >= math.ceil(len(items) * 0.60)
            and float(vd.min()) >= -0.10
        )
        gap = float(np.mean(td - vd))
        rank_score = float(vd.mean() + 0.35 * vd.min() - 0.10 * max(0.0, gap))
        out.append(SummaryRow(
            weight=float(weight), updates=int(updates), folds=len(items),
            mean_baseline_validation_accuracy=float(bv.mean()),
            mean_final_validation_accuracy=float(fv.mean()),
            mean_delta_validation_accuracy=float(vd.mean()),
            median_delta_validation_accuracy=float(np.median(vd)),
            worst_delta_validation_accuracy=float(vd.min()),
            best_delta_validation_accuracy=float(vd.max()),
            improved_folds=improved, tied_folds=tied, regressed_folds=regressed,
            mean_delta_train_accuracy=float(td.mean()),
            mean_train_validation_delta_gap=gap,
            conservative_pass=conservative_pass, rank_score=rank_score,
        ))
    out.sort(key=lambda r: (
        r.conservative_pass, r.rank_score, r.mean_delta_validation_accuracy,
        r.worst_delta_validation_accuracy, -r.updates, -r.weight,
    ), reverse=True)
    return out


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    data = [asdict(r) for r in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader(); writer.writerows(data)


def main() -> None:
    p = argparse.ArgumentParser(description="V8.8.7 seed-group CV sweep; no checkpoint saved.")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--weights", default="0.001,0.0025,0.005,0.01,0.02")
    p.add_argument("--updates", default="1,2,5,10,20,40")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sample-seed", type=int, default=20260901)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: this sweep resumes production capturable AdamW state.")
    weights = parse_float_list(args.weights)
    updates_grid = parse_int_list(args.updates)
    device = torch.device("cuda")

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    corpus_path = PROJECT_ROOT / args.corpus
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("env_steps", -1)) != 31_200_000:
        raise RuntimeError("CV base must be formal V8.8.6 31.2M Champion.")
    if "optimizer_state_dict" not in checkpoint:
        raise RuntimeError("31.2M checkpoint has no optimizer_state_dict.")

    base_model_state = cpu_clone_state_dict(checkpoint["model_state_dict"])
    optimizer_state = copy.deepcopy(checkpoint["optimizer_state_dict"])
    corpus = OfflineRankingCorpus(corpus_path, device=device)
    raw = np.load(corpus_path, allow_pickle=False)
    if "seed" not in raw:
        raise RuntimeError("Ranking corpus must contain per-state seed array.")
    seed_array = np.asarray(raw["seed"], dtype=np.int64)
    if seed_array.shape != (len(corpus),):
        raise RuntimeError(f"seed array shape mismatch: {seed_array.shape}")
    unique_seeds = sorted({int(x) for x in seed_array.tolist()})
    folds = make_seed_folds(unique_seeds, args.folds, args.sample_seed)

    print("=" * 108)
    print("V8.8.7 RANKING AUXILIARY — SEED-GROUP CROSS-VALIDATION SWEEP")
    print("=" * 108)
    print("Checkpoint :", checkpoint_path)
    print("Corpus     :", corpus_path)
    print("States     :", len(corpus))
    print("Seeds      :", unique_seeds)
    print("Folds      :", folds)
    print("Weights    :", weights)
    print("Updates    :", updates_grid)
    print("Optimizer  : production 31.2M AdamW state RESUMED per run")
    print("No checkpoint will be written.\n")

    started = time.perf_counter()
    baseline_model = ObservableSafeQNetwork().to(device)
    baseline_model.load_state_dict(base_model_state)
    baseline_model.eval()
    fold_cache = {}
    for fold_index, val_seed_list in enumerate(folds, 1):
        train_idx, val_idx = indices_for_seed_sets(seed_array, set(val_seed_list), device)
        bt = evaluate_indices(baseline_model, corpus, train_idx)
        bv = evaluate_indices(baseline_model, corpus, val_idx)
        fold_cache[fold_index] = (train_idx, val_idx, bt, bv)
        print(f"Fold {fold_index}: val seeds={sorted(val_seed_list)} | train={int(train_idx.numel())} val={int(val_idx.numel())} | baseline train={bt['accuracy']*100:.1f}% val={bv['accuracy']*100:.1f}%")
    del baseline_model
    torch.cuda.empty_cache()

    all_rows = []
    total_runs = len(folds) * len(weights) * len(updates_grid)
    run_index = 0
    for weight in weights:
        for updates in updates_grid:
            print(f"\nCONFIG weight={weight:g} updates={updates}")
            for fold_index, val_seed_list in enumerate(folds, 1):
                run_index += 1
                train_idx, val_idx, bt, bv = fold_cache[fold_index]
                t0 = time.perf_counter()
                row = run_one_configuration(
                    base_model_state=base_model_state,
                    checkpoint_optimizer_state=optimizer_state,
                    corpus=corpus, train_indices=train_idx, validation_indices=val_idx,
                    baseline_train=bt, baseline_validation=bv,
                    fold=fold_index, validation_seeds=val_seed_list,
                    weight=weight, updates=updates, batch_size=args.batch_size,
                    temperature=args.temperature, lr=args.lr,
                    weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                    sample_seed=args.sample_seed, device=device,
                )
                all_rows.append(row)
                print(f"  fold {fold_index}/{len(folds)} VAL {row.baseline_validation_accuracy*100:5.1f}% -> {row.final_validation_accuracy*100:5.1f}% ({row.delta_validation_accuracy*100:+5.1f}pp) | TRAIN Δ={row.delta_train_accuracy*100:+5.1f}pp | {time.perf_counter()-t0:.2f}s [{run_index}/{total_runs}]")

    summaries = summarize(all_rows)
    elapsed = time.perf_counter() - started
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "v8_8_7_ranking_cv" / f"sweep_{stamp}")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fold_results.csv", all_rows)
    write_csv(output_dir / "summary.csv", summaries)

    best = summaries[0]
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "V8.8.7 RANKING AUX CV SWEEP - NO MODEL SAVED",
        "checkpoint": str(checkpoint_path), "corpus": str(corpus_path),
        "unique_seeds": unique_seeds, "folds": folds,
        "weights": weights, "updates": updates_grid,
        "optimizer": "production capturable AdamW state resumed fresh for every fold/configuration",
        "elapsed_seconds": elapsed,
        "best": asdict(best), "summaries": [asdict(x) for x in summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "V8.8.7 RANKING AUXILIARY — SEED-GROUP CROSS-VALIDATION SWEEP",
        "=" * 108, "", "NO MODEL CHECKPOINT WAS SAVED.", "", "TOP CONFIGURATIONS", "-" * 108,
    ]
    for rank, row in enumerate(summaries[:10], 1):
        lines.append(
            f"{rank:>2}. weight={row.weight:g} updates={row.updates:<3} | mean VAL Δ={row.mean_delta_validation_accuracy*100:+6.2f}pp | worst={row.worst_delta_validation_accuracy*100:+6.2f}pp | folds +/=/− {row.improved_folds}/{row.tied_folds}/{row.regressed_folds} | train Δ={row.mean_delta_train_accuracy*100:+6.2f}pp | {'PASS' if row.conservative_pass else 'FAIL'}"
        )
    lines += [
        "", "BEST CONFIGURATION", "-" * 108,
        f"weight: {best.weight:g}", f"updates: {best.updates}",
        f"baseline held-out accuracy: {best.mean_baseline_validation_accuracy*100:.2f}%",
        f"final held-out accuracy: {best.mean_final_validation_accuracy*100:.2f}%",
        f"mean held-out delta: {best.mean_delta_validation_accuracy*100:+.2f}pp",
        f"worst fold delta: {best.worst_delta_validation_accuracy*100:+.2f}pp",
        f"improved/tied/regressed: {best.improved_folds}/{best.tied_folds}/{best.regressed_folds}",
        f"mean training delta: {best.mean_delta_train_accuracy*100:+.2f}pp",
        f"conservative preflight gate: {'PASS' if best.conservative_pass else 'FAIL'}",
        "", "If every configuration FAILS, do not run the production V8.8.7 pilot yet.",
    ]
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 108)
    print("V8.8.7 RANKING CV SWEEP COMPLETE")
    print("=" * 108)
    print("Elapsed:", f"{elapsed:.2f}s")
    print("Configurations:", len(summaries))
    print("Best:", f"weight={best.weight:g}", f"updates={best.updates}")
    print("Held-out:", f"{best.mean_baseline_validation_accuracy*100:.2f}% -> {best.mean_final_validation_accuracy*100:.2f}%", f"(Δ {best.mean_delta_validation_accuracy*100:+.2f}pp)")
    print("Worst fold:", f"{best.worst_delta_validation_accuracy*100:+.2f}pp")
    print("Improved/Tied/Regressed:", f"{best.improved_folds}/{best.tied_folds}/{best.regressed_folds}")
    print("Conservative preflight gate:", "PASS" if best.conservative_pass else "FAIL")
    print("Output:", output_dir)
    print("Files: fold_results.csv, summary.csv, summary.json, report.txt")
    print("=" * 108)


if __name__ == "__main__":
    main()
