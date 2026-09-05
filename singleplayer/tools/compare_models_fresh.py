from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from singleplayer.tools.watch_models import (
        HeuristicTeacherV2,
        ModelSession,
        choose_device,
        load_policy,
    )
except ImportError as exc:
    raise RuntimeError(
        "This tool requires tools\\watch_models.py and should be placed at "
        "F:\\tetris-learning-ai\\tools\\compare_models_fresh.py"
    ) from exc

DEFAULT_BASELINE = r"models\v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
DEFAULT_CHALLENGER = r"models\v8_8_6_control_continued_td_41200k.pt"
PROTECTED_FINAL_SEEDS = set(range(6, 21))


@dataclass
class GameRow:
    model: str
    seed: int
    pieces: int
    lines: int
    tetrises: int
    value: int
    reward_per_1000: float
    avg_height: float
    final_height: int
    max_height: int
    final_holes: int
    max_holes: int
    q_switches: int
    q_switch_rate: float
    gameover: int
    done_reason: str
    elapsed_seconds: float


@dataclass
class PairedRow:
    seed: int
    delta_pieces: int
    delta_lines: int
    delta_tetrises: int
    delta_value: int
    delta_reward_per_1000: float
    delta_avg_height: float
    delta_max_height: int
    delta_max_holes: int
    delta_q_switch_rate: float
    delta_gameover: int


def parse_seed_spec(text: str) -> list[int]:
    result = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            a_s, b_s = item.split("-", 1)
            a, b = int(a_s), int(b_s)
            if b < a:
                a, b = b, a
            result.extend(range(a, b + 1))
        else:
            result.append(int(item))
    if not result:
        raise ValueError("No seeds parsed.")
    return list(dict.fromkeys(result))


def mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def bootstrap_mean_ci(values, resamples=20000, rng_seed=20260829):
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, x.size, size=(resamples, x.size))
    boot = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def run_one(policy, teacher, device, seed, max_pieces, top_k, label):
    session = ModelSession(
        policy,
        seed=seed,
        max_pieces=max_pieces,
        top_k=top_k,
        device=device,
        teacher=teacher,
    )
    started = time.perf_counter()
    try:
        while not session.done:
            if session.stats.pieces >= max_pieces:
                break
            session.step()
            session.visual_drop = None

        elapsed = time.perf_counter() - started
        pieces = int(session.stats.pieces)
        value = int(session.stats.value)
        return GameRow(
            model=label,
            seed=int(seed),
            pieces=pieces,
            lines=int(session.stats.lines),
            tetrises=int(session.stats.tetrises),
            value=value,
            reward_per_1000=(0.0 if pieces == 0 else value * 1000.0 / pieces),
            avg_height=(0.0 if pieces == 0 else float(session.stats.height_sum) / pieces),
            final_height=int(session.stats.current_height),
            max_height=int(session.stats.max_height),
            final_holes=int(session.stats.holes),
            max_holes=int(session.stats.max_holes),
            q_switches=int(session.stats.interventions),
            q_switch_rate=(0.0 if pieces == 0 else float(session.stats.interventions) / pieces),
            gameover=int(bool(session.game_over)),
            done_reason=str(session.done_reason),
            elapsed_seconds=float(elapsed),
        )
    finally:
        try:
            session.close()
        except Exception:
            pass


def aggregate(rows):
    return {
        "games": len(rows),
        "pieces_mean": mean(r.pieces for r in rows),
        "lines_mean": mean(r.lines for r in rows),
        "tetrises_mean": mean(r.tetrises for r in rows),
        "value_mean": mean(r.value for r in rows),
        "reward_per_1000_mean": mean(r.reward_per_1000 for r in rows),
        "avg_height_mean": mean(r.avg_height for r in rows),
        "max_height_mean": mean(r.max_height for r in rows),
        "max_height_worst": max((r.max_height for r in rows), default=0),
        "max_holes_mean": mean(r.max_holes for r in rows),
        "max_holes_worst": max((r.max_holes for r in rows), default=0),
        "q_switch_rate_mean": mean(r.q_switch_rate for r in rows),
        "gameovers": sum(r.gameover for r in rows),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    data = [asdict(r) for r in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def safe_slug(text: str) -> str:
    chars = []
    for ch in text.strip().lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "model"


def main():
    p = argparse.ArgumentParser(
        description="Fresh paired whole-game comparison between two Tetris AI checkpoints."
    )
    p.add_argument(
        "--baseline",
        "--champion",
        dest="baseline",
        default=DEFAULT_BASELINE,
        help="Baseline checkpoint. --champion is kept as a compatibility alias.",
    )
    p.add_argument("--challenger", default=DEFAULT_CHALLENGER)
    p.add_argument("--baseline-label", default="Baseline")
    p.add_argument("--challenger-label", default="Challenger")
    p.add_argument("--seeds", default="4741-4760")
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--gate", type=float, default=0.600)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--allow-protected-seeds", action="store_true")
    args = p.parse_args()

    seeds = parse_seed_spec(args.seeds)
    protected = [s for s in seeds if s in PROTECTED_FINAL_SEEDS]
    if protected and not args.allow_protected_seeds:
        raise SystemExit(f"Protected final-report seeds blocked: {protected}")

    baseline_path = PROJECT_ROOT / args.baseline
    challenger_path = PROJECT_ROOT / args.challenger
    if not baseline_path.is_file():
        raise SystemExit(f"Missing baseline: {baseline_path}")
    if not challenger_path.is_file():
        raise SystemExit(f"Missing challenger: {challenger_path}")

    device = choose_device(args.device)
    teacher = HeuristicTeacherV2()
    baseline_policy = load_policy(
        str(baseline_path),
        label=args.baseline_label,
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )
    challenger_policy = load_policy(
        str(challenger_path),
        label=args.challenger_label,
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )

    print("=" * 108)
    print(f"{args.baseline_label} vs {args.challenger_label} — FRESH DEVELOPMENT")
    print("=" * 108)
    print("Baseline:", args.baseline)
    print("Challenger:", args.challenger)
    print("Seeds:", args.seeds)
    print("Piece cap:", args.max_pieces)
    print("Device:", device)
    print("Gate:", f"{args.gate:.3f}")
    print()

    baseline_rows, challenger_rows, pairs = [], [], []
    started = time.perf_counter()

    for i, seed in enumerate(seeds, 1):
        b = run_one(baseline_policy, teacher, device, seed, args.max_pieces, args.top_k, args.baseline_label)
        c = run_one(challenger_policy, teacher, device, seed, args.max_pieces, args.top_k, args.challenger_label)
        baseline_rows.append(b)
        challenger_rows.append(c)

        pair = PairedRow(
            seed=seed,
            delta_pieces=c.pieces - b.pieces,
            delta_lines=c.lines - b.lines,
            delta_tetrises=c.tetrises - b.tetrises,
            delta_value=c.value - b.value,
            delta_reward_per_1000=c.reward_per_1000 - b.reward_per_1000,
            delta_avg_height=c.avg_height - b.avg_height,
            delta_max_height=c.max_height - b.max_height,
            delta_max_holes=c.max_holes - b.max_holes,
            delta_q_switch_rate=c.q_switch_rate - b.q_switch_rate,
            delta_gameover=c.gameover - b.gameover,
        )
        pairs.append(pair)

        print(
            f"[{i:>2}/{len(seeds)}] seed {seed} | "
            f"{args.baseline_label} P{b.pieces} L{b.lines} T{b.tetrises} V{b.value} "
            f"H{b.max_height} holes{b.max_holes} GO{b.gameover} || "
            f"{args.challenger_label} P{c.pieces} L{c.lines} T{c.tetrises} V{c.value} "
            f"H{c.max_height} holes{c.max_holes} GO{c.gameover} || "
            f"ΔV={pair.delta_value:+d} ΔT={pair.delta_tetrises:+d}"
        )

    elapsed = time.perf_counter() - started
    baseline = aggregate(baseline_rows)
    challenger = aggregate(challenger_rows)

    dv = [p.delta_value for p in pairs]
    dr = [p.delta_reward_per_1000 for p in pairs]
    ci_v = bootstrap_mean_ci(dv)
    ci_r = bootstrap_mean_ci(dr)

    wins = sum(x > 0 for x in dv)
    ties = sum(x == 0 for x in dv)
    losses = sum(x < 0 for x in dv)

    gates = {
        "gameovers_not_worse": challenger["gameovers"] <= baseline["gameovers"],
        "pieces_not_worse": challenger["pieces_mean"] >= baseline["pieces_mean"],
        "reward_per_1000_better": challenger["reward_per_1000_mean"] > baseline["reward_per_1000_mean"],
        "value_better": challenger["value_mean"] > baseline["value_mean"],
    }
    all_pass = all(gates.values())

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.output_dir
    if out is None:
        lhs = safe_slug(args.baseline_label)
        rhs = safe_slug(args.challenger_label)
        out = PROJECT_ROOT / "artifacts" / "model_comparisons" / f"{lhs}_vs_{rhs}" / f"fresh_{seeds[0]}_{seeds[-1]}_{stamp}"
    elif not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "games.csv", baseline_rows + challenger_rows)
    write_csv(out / "paired.csv", pairs)

    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": seeds,
            "max_pieces": args.max_pieces,
            "top_k": args.top_k,
            "gate": args.gate,
            "device": str(device),
            "elapsed_seconds": elapsed,
            "status": "DEVELOPMENT ONLY - NO PROMOTION",
            "baseline_label": args.baseline_label,
            "challenger_label": args.challenger_label,
            "baseline_checkpoint": str(args.baseline),
            "challenger_checkpoint": str(args.challenger),
        },
        "baseline": baseline,
        "challenger": challenger,
        "paired": {
            "mean_delta_value": mean(dv),
            "delta_value_bootstrap_95ci": list(ci_v),
            "mean_delta_reward_per_1000": mean(dr),
            "delta_reward_per_1000_bootstrap_95ci": list(ci_r),
            "mean_delta_tetrises": mean(p.delta_tetrises for p in pairs),
            "mean_delta_lines": mean(p.delta_lines for p in pairs),
            "mean_delta_avg_height": mean(p.delta_avg_height for p in pairs),
            "mean_delta_max_height": mean(p.delta_max_height for p in pairs),
            "mean_delta_max_holes": mean(p.delta_max_holes for p in pairs),
            "value_wtl": [wins, ties, losses],
        },
        "development_gates": gates,
        "all_development_gates_pass": all_pass,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""{args.baseline_label} vs {args.challenger_label}
====================================================================================================

DEVELOPMENT ONLY — NO PROMOTION.

Baseline checkpoint : {args.baseline}
Challenger checkpoint: {args.challenger}
Seeds               : {args.seeds}
Piece cap           : {args.max_pieces}
Gate                : {args.gate:.3f}

{args.baseline_label}
  pieces mean : {baseline['pieces_mean']:.2f}
  lines mean  : {baseline['lines_mean']:.2f}
  Tetris mean : {baseline['tetrises_mean']:.2f}
  value mean  : {baseline['value_mean']:.2f}
  R/1000 mean : {baseline['reward_per_1000_mean']:.2f}
  avgH mean   : {baseline['avg_height_mean']:.3f}
  worst maxH  : {baseline['max_height_worst']}
  worst holes : {baseline['max_holes_worst']}
  GO          : {baseline['gameovers']}
  Qswitch     : {baseline['q_switch_rate_mean']*100:.2f}%

{args.challenger_label}
  pieces mean : {challenger['pieces_mean']:.2f}
  lines mean  : {challenger['lines_mean']:.2f}
  Tetris mean : {challenger['tetrises_mean']:.2f}
  value mean  : {challenger['value_mean']:.2f}
  R/1000 mean : {challenger['reward_per_1000_mean']:.2f}
  avgH mean   : {challenger['avg_height_mean']:.3f}
  worst maxH  : {challenger['max_height_worst']}
  worst holes : {challenger['max_holes_worst']}
  GO          : {challenger['gameovers']}
  Qswitch     : {challenger['q_switch_rate_mean']*100:.2f}%

PAIRED {args.challenger_label} - {args.baseline_label}
  mean Δvalue : {mean(dv):+.2f}
  95% CI      : [{ci_v[0]:+.2f}, {ci_v[1]:+.2f}]
  value W/T/L : {wins}/{ties}/{losses}
  mean ΔR/1000: {mean(dr):+.2f}
  95% CI      : [{ci_r[0]:+.2f}, {ci_r[1]:+.2f}]
  mean ΔTetris: {mean(p.delta_tetrises for p in pairs):+.2f}
  mean Δlines : {mean(p.delta_lines for p in pairs):+.2f}
  mean ΔavgH  : {mean(p.delta_avg_height for p in pairs):+.3f}
  mean ΔmaxH  : {mean(p.delta_max_height for p in pairs):+.2f}
  mean Δholes : {mean(p.delta_max_holes for p in pairs):+.2f}

DEVELOPMENT GATES
  gameovers_not_worse      : {'PASS' if gates['gameovers_not_worse'] else 'FAIL'}
  pieces_not_worse         : {'PASS' if gates['pieces_not_worse'] else 'FAIL'}
  reward_per_1000_better   : {'PASS' if gates['reward_per_1000_better'] else 'FAIL'}
  value_better             : {'PASS' if gates['value_better'] else 'FAIL'}
  overall                  : {'PASS' if all_pass else 'FAIL'}
"""
    (out / "report.txt").write_text(report, encoding="utf-8")

    print()
    print("=" * 108)
    print("FRESH DEVELOPMENT COMPARISON COMPLETE")
    print("=" * 108)
    print(f"{args.baseline_label} mean value   : {baseline['value_mean']:.2f}")
    print(f"{args.challenger_label} mean value : {challenger['value_mean']:.2f}")
    print(f"Paired mean Δvalue  : {mean(dv):+.2f}")
    print(f"Paired 95% CI       : [{ci_v[0]:+.2f}, {ci_v[1]:+.2f}]")
    print(f"Value W/T/L         : {wins}/{ties}/{losses}")
    print(f"{args.baseline_label} mean Tetris  : {baseline['tetrises_mean']:.2f}")
    print(f"{args.challenger_label} mean Tetris: {challenger['tetrises_mean']:.2f}")
    print(f"{args.baseline_label} gameovers    : {baseline['gameovers']}")
    print(f"{args.challenger_label} gameovers  : {challenger['gameovers']}")
    print(f"Development gates   : {'PASS' if all_pass else 'FAIL'}")
    print("Output              :", out)
    print()
    print("Seeds used by this run are now DEVELOPMENT-CONSUMED.")
    print("=" * 108)


if __name__ == "__main__":
    main()
