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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.watch_models import (
        HeuristicTeacherV2,
        ModelSession,
        choose_device,
        load_policy,
    )
except ImportError as exc:
    raise RuntimeError(
        "This tool requires tools\\watch_models.py and should be placed at "
        "F:\\tetris-learning-ai\\tools\\compare_v8_8_6_31m_vs_41m_fresh.py"
    ) from exc

CHAMPION = r"models\v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
CHALLENGER = r"models\v8_8_6_control_continued_td_41200k.pt"
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


def main():
    p = argparse.ArgumentParser(
        description="Fresh development comparison: 31.2M Champion vs 41.2M same-recipe control."
    )
    p.add_argument("--champion", default=CHAMPION)
    p.add_argument("--challenger", default=CHALLENGER)
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

    champion_path = PROJECT_ROOT / args.champion
    challenger_path = PROJECT_ROOT / args.challenger
    if not champion_path.is_file():
        raise SystemExit(f"Missing champion: {champion_path}")
    if not challenger_path.is_file():
        raise SystemExit(f"Missing challenger: {challenger_path}")

    device = choose_device(args.device)
    teacher = HeuristicTeacherV2()
    champion_policy = load_policy(
        str(champion_path),
        label="31.2M Champion",
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )
    challenger_policy = load_policy(
        str(challenger_path),
        label="41.2M Control",
        device=device,
        gate_override=args.gate,
        semantics_override="normalized_q_margin",
    )

    print("=" * 108)
    print("31.2M CHAMPION vs 41.2M SAME-RECIPE CONTROL — FRESH DEVELOPMENT")
    print("=" * 108)
    print("Seeds:", args.seeds)
    print("Piece cap:", args.max_pieces)
    print("Device:", device)
    print("Gate:", f"{args.gate:.3f}")
    print()

    champ_rows, chall_rows, pairs = [], [], []
    started = time.perf_counter()

    for i, seed in enumerate(seeds, 1):
        c = run_one(champion_policy, teacher, device, seed, args.max_pieces, args.top_k, "31.2M Champion")
        h = run_one(challenger_policy, teacher, device, seed, args.max_pieces, args.top_k, "41.2M Control")
        champ_rows.append(c)
        chall_rows.append(h)

        pair = PairedRow(
            seed=seed,
            delta_pieces=h.pieces - c.pieces,
            delta_lines=h.lines - c.lines,
            delta_tetrises=h.tetrises - c.tetrises,
            delta_value=h.value - c.value,
            delta_reward_per_1000=h.reward_per_1000 - c.reward_per_1000,
            delta_avg_height=h.avg_height - c.avg_height,
            delta_max_height=h.max_height - c.max_height,
            delta_max_holes=h.max_holes - c.max_holes,
            delta_q_switch_rate=h.q_switch_rate - c.q_switch_rate,
            delta_gameover=h.gameover - c.gameover,
        )
        pairs.append(pair)

        print(
            f"[{i:>2}/{len(seeds)}] seed {seed} | "
            f"31.2M P{c.pieces} L{c.lines} T{c.tetrises} V{c.value} H{c.max_height} holes{c.max_holes} GO{c.gameover} || "
            f"41.2M P{h.pieces} L{h.lines} T{h.tetrises} V{h.value} H{h.max_height} holes{h.max_holes} GO{h.gameover} || "
            f"ΔV={pair.delta_value:+d} ΔT={pair.delta_tetrises:+d}"
        )

    elapsed = time.perf_counter() - started
    champ = aggregate(champ_rows)
    chall = aggregate(chall_rows)

    dv = [p.delta_value for p in pairs]
    dr = [p.delta_reward_per_1000 for p in pairs]
    ci_v = bootstrap_mean_ci(dv)
    ci_r = bootstrap_mean_ci(dr)

    wins = sum(x > 0 for x in dv)
    ties = sum(x == 0 for x in dv)
    losses = sum(x < 0 for x in dv)

    gates = {
        "gameovers_not_worse": chall["gameovers"] <= champ["gameovers"],
        "pieces_not_worse": chall["pieces_mean"] >= champ["pieces_mean"],
        "reward_per_1000_better": chall["reward_per_1000_mean"] > champ["reward_per_1000_mean"],
        "value_better": chall["value_mean"] > champ["value_mean"],
    }
    all_pass = all(gates.values())

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.output_dir
    if out is None:
        out = PROJECT_ROOT / "artifacts" / "control_31m_vs_41m" / f"fresh_{seeds[0]}_{seeds[-1]}_{stamp}"
    elif not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "games.csv", champ_rows + chall_rows)
    write_csv(out / "paired.csv", pairs)

    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seeds": seeds,
            "max_pieces": args.max_pieces,
            "elapsed_seconds": elapsed,
            "status": "DEVELOPMENT ONLY - NO PROMOTION",
        },
        "champion": champ,
        "challenger": chall,
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

    report = f"""31.2M CHAMPION vs 41.2M SAME-RECIPE CONTROL
====================================================================================================

DEVELOPMENT ONLY — NO PROMOTION.

Seeds: {args.seeds}
Piece cap: {args.max_pieces}

31.2M
  pieces mean : {champ['pieces_mean']:.2f}
  lines mean  : {champ['lines_mean']:.2f}
  Tetris mean : {champ['tetrises_mean']:.2f}
  value mean  : {champ['value_mean']:.2f}
  R/1000 mean : {champ['reward_per_1000_mean']:.2f}
  avgH mean   : {champ['avg_height_mean']:.3f}
  worst maxH  : {champ['max_height_worst']}
  worst holes : {champ['max_holes_worst']}
  GO          : {champ['gameovers']}
  Qswitch     : {champ['q_switch_rate_mean']*100:.2f}%

41.2M
  pieces mean : {chall['pieces_mean']:.2f}
  lines mean  : {chall['lines_mean']:.2f}
  Tetris mean : {chall['tetrises_mean']:.2f}
  value mean  : {chall['value_mean']:.2f}
  R/1000 mean : {chall['reward_per_1000_mean']:.2f}
  avgH mean   : {chall['avg_height_mean']:.3f}
  worst maxH  : {chall['max_height_worst']}
  worst holes : {chall['max_holes_worst']}
  GO          : {chall['gameovers']}
  Qswitch     : {chall['q_switch_rate_mean']*100:.2f}%

PAIRED 41.2M - 31.2M
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
    print(f"31.2M mean value     : {champ['value_mean']:.2f}")
    print(f"41.2M mean value     : {chall['value_mean']:.2f}")
    print(f"Paired mean Δvalue   : {mean(dv):+.2f}")
    print(f"Paired 95% CI        : [{ci_v[0]:+.2f}, {ci_v[1]:+.2f}]")
    print(f"Value W/T/L          : {wins}/{ties}/{losses}")
    print(f"31.2M mean Tetris    : {champ['tetrises_mean']:.2f}")
    print(f"41.2M mean Tetris    : {chall['tetrises_mean']:.2f}")
    print(f"31.2M gameovers      : {champ['gameovers']}")
    print(f"41.2M gameovers      : {chall['gameovers']}")
    print(f"Development gates    : {'PASS' if all_pass else 'FAIL'}")
    print("Output               :", out)
    print()
    print("Seeds used by this run are now DEVELOPMENT-CONSUMED.")
    print("=" * 108)


if __name__ == "__main__":
    main()
