import argparse
import atexit
import json
import math
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Evaluation is CPU-heavy. Use independent single-thread workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from tetris_core import GymTetrisAdapter
from teacher import HeuristicTeacherV2
from gym_executor import execute_placement
from ai.state_encoder import encode_state
from ai.observable_q_network import ObservableSafeQNetwork
from v8_successor import preview_top_k_successors
from v8_4_observable import compact_candidate_arrays, conservative_choice
from v8_7_scale_invariant_policy import normalized_margin_choice


TOP_K = 4
LINE_VALUE = {0: 0, 1: 900, 2: 2000, 3: 3300, 4: 6000}
PERMANENT_BENCHMARK_FIRST = 6
PERMANENT_BENCHMARK_LAST = 20

_W_MODEL_50K = None
_W_MODEL_30K = None
_W_TEACHER = None
_W_ADAPTER = None


def _configure_worker_threads():
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _close_worker():
    global _W_ADAPTER
    if _W_ADAPTER is not None:
        try:
            _W_ADAPTER.close()
        except Exception:
            pass
        _W_ADAPTER = None


def _load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model = ObservableSafeQNetwork().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _worker_init(checkpoint_30k, checkpoint_50k):
    global _W_MODEL_30K, _W_MODEL_50K, _W_TEACHER, _W_ADAPTER

    _configure_worker_threads()

    _W_MODEL_30K = _load_model(checkpoint_30k)
    _W_MODEL_50K = _load_model(checkpoint_50k)

    _W_TEACHER = HeuristicTeacherV2()
    _W_ADAPTER = GymTetrisAdapter()

    atexit.register(_close_worker)


def _height_from_features(features):
    board = np.asarray(features[:200], dtype=np.float32).reshape(20, 10)
    rows = np.where(np.any(board != 0.0, axis=1))[0]
    return 0 if rows.size == 0 else int(20 - rows[0])


@torch.inference_mode()
def _q_values(model, state_features, successors):
    candidates, rewards, scores, ranks = compact_candidate_arrays(successors)

    q = model(
        state=torch.from_numpy(
            np.asarray(state_features, dtype=np.float32)
        ).unsqueeze(0),
        candidates=torch.from_numpy(candidates).unsqueeze(0),
        rewards=torch.from_numpy(rewards).unsqueeze(0),
        teacher_scores=torch.from_numpy(scores).unsqueeze(0),
        teacher_ranks=torch.from_numpy(ranks).unsqueeze(0),
    )[0]

    return q.detach().numpy()


def _run_game(task):
    seed = int(task["seed"])
    policy = str(task["policy"])
    gate = task["gate"]
    max_pieces = int(task["max_pieces"])

    adapter = _W_ADAPTER
    teacher = _W_TEACHER

    if policy == "teacher":
        model = None
    elif policy == "v8_5_30k_raw":
        model = _W_MODEL_30K
    elif policy == "v8_6_50k_norm":
        model = _W_MODEL_50K
    else:
        raise RuntimeError(f"Unknown policy: {policy}")

    state = adapter.reset(seed=seed)
    adapter.raw.gravity_enabled = False
    state_features = encode_state(state).astype(np.float32, copy=True)

    pieces = 0
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    interventions = 0
    max_height = _height_from_features(state_features)
    game_over = False

    while pieces < max_pieces:
        successors = preview_top_k_successors(
            adapter=adapter,
            teacher=teacher,
            state=state,
            top_k=TOP_K,
        )

        if not successors:
            game_over = True
            break

        if policy == "teacher":
            chosen_index = 0
        else:
            q = _q_values(
                model,
                state_features,
                successors,
            )

            if policy == "v8_5_30k_raw":
                chosen_index, _ = conservative_choice(
                    q,
                    float(gate),
                )
            elif policy == "v8_6_50k_norm":
                chosen_index, _ = normalized_margin_choice(
                    q,
                    float(gate),
                )
            else:
                raise RuntimeError(f"Unsupported Q policy: {policy}")

            if chosen_index != 0:
                interventions += 1

        chosen = successors[chosen_index]
        result = execute_placement(adapter, chosen.action)

        state = result["state"]
        pieces += 1

        # Observable-safe runtime identity was already validated separately.
        state_features = np.asarray(
            chosen.next_state_features,
            dtype=np.float32,
        ).copy()

        lines = int(result["info"].get("lines_cleared", 0))
        if lines in counts:
            counts[lines] += 1

        max_height = max(
            max_height,
            _height_from_features(state_features),
        )

        if bool(result["terminated"] or result["truncated"]):
            game_over = True
            break

    total_lines = sum(k * v for k, v in counts.items())
    value = sum(LINE_VALUE[k] * v for k, v in counts.items())

    return {
        "seed": seed,
        "policy": policy,
        "gate": None if gate is None else float(gate),
        "pieces": int(pieces),
        "lines": int(total_lines),
        "singles": int(counts[1]),
        "doubles": int(counts[2]),
        "triples": int(counts[3]),
        "tetrises": int(counts[4]),
        "value": int(value),
        "interventions": int(interventions),
        "intervention_rate": interventions / max(pieces, 1) * 100.0,
        "max_height": int(max_height),
        "game_over": bool(game_over),
    }


def result_key(seed, policy):
    return f"{int(seed)}|{policy}"


def atomic_save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def load_or_create_cache(
    path,
    checkpoint_30k,
    checkpoint_50k,
    gate_30k,
    gate_50k,
    seed_start,
    games,
    max_pieces,
):
    p = Path(path)

    expected = {
        "checkpoint_30k": checkpoint_30k,
        "checkpoint_50k": checkpoint_50k,
        "gate_30k_raw": float(gate_30k),
        "gate_50k_normalized": float(gate_50k),
        "seed_start": int(seed_start),
        "games": int(games),
        "max_pieces": int(max_pieces),
    }

    if not p.exists():
        return {
            "meta": expected,
            "results": {},
        }

    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("results", {})

    meta = data.get("meta", {})

    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Cache metadata mismatch for {key}: "
                f"cache={meta.get(key)!r}, requested={value!r}"
            )

    return data


def protect_seeds(seeds):
    bad = [
        seed
        for seed in seeds
        if PERMANENT_BENCHMARK_FIRST <= seed <= PERMANENT_BENCHMARK_LAST
    ]

    if bad:
        raise RuntimeError(
            f"Permanent benchmark seeds 6~20 are protected: {bad}"
        )


def auto_workers():
    logical = os.cpu_count() or 4
    if logical >= 20:
        return 16
    return max(1, logical - 2)


def summarize(rows):
    total_pieces = sum(row["pieces"] for row in rows)
    total_value = sum(row["value"] for row in rows)
    total_switch = sum(row["interventions"] for row in rows)

    return {
        "pieces": float(np.mean([row["pieces"] for row in rows])),
        "lines": float(np.mean([row["lines"] for row in rows])),
        "tetrises": float(np.mean([row["tetrises"] for row in rows])),
        "value": float(np.mean([row["value"] for row in rows])),
        "reward_per_1000": total_value / max(total_pieces, 1) * 1000.0,
        "avg_height": float(np.mean([row["max_height"] for row in rows])),
        "worst_height": int(max(row["max_height"] for row in rows)),
        "gameovers": int(sum(bool(row["game_over"]) for row in rows)),
        "switch_rate": total_switch / max(total_pieces, 1) * 100.0,
        "total_pieces": int(total_pieces),
        "total_value": int(total_value),
    }


def paired_stats(rows_a, rows_b):
    a = {row["seed"]: row for row in rows_a}
    b = {row["seed"]: row for row in rows_b}

    seeds = sorted(set(a) & set(b))

    diffs = np.asarray(
        [
            float(a[seed]["value"] - b[seed]["value"])
            for seed in seeds
        ],
        dtype=np.float64,
    )

    mean = float(np.mean(diffs))
    wins = int(np.sum(diffs > 0))
    ties = int(np.sum(diffs == 0))
    losses = int(np.sum(diffs < 0))

    if len(diffs) > 1:
        se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se
    else:
        lo = hi = mean

    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def print_summary(label, summary):
    print(
        f"{label:<14} "
        f"pieces={summary['pieces']:7.2f} "
        f"lines={summary['lines']:7.2f} "
        f"Tetris={summary['tetrises']:6.2f} "
        f"value={summary['value']:9.2f} "
        f"R/1000={summary['reward_per_1000']:9.2f} "
        f"avgH={summary['avg_height']:5.2f} "
        f"worstH={summary['worst_height']:>2} "
        f"GO={summary['gameovers']} "
        f"switch={summary['switch_rate']:5.2f}%"
    )


def print_relative(label, candidate, reference):
    reward_delta = (
        candidate["reward_per_1000"]
        / reference["reward_per_1000"]
        - 1.0
    ) * 100.0

    value_delta = (
        candidate["value"]
        / reference["value"]
        - 1.0
    ) * 100.0

    lines_delta = (
        candidate["lines"]
        / reference["lines"]
        - 1.0
    ) * 100.0

    tetris_delta = (
        candidate["tetrises"]
        / max(reference["tetrises"], 1e-9)
        - 1.0
    ) * 100.0

    print(
        f"{label:<28} "
        f"R/1000={reward_delta:+7.2f}% "
        f"value={value_delta:+7.2f}% "
        f"lines={lines_delta:+7.2f}% "
        f"Tetris={tetris_delta:+8.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Untouched 3001~3020 Champion qualification after V8.7 gate migration: "
            "Teacher vs V8.5-30K raw@0.060 vs V8.6-50K normalized@0.600."
        )
    )

    parser.add_argument(
        "--checkpoint-30k",
        default="models/v8_5_risk_aware_observable_safe_td_30k.pt",
    )

    parser.add_argument(
        "--checkpoint-50k",
        default="models/v8_6_risk_aware_observable_safe_td_50k.pt",
    )

    parser.add_argument(
        "--gate-30k",
        type=float,
        default=0.060,
    )

    parser.add_argument(
        "--gate-50k",
        type=float,
        default=0.600,
    )

    parser.add_argument("--seed-start", type=int, default=3001)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-pieces", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--cache",
        default="data/v8_7_50k_champion_qualification_3001_3020.json",
    )

    parser.add_argument(
        "--quiet-games",
        action="store_true",
    )

    args = parser.parse_args()

    if args.gate_30k < 0.0 or args.gate_50k < 0.0:
        raise ValueError("Gates must be >= 0.")

    if args.games <= 0:
        raise ValueError("--games must be > 0")

    if args.max_pieces <= 0:
        raise ValueError("--max-pieces must be > 0")

    seeds = list(range(args.seed_start, args.seed_start + args.games))
    protect_seeds(seeds)

    workers = args.workers if args.workers > 0 else auto_workers()

    cache = load_or_create_cache(
        args.cache,
        args.checkpoint_30k,
        args.checkpoint_50k,
        args.gate_30k,
        args.gate_50k,
        args.seed_start,
        args.games,
        args.max_pieces,
    )

    policies = [
        ("teacher", None),
        ("v8_5_30k_raw", float(args.gate_30k)),
        ("v8_6_50k_norm", float(args.gate_50k)),
    ]

    requested = [
        {
            "seed": seed,
            "policy": policy,
            "gate": gate,
            "max_pieces": args.max_pieces,
        }
        for policy, gate in policies
        for seed in seeds
    ]

    missing = [
        task
        for task in requested
        if result_key(task["seed"], task["policy"])
        not in cache["results"]
    ]

    print()
    print("=" * 80)
    print("V8.7 MIGRATED-GATE UNTOUCHED CHAMPION QUALIFICATION")
    print("=" * 80)
    print()
    print("Teacher baseline: ENABLED")
    print("Current champion:", args.checkpoint_30k)
    print("Champion raw gate:", args.gate_30k)
    print("V8.6 challenger:", args.checkpoint_50k)
    print("V8.7 frozen normalized gate:", args.gate_50k)
    print("Untouched seeds:", seeds)
    print("Logical CPUs:", os.cpu_count())
    print("Parallel workers:", workers)
    print("Requested games:", len(requested))
    print("Already cached:", len(requested) - len(missing))
    print("New games to simulate:", len(missing))
    print("Per-game resume:", "ENABLED")
    print("Two Q models loaded once / worker:", "ENABLED")
    print("Process-local CPU Q inference:", "ENABLED")
    print("Worker threads:", 1)
    print("Adapter reuse:", "ENABLED")
    print("Redundant post-step encode_state:", "REMOVED")
    print("Permanent seeds 6~20:", "PROTECTED")
    print("30K policy gate semantics:", "RAW Q GAP")
    print("50K policy gate semantics:", "NORMALIZED Q MARGIN")
    print()
    print(
        "IMPORTANT: 3001~3020 are qualification-only. "
        "Do not retune the normalized gate from this result."
    )

    start = time.perf_counter()

    if missing:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                args.checkpoint_30k,
                args.checkpoint_50k,
            ),
        ) as executor:

            futures = [
                executor.submit(_run_game, task)
                for task in missing
            ]

            done = 0

            for future in as_completed(futures):
                result = future.result()

                cache["results"][
                    result_key(
                        result["seed"],
                        result["policy"],
                    )
                ] = result

                atomic_save(args.cache, cache)

                done += 1

                if done == 1 or done % max(1, len(missing) // 20) == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"new={done:>3}/{len(missing)} "
                        f"elapsed={elapsed:7.1f}s "
                        f"games/s={done / max(elapsed, 1e-9):.3f}"
                    )

    elapsed = time.perf_counter() - start

    rows = {}

    for policy, _ in policies:
        policy_rows = [
            cache["results"][result_key(seed, policy)]
            for seed in seeds
        ]
        policy_rows.sort(key=lambda row: row["seed"])
        rows[policy] = policy_rows

    teacher_rows = rows["teacher"]
    champion_rows = rows["v8_5_30k_raw"]
    challenger_rows = rows["v8_6_50k_norm"]

    if not args.quiet_games:
        print()
        print("=" * 80)
        print("PER-SEED RESULTS")
        print("=" * 80)
        print()

        for seed in seeds:
            t = next(row for row in teacher_rows if row["seed"] == seed)
            c = next(row for row in champion_rows if row["seed"] == seed)
            n = next(row for row in challenger_rows if row["seed"] == seed)

            print(
                f"seed={seed} | "
                f"T value={t['value']:>7} GO={int(t['game_over'])} H={t['max_height']:>2} | "
                f"30K value={c['value']:>7} GO={int(c['game_over'])} "
                f"T={c['tetrises']:>3} sw={c['intervention_rate']:5.2f}% H={c['max_height']:>2} | "
                f"50K value={n['value']:>7} GO={int(n['game_over'])} "
                f"T={n['tetrises']:>3} sw={n['intervention_rate']:5.2f}% H={n['max_height']:>2}"
            )

    s_teacher = summarize(teacher_rows)
    s_champion = summarize(champion_rows)
    s_challenger = summarize(challenger_rows)

    pair_champion_teacher = paired_stats(
        champion_rows,
        teacher_rows,
    )

    pair_challenger_teacher = paired_stats(
        challenger_rows,
        teacher_rows,
    )

    pair_challenger_champion = paired_stats(
        challenger_rows,
        champion_rows,
    )

    print()
    print("=" * 80)
    print("FRESH QUALIFICATION SUMMARY")
    print("=" * 80)
    print()

    print_summary("Teacher", s_teacher)
    print_summary("V8.5-30K", s_champion)
    print_summary("V8.6-50K", s_challenger)

    print()
    print("=" * 80)
    print("RELATIVE PERFORMANCE")
    print("=" * 80)
    print()

    print_relative(
        "30K champion vs Teacher",
        s_champion,
        s_teacher,
    )

    print_relative(
        "50K challenger vs Teacher",
        s_challenger,
        s_teacher,
    )

    print_relative(
        "50K challenger vs 30K",
        s_challenger,
        s_champion,
    )

    print()
    print("=" * 80)
    print("PAIRED VALUE COMPARISONS")
    print("=" * 80)
    print()

    print(
        "30K - Teacher:",
        f"mean={pair_champion_teacher['mean']:+.1f}",
        f"95%CI=[{pair_champion_teacher['lo']:+.1f}, {pair_champion_teacher['hi']:+.1f}]",
        f"W/T/L={pair_champion_teacher['wins']}/{pair_champion_teacher['ties']}/{pair_champion_teacher['losses']}",
    )

    print(
        "30K - Teacher:",
        f"mean={pair_challenger_teacher['mean']:+.1f}",
        f"95%CI=[{pair_challenger_teacher['lo']:+.1f}, {pair_challenger_teacher['hi']:+.1f}]",
        f"W/T/L={pair_challenger_teacher['wins']}/{pair_challenger_teacher['ties']}/{pair_challenger_teacher['losses']}",
    )

    print(
        "50K - 30K:",
        f"mean={pair_challenger_champion['mean']:+.1f}",
        f"95%CI=[{pair_challenger_champion['lo']:+.1f}, {pair_challenger_champion['hi']:+.1f}]",
        f"W/T/L={pair_challenger_champion['wins']}/{pair_challenger_champion['ties']}/{pair_challenger_champion['losses']}",
    )

    simulated_pieces = (
        sum(row["pieces"] for row in teacher_rows)
        + sum(row["pieces"] for row in champion_rows)
        + sum(row["pieces"] for row in challenger_rows)
    )

    print()
    print("=" * 80)
    print("PERFORMANCE")
    print("=" * 80)
    print()
    print("Wall time:", f"{elapsed:.2f}s")
    print("Games:", len(requested))
    print("Simulated pieces:", simulated_pieces)
    print(
        "Throughput:",
        f"{simulated_pieces / max(elapsed, 1e-9):.1f} pieces/s",
    )
    print(
        "Game throughput:",
        f"{len(requested) / max(elapsed, 1e-9):.3f} games/s",
    )

    # Pre-declared promotion rules:
    # 1) challenger must be at least as safe as current champion;
    # 2) reward efficiency must improve;
    # 3) average total value must improve.
    survival_pass = (
        s_challenger["gameovers"] <= s_champion["gameovers"]
        and s_challenger["pieces"] >= s_champion["pieces"]
    )

    reward_pass = (
        s_challenger["reward_per_1000"]
        > s_champion["reward_per_1000"]
    )

    value_pass = (
        s_challenger["value"]
        > s_champion["value"]
    )

    tetris_pass = (
        s_challenger["tetrises"]
        > s_champion["tetrises"]
    )

    paired_direction_pass = (
        pair_challenger_champion["wins"]
        > pair_challenger_champion["losses"]
    )

    paired_ci_pass = (
        pair_challenger_champion["lo"] > 0.0
    )

    print()
    print("=" * 80)
    print("CHAMPION PROMOTION DECISION")
    print("=" * 80)
    print()

    print(
        "50K survival >= 30K:",
        "PASS" if survival_pass else "FAIL",
    )

    print(
        "50K Reward/1000 > 30K:",
        "PASS" if reward_pass else "FAIL",
    )

    print(
        "50K total value > 30K:",
        "PASS" if value_pass else "FAIL",
    )

    print(
        "50K Tetris/game > 30K:",
        "PASS" if tetris_pass else "FAIL",
        "(diagnostic; not required)",
    )

    print(
        "50K paired W > L:",
        "PASS" if paired_direction_pass else "FAIL",
        "(diagnostic; not required)",
    )

    print(
        "50K-30K paired 95% CI > 0:",
        "PASS" if paired_ci_pass else "FAIL",
        "(strong-evidence diagnostic; not required)",
    )

    if survival_pass and reward_pass and value_pass:
        print()
        print("V8.6-50K + V8.7 NORMALIZED GATE CHAMPION PROMOTION: PASS")
        print(
            "NEW CHAMPION:",
            args.checkpoint_50k,
            f"@ normalized gate {args.gate_50k:.3f}",
        )
    else:
        print()
        print("V8.6-50K + V8.7 NORMALIZED GATE CHAMPION PROMOTION: FAIL")
        print(
            "KEEP CURRENT CHAMPION:",
            args.checkpoint_30k,
            f"@ raw gate {args.gate_30k:.3f}",
        )

    print()
    print(
        "Do not retune on 3001~3020. "
        "The normalized gate remains frozen. Permanent seeds 6~20 remain sealed."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
