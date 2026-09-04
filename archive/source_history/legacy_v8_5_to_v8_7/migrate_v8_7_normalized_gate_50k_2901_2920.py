import argparse
import atexit
import json
import math
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
from v8_4_observable import compact_candidate_arrays
from v8_7_scale_invariant_policy import normalized_margin_choice


TOP_K = 4
LINE_VALUE = {0: 0, 1: 900, 2: 2000, 3: 3300, 4: 6000}
PERMANENT_FIRST = 6
PERMANENT_LAST = 20

_W_MODEL = None
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


def _worker_init(checkpoint_path):
    global _W_MODEL, _W_TEACHER, _W_ADAPTER
    _configure_worker_threads()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = ObservableSafeQNetwork().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _W_MODEL = model
    _W_TEACHER = HeuristicTeacherV2()
    _W_ADAPTER = GymTetrisAdapter()
    atexit.register(_close_worker)


def _height(features):
    board = np.asarray(features[:200], dtype=np.float32).reshape(20, 10)
    rows = np.where(np.any(board != 0.0, axis=1))[0]
    return 0 if rows.size == 0 else int(20 - rows[0])


@torch.inference_mode()
def _q_values(state_features, successors):
    candidates, rewards, scores, ranks = compact_candidate_arrays(successors)

    q = _W_MODEL(
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
    gate = task["gate"]
    max_pieces = int(task["max_pieces"])

    adapter = _W_ADAPTER
    teacher = _W_TEACHER

    state = adapter.reset(seed=seed)
    adapter.raw.gravity_enabled = False
    state_features = encode_state(state).astype(np.float32, copy=True)

    pieces = 0
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    interventions = 0
    max_height = _height(state_features)
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

        if gate is None:
            chosen_index = 0
        else:
            q = _q_values(state_features, successors)
            chosen_index, _ = normalized_margin_choice(q, float(gate))
            if chosen_index != 0:
                interventions += 1

        chosen = successors[chosen_index]
        result = execute_placement(adapter, chosen.action)

        state = result["state"]
        pieces += 1

        state_features = np.asarray(
            chosen.next_state_features,
            dtype=np.float32,
        ).copy()

        lines = int(result["info"].get("lines_cleared", 0))
        if lines in counts:
            counts[lines] += 1

        max_height = max(max_height, _height(state_features))

        if bool(result["terminated"] or result["truncated"]):
            game_over = True
            break

    total_lines = sum(k * v for k, v in counts.items())
    value = sum(LINE_VALUE[k] * v for k, v in counts.items())

    return {
        "seed": seed,
        "gate": None if gate is None else float(gate),
        "pieces": int(pieces),
        "lines": int(total_lines),
        "tetrises": int(counts[4]),
        "value": int(value),
        "interventions": int(interventions),
        "intervention_rate": interventions / max(pieces, 1) * 100.0,
        "max_height": int(max_height),
        "game_over": bool(game_over),
    }


def result_key(seed, gate):
    if gate is None:
        return f"{int(seed)}|teacher"
    return f"{int(seed)}|{float(gate):.6f}"


def atomic_save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def load_or_create_cache(path, checkpoint, seeds, max_pieces):
    p = Path(path)
    expected = {
        "checkpoint": checkpoint,
        "seeds": [int(s) for s in seeds],
        "max_pieces": int(max_pieces),
        "purpose": "V8_7_ONE_TIME_SCALE_INVARIANT_GATE_MIGRATION",
    }

    if not p.exists():
        return {"meta": expected, "results": {}}

    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("results", {})

    meta = data.get("meta", {})
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Cache metadata mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    return data


def load_progression_cache(path):
    p = Path(path)
    if not p.exists():
        return None

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(data.get("results"), dict):
        return None

    return data


def parse_gates(text):
    values = []
    seen = set()

    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue

        gate = float(token)
        if gate < 0:
            raise ValueError("Gate must be >= 0")

        key = round(gate, 12)
        if key not in seen:
            seen.add(key)
            values.append(gate)

    if not values:
        raise ValueError("At least one gate is required.")

    return values


def rows_for_gate(cache, seeds, gate):
    rows = []
    for seed in seeds:
        key = result_key(seed, gate)
        if key in cache["results"]:
            rows.append(cache["results"][key])

    rows.sort(key=lambda x: x["seed"])
    return rows


def summarize(rows):
    if not rows:
        return None

    total_pieces = sum(r["pieces"] for r in rows)
    total_value = sum(r["value"] for r in rows)
    total_switch = sum(r["interventions"] for r in rows)

    return {
        "n": len(rows),
        "pieces": float(np.mean([r["pieces"] for r in rows])),
        "lines": float(np.mean([r["lines"] for r in rows])),
        "tetrises": float(np.mean([r["tetrises"] for r in rows])),
        "value": float(np.mean([r["value"] for r in rows])),
        "reward_per_1000": total_value / max(total_pieces, 1) * 1000.0,
        "avg_height": float(np.mean([r["max_height"] for r in rows])),
        "worst_height": int(max(r["max_height"] for r in rows)),
        "gameovers": int(sum(bool(r["game_over"]) for r in rows)),
        "switch_rate": total_switch / max(total_pieces, 1) * 100.0,
    }


def paired_stats(rows_a, rows_b):
    a = {r["seed"]: r for r in rows_a}
    b = {r["seed"]: r for r in rows_b}
    seeds = sorted(set(a) & set(b))

    diffs = np.asarray(
        [float(a[s]["value"] - b[s]["value"]) for s in seeds],
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


def copy_progression_baselines(
    gate_cache,
    progression,
    seeds,
    reference_gate,
):
    if progression is None:
        return 0, []

    copied = 0
    reference_30k = []
    src = progression.get("results", {})

    for seed in seeds:
        teacher_src_key = f"{seed}|teacher"
        teacher_dst_key = result_key(seed, None)

        if (
            teacher_dst_key not in gate_cache["results"]
            and teacher_src_key in src
        ):
            row = dict(src[teacher_src_key])
            row.pop("policy", None)
            row["gate"] = None
            gate_cache["results"][teacher_dst_key] = row
            copied += 1

        thirty_src_key = f"{seed}|v8_5_30k"
        if thirty_src_key in src:
            reference_30k.append(dict(src[thirty_src_key]))

    reference_30k.sort(key=lambda r: r["seed"])
    return copied, reference_30k


def risk_order_from_progression(progression, seeds):
    if progression is None:
        return list(seeds)

    src = progression.get("results", {})
    score = {seed: 0.0 for seed in seeds}

    for seed in seeds:
        for policy in ("v8_5_30k", "v8_6_40k", "v8_6_50k"):
            row = src.get(f"{seed}|{policy}")
            if not row:
                continue

            if bool(row.get("game_over", False)):
                score[seed] += 1000.0

            pieces = int(row.get("pieces", 2000))
            max_h = int(row.get("max_height", 0))
            value = int(row.get("value", 0))

            score[seed] += max(0, 2000 - pieces) / 10.0
            score[seed] += max(0, max_h - 15) * 2.0
            score[seed] += max(0, 950000 - value) / 50000.0

    return sorted(seeds, key=lambda s: (-score[s], s))


def print_summary(label, s, teacher, total_games):
    if s is None:
        print(f"{label:<12} NO RESULTS")
        return

    if teacher is None:
        dr = dv = 0.0
    else:
        dr = (
            s["reward_per_1000"]
            / teacher["reward_per_1000"]
            - 1.0
        ) * 100.0

        dv = (
            s["value"]
            / teacher["value"]
            - 1.0
        ) * 100.0

    complete = s["n"] == total_games
    status = (
        "SAFE"
        if complete and s["gameovers"] == 0
        else "UNSAFE"
        if s["gameovers"] > 0
        else "PARTIAL"
    )

    print(
        f"{label:<12} "
        f"n={s['n']:>2}/{total_games} "
        f"pieces={s['pieces']:7.2f} "
        f"lines={s['lines']:7.2f} "
        f"Tetris={s['tetrises']:6.2f} "
        f"value={s['value']:9.2f} "
        f"R/1000={s['reward_per_1000']:9.2f} "
        f"avgH={s['avg_height']:5.2f} "
        f"worstH={s['worst_height']:>2} "
        f"GO={s['gameovers']} "
        f"switch={s['switch_rate']:5.2f}% "
        f"dR={dr:+6.2f}% "
        f"dValue={dv:+6.2f}% "
        f"[{status}]"
    )


def run_tasks(tasks, workers, checkpoint, cache, cache_path, label):
    if not tasks:
        return 0, 0.0

    start = time.perf_counter()
    done = 0
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(checkpoint,),
    ) as executor:
        futures = [executor.submit(_run_game, t) for t in tasks]

        for future in as_completed(futures):
            result = future.result()

            cache["results"][
                result_key(result["seed"], result["gate"])
            ] = result

            atomic_save(cache_path, cache)
            done += 1

            if done == 1 or done % max(1, len(tasks) // 10) == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"{label}={done:>3}/{len(tasks)} "
                    f"elapsed={elapsed:7.1f}s "
                    f"games/s={done / max(elapsed, 1e-9):.3f}"
                )

    return done, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Final V8.6-50K development gate calibration on 2901~2920. "
            "This is the only gate sweep for the 50K checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default="models/v8_6_risk_aware_observable_safe_td_50k.pt",
    )

    parser.add_argument(
        "--progression-cache",
        default="data/v8_6_checkpoint_progression_2901_2920.json",
    )

    parser.add_argument(
        "--cache",
        default="data/v8_7_normalized_gate_migration_2901_2920.json",
    )

    parser.add_argument("--seed-start", type=int, default=2901)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-pieces", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)

    parser.add_argument(
        "--gates",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80",
    )

    parser.add_argument(
        "--reference-gate",
        type=float,
        default=0.060,
        help="Historical absolute gate for the 30K reference only; not a normalized gate.",
    )

    parser.add_argument(
        "--screen-seeds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    gates = parse_gates(args.gates)
    seeds = list(range(args.seed_start, args.seed_start + args.games))

    if any(PERMANENT_FIRST <= s <= PERMANENT_LAST for s in seeds):
        raise RuntimeError("Permanent seeds 6~20 are protected.")

    if args.screen_seeds <= 0 or args.screen_seeds > len(seeds):
        raise ValueError("--screen-seeds must be in [1, games].")

    cache = load_or_create_cache(
        args.cache,
        args.checkpoint,
        seeds,
        args.max_pieces,
    )

    progression = load_progression_cache(args.progression_cache)

    copied, reference_30k = copy_progression_baselines(
        cache,
        progression,
        seeds,
        args.reference_gate,
    )

    if copied:
        atomic_save(args.cache, cache)

    risk_order = risk_order_from_progression(
        progression,
        seeds,
    )

    print()
    print("=" * 80)
    print("V8.7 ONE-TIME SCALE-INVARIANT GATE MIGRATION")
    print("=" * 80)
    print()
    print("Checkpoint:", args.checkpoint)
    print("Development seeds:", seeds)
    print("Normalized gates:", gates)
    print("Historical 30K absolute gate:", args.reference_gate)
    print("Workers:", args.workers)
    print("Progression cache:", args.progression_cache)
    print("Gate cache:", args.cache)
    print("Copied Teacher rows:", copied)
    print("Historical hard-seed order:", risk_order)
    print("Stage-1 screen seeds/gate:", args.screen_seeds)
    print("Two-stage safety pruning: ENABLED")
    print("Per-game resume: ENABLED")
    print("Permanent seeds 6~20: PROTECTED")
    print("3001~3020 fresh qualification: NOT USED")
    print()
    print(
        "IMPORTANT: this is a ONE-TIME migration from raw-Q gap to normalized-Q margin. "
        "After this, freeze the normalized gate across future checkpoints."
    )

    total_start = time.perf_counter()

    # Teacher fallback only if progression cache was missing/incomplete.
    teacher_missing = [
        {
            "seed": seed,
            "gate": None,
            "max_pieces": args.max_pieces,
        }
        for seed in seeds
        if result_key(seed, None) not in cache["results"]
    ]

    if teacher_missing:
        print()
        print("Teacher fallback games:", len(teacher_missing))
        run_tasks(
            teacher_missing,
            args.workers,
            args.checkpoint,
            cache,
            args.cache,
            "teacher",
        )

    known_unsafe = {
        gate
        for gate in gates
        if any(
            bool(r["game_over"])
            for r in rows_for_gate(cache, seeds, gate)
        )
    }

    stage1 = []

    for gate in gates:
        if gate in known_unsafe:
            continue

        tested = {
            r["seed"]
            for r in rows_for_gate(cache, seeds, gate)
        }

        for seed in risk_order[: args.screen_seeds]:
            if seed not in tested:
                stage1.append(
                    {
                        "seed": seed,
                        "gate": gate,
                        "max_pieces": args.max_pieces,
                    }
                )

    print()
    print("Stage-1 new games:", len(stage1))

    run_tasks(
        stage1,
        args.workers,
        args.checkpoint,
        cache,
        args.cache,
        "screen",
    )

    disqualified = {
        gate
        for gate in gates
        if any(
            bool(r["game_over"])
            for r in rows_for_gate(cache, seeds, gate)
        )
    }

    print("Stage-1 disqualified gates:", sorted(disqualified))

    stage2 = []

    for gate in gates:
        if gate in disqualified:
            continue

        tested = {
            r["seed"]
            for r in rows_for_gate(cache, seeds, gate)
        }

        for seed in seeds:
            if seed not in tested:
                stage2.append(
                    {
                        "seed": seed,
                        "gate": gate,
                        "max_pieces": args.max_pieces,
                    }
                )

    print("Stage-2 new games:", len(stage2))

    run_tasks(
        stage2,
        args.workers,
        args.checkpoint,
        cache,
        args.cache,
        "full",
    )

    teacher_rows = rows_for_gate(cache, seeds, None)
    teacher_summary = summarize(teacher_rows)

    gate_rows = {
        gate: rows_for_gate(cache, seeds, gate)
        for gate in gates
    }

    gate_summaries = {
        gate: summarize(rows)
        for gate, rows in gate_rows.items()
    }

    print()
    print("=" * 80)
    print("V8.7 NORMALIZED-GATE MIGRATION SUMMARY")
    print("=" * 80)
    print()

    print_summary(
        "Teacher",
        teacher_summary,
        None,
        len(seeds),
    )

    for gate in gates:
        print_summary(
            f"NGate {gate:.2f}",
            gate_summaries[gate],
            teacher_summary,
            len(seeds),
        )

    safe = []

    for gate in gates:
        s = gate_summaries[gate]
        if (
            s is not None
            and s["n"] == len(seeds)
            and s["gameovers"] == 0
        ):
            safe.append((gate, s))

    if not safe:
        raise RuntimeError(
            "No fully evaluated zero-GO gate survived the 50K calibration."
        )

    best_gate, best = max(
        safe,
        key=lambda item: (
            item[1]["reward_per_1000"],
            -item[1]["avg_height"],
            -item[1]["switch_rate"],
        ),
    )

    best_vs_teacher = paired_stats(
        gate_rows[best_gate],
        teacher_rows,
    )

    print()
    print("=" * 80)
    print("ONE-TIME NORMALIZED GATE RESULT")
    print("=" * 80)
    print()
    print("Frozen V8.7 normalized gate:", f"{best_gate:.3f}")
    print("Game overs:", best["gameovers"])
    print("Reward/1000:", f"{best['reward_per_1000']:.2f}")
    print(
        "vs Teacher:",
        f"{(best['reward_per_1000'] / teacher_summary['reward_per_1000'] - 1.0) * 100.0:+.2f}%",
    )
    print(
        "Total value vs Teacher:",
        f"{(best['value'] / teacher_summary['value'] - 1.0) * 100.0:+.2f}%",
    )
    print("Tetris/game:", f"{best['tetrises']:.2f}")
    print("Switch rate:", f"{best['switch_rate']:.2f}%")
    print("Avg/Worst maxH:", f"{best['avg_height']:.2f} / {best['worst_height']}")
    print(
        "Paired vs Teacher:",
        f"mean={best_vs_teacher['mean']:+.1f}",
        f"95%CI=[{best_vs_teacher['lo']:+.1f}, {best_vs_teacher['hi']:+.1f}]",
        f"W/T/L={best_vs_teacher['wins']}/{best_vs_teacher['ties']}/{best_vs_teacher['losses']}",
    )

    if len(reference_30k) == len(seeds):
        reference_30k.sort(key=lambda r: r["seed"])
        s30 = summarize(reference_30k)
        p30 = paired_stats(
            gate_rows[best_gate],
            reference_30k,
        )

        print()
        print("Reference V8.5-30K @0.060 on same dev seeds:")
        print_summary(
            "V8.5-30K",
            s30,
            teacher_summary,
            len(seeds),
        )

        print(
            "Best 50K vs 30K:",
            f"R/1000={(best['reward_per_1000'] / s30['reward_per_1000'] - 1.0) * 100.0:+.2f}%",
            f"value={(best['value'] / s30['value'] - 1.0) * 100.0:+.2f}%",
        )

        print(
            "Paired 50K - 30K:",
            f"mean={p30['mean']:+.1f}",
            f"95%CI=[{p30['lo']:+.1f}, {p30['hi']:+.1f}]",
            f"W/T/L={p30['wins']}/{p30['ties']}/{p30['losses']}",
        )

    elapsed = time.perf_counter() - total_start

    print()
    print("=" * 80)
    print("FREEZE DECISION")
    print("=" * 80)
    print()
    print(
        f"V8.7 NORMALIZED GATE FROZEN: {best_gate:.3f}"
    )
    print("No further normalized-gate tuning on 2901~2920 or later checkpoints.")
    print("Next: untouched 3001~3020 migration/Champion qualification.")
    print(
        "Compare current Champion V8.5-30K @ raw gate 0.060 "
        "against V8.6-50K at this frozen normalized gate."
    )
    print()
    print("Wall time:", f"{elapsed:.2f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
