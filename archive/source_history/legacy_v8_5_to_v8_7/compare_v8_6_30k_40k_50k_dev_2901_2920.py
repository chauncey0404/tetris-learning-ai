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
from v8_4_observable import compact_candidate_arrays, conservative_choice


TOP_K = 4
LINE_VALUE = {0: 0, 1: 900, 2: 2000, 3: 3300, 4: 6000}

_W_MODELS = {}
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
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ObservableSafeQNetwork().cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _worker_init(checkpoint_30k, checkpoint_40k, checkpoint_50k):
    global _W_MODELS, _W_TEACHER, _W_ADAPTER
    _configure_worker_threads()
    _W_MODELS = {
        "v8_5_30k": _load_model(checkpoint_30k),
        "v8_6_40k": _load_model(checkpoint_40k),
        "v8_6_50k": _load_model(checkpoint_50k),
    }
    _W_TEACHER = HeuristicTeacherV2()
    _W_ADAPTER = GymTetrisAdapter()
    atexit.register(_close_worker)


def _height(features):
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
    model = None if policy == "teacher" else _W_MODELS[policy]

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

        if policy == "teacher":
            chosen_index = 0
        else:
            q = _q_values(model, state_features, successors)
            chosen_index, _ = conservative_choice(q, float(gate))
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
        "policy": policy,
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


def result_key(seed, policy):
    return f"{int(seed)}|{policy}"


def atomic_save(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def load_cache(path, meta):
    p = Path(path)
    if not p.exists():
        return {"meta": meta, "results": {}}

    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("results", {})
    old_meta = data.get("meta", {})
    for key, value in meta.items():
        if old_meta.get(key) != value:
            raise RuntimeError(
                f"Cache metadata mismatch for {key}: "
                f"{old_meta.get(key)!r} != {value!r}"
            )
    return data


def summarize(rows):
    total_pieces = sum(r["pieces"] for r in rows)
    total_value = sum(r["value"] for r in rows)
    total_switch = sum(r["interventions"] for r in rows)
    return {
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


def paired(rows_a, rows_b):
    a = {r["seed"]: r for r in rows_a}
    b = {r["seed"]: r for r in rows_b}
    seeds = sorted(set(a) & set(b))
    diffs = np.asarray(
        [float(a[s]["value"] - b[s]["value"]) for s in seeds],
        dtype=np.float64,
    )
    mean = float(np.mean(diffs))
    if len(diffs) > 1:
        se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    else:
        lo = hi = mean
    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "wins": int(np.sum(diffs > 0)),
        "ties": int(np.sum(diffs == 0)),
        "losses": int(np.sum(diffs < 0)),
    }


def print_summary(label, s):
    print(
        f"{label:<12} "
        f"pieces={s['pieces']:7.2f} "
        f"lines={s['lines']:7.2f} "
        f"Tetris={s['tetrises']:6.2f} "
        f"value={s['value']:9.2f} "
        f"R/1000={s['reward_per_1000']:9.2f} "
        f"avgH={s['avg_height']:5.2f} "
        f"worstH={s['worst_height']:>2} "
        f"GO={s['gameovers']} "
        f"switch={s['switch_rate']:5.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Development checkpoint progression test on untouched-for-training "
            "seeds 2901~2920. Uses the same frozen gate 0.060 for 30K/40K/50K "
            "before any new gate calibration."
        )
    )
    parser.add_argument(
        "--checkpoint-30k",
        default="models/v8_5_risk_aware_observable_safe_td_30k.pt",
    )
    parser.add_argument(
        "--checkpoint-40k",
        default="models/v8_6_risk_aware_observable_safe_td_40k.pt",
    )
    parser.add_argument(
        "--checkpoint-50k",
        default="models/v8_6_risk_aware_observable_safe_td_50k.pt",
    )
    parser.add_argument("--gate", type=float, default=0.060)
    parser.add_argument("--seed-start", type=int, default=2901)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-pieces", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--cache",
        default="data/v8_6_checkpoint_progression_2901_2920.json",
    )
    parser.add_argument("--quiet-games", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.games))
    if any(6 <= s <= 20 for s in seeds):
        raise RuntimeError("Permanent seeds 6~20 are protected.")

    meta = {
        "checkpoint_30k": args.checkpoint_30k,
        "checkpoint_40k": args.checkpoint_40k,
        "checkpoint_50k": args.checkpoint_50k,
        "gate": float(args.gate),
        "seeds": seeds,
        "max_pieces": int(args.max_pieces),
        "purpose": "DEV_CHECKPOINT_PROGRESSION_PRE_GATE_CALIBRATION",
    }
    cache = load_cache(args.cache, meta)

    policies = [
        ("teacher", None),
        ("v8_5_30k", args.gate),
        ("v8_6_40k", args.gate),
        ("v8_6_50k", args.gate),
    ]

    tasks = [
        {
            "seed": seed,
            "policy": policy,
            "gate": gate,
            "max_pieces": args.max_pieces,
        }
        for policy, gate in policies
        for seed in seeds
        if result_key(seed, policy) not in cache["results"]
    ]

    print("=" * 80)
    print("V8.6 DEVELOPMENT CHECKPOINT PROGRESSION: 30K vs 40K vs 50K")
    print("=" * 80)
    print("Development seeds:", seeds)
    print("Common pre-calibration gate:", args.gate)
    print("Workers:", args.workers)
    print("Requested games:", len(policies) * len(seeds))
    print("Already cached:", len(policies) * len(seeds) - len(tasks))
    print("New games:", len(tasks))
    print("Permanent seeds 6~20: PROTECTED")
    print("3001~3020 fresh qualification set: NOT USED")
    print()
    print(
        "IMPORTANT: this run compares checkpoint progression at the SAME gate. "
        "Do not freeze a new gate from this test alone."
    )

    start = time.perf_counter()

    if tasks:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                args.checkpoint_30k,
                args.checkpoint_40k,
                args.checkpoint_50k,
            ),
        ) as ex:
            futures = [ex.submit(_run_game, t) for t in tasks]
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                cache["results"][result_key(r["seed"], r["policy"])] = r
                atomic_save(args.cache, cache)
                done += 1
                if done == 1 or done % max(1, len(tasks) // 10) == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"new={done:>3}/{len(tasks)} "
                        f"elapsed={elapsed:7.1f}s "
                        f"games/s={done / max(elapsed, 1e-9):.3f}"
                    )

    elapsed = time.perf_counter() - start

    rows = {}
    for policy, _ in policies:
        rows[policy] = sorted(
            [cache["results"][result_key(s, policy)] for s in seeds],
            key=lambda r: r["seed"],
        )

    if not args.quiet_games:
        print()
        print("=" * 80)
        print("PER-SEED RESULTS")
        print("=" * 80)
        for seed in seeds:
            r = {p: next(x for x in rows[p] if x["seed"] == seed) for p, _ in policies}
            print(
                f"seed={seed} | "
                f"T={r['teacher']['value']:>7} GO={int(r['teacher']['game_over'])} | "
                f"30K={r['v8_5_30k']['value']:>7} GO={int(r['v8_5_30k']['game_over'])} "
                f"sw={r['v8_5_30k']['intervention_rate']:5.2f}% | "
                f"40K={r['v8_6_40k']['value']:>7} GO={int(r['v8_6_40k']['game_over'])} "
                f"sw={r['v8_6_40k']['intervention_rate']:5.2f}% | "
                f"50K={r['v8_6_50k']['value']:>7} GO={int(r['v8_6_50k']['game_over'])} "
                f"sw={r['v8_6_50k']['intervention_rate']:5.2f}%"
            )

    summaries = {p: summarize(rows[p]) for p, _ in policies}

    print()
    print("=" * 80)
    print("CHECKPOINT PROGRESSION SUMMARY @ GATE 0.060")
    print("=" * 80)
    print_summary("Teacher", summaries["teacher"])
    print_summary("V8.5-30K", summaries["v8_5_30k"])
    print_summary("V8.6-40K", summaries["v8_6_40k"])
    print_summary("V8.6-50K", summaries["v8_6_50k"])

    comparisons = [
        ("40K - 30K", "v8_6_40k", "v8_5_30k"),
        ("50K - 30K", "v8_6_50k", "v8_5_30k"),
        ("50K - 40K", "v8_6_50k", "v8_6_40k"),
    ]

    print()
    print("=" * 80)
    print("PAIRED VALUE COMPARISONS")
    print("=" * 80)
    for label, a, b in comparisons:
        p = paired(rows[a], rows[b])
        print(
            f"{label:<12} mean={p['mean']:+9.1f} "
            f"95%CI=[{p['lo']:+9.1f}, {p['hi']:+9.1f}] "
            f"W/T/L={p['wins']}/{p['ties']}/{p['losses']}"
        )

    s30 = summaries["v8_5_30k"]
    s40 = summaries["v8_6_40k"]
    s50 = summaries["v8_6_50k"]

    print()
    print("=" * 80)
    print("NEXT-STEP SIGNAL")
    print("=" * 80)

    for label, s in [("40K", s40), ("50K", s50)]:
        survival = (
            s["gameovers"] <= s30["gameovers"]
            and s["pieces"] >= s30["pieces"]
        )
        reward = s["reward_per_1000"] > s30["reward_per_1000"]
        value = s["value"] > s30["value"]
        print(
            f"{label}: survival={'PASS' if survival else 'FAIL'} "
            f"R/1000={'PASS' if reward else 'FAIL'} "
            f"value={'PASS' if value else 'FAIL'}"
        )

    print()
    print(
        "This is a progression screen, not final model selection. "
        "Next calibrate gates only for the checkpoint(s) that remain competitive."
    )
    print("Wall time:", f"{elapsed:.2f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
