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
from v8_7_scale_invariant_policy import normalized_margin_choice


TOP_K = 4
LINE_VALUE = {0: 0, 1: 900, 2: 2000, 3: 3300, 4: 6000}
PERMANENT_BENCHMARK_FIRST = 6
PERMANENT_BENCHMARK_LAST = 20

MODEL_SPECS = [
    ("champion_30k_raw", "models/v8_5_risk_aware_observable_safe_td_30k.pt", "raw", 0.060),
    ("v8_7_100k_norm", "models/v8_7_normalized_gpu_td_100k.pt", "norm", 0.600),
    ("v8_8_110k_norm", "models/v8_8_jax_vectorized_td_110k.pt", "norm", 0.600),
    ("v8_8_120k_norm", "models/v8_8_jax_vectorized_td_120k.pt", "norm", 0.600),
    ("v8_8_130k_norm", "models/v8_8_jax_vectorized_td_130k.pt", "norm", 0.600),
    ("v8_8_140k_norm", "models/v8_8_jax_vectorized_td_140k.pt", "norm", 0.600),
    ("v8_8_150k_norm", "models/v8_8_jax_vectorized_td_150k.pt", "norm", 0.600),
]

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


def _worker_init(model_specs):
    global _W_MODELS, _W_TEACHER, _W_ADAPTER
    _configure_worker_threads()
    _W_MODELS = {
        label: _load_model(path)
        for label, path, gate_kind, gate in model_specs
    }
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
    gate_kind = str(task["gate_kind"])
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
            q = _q_values(model, state_features, successors)
            if gate_kind == "raw":
                chosen_index, _ = conservative_choice(q, float(gate))
            elif gate_kind == "norm":
                chosen_index, _ = normalized_margin_choice(q, float(gate))
            else:
                raise RuntimeError(f"Unknown gate_kind: {gate_kind}")
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
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def protect_seeds(seeds):
    bad = [s for s in seeds if PERMANENT_BENCHMARK_FIRST <= s <= PERMANENT_BENCHMARK_LAST]
    if bad:
        raise RuntimeError(f"Permanent benchmark seeds 6~20 are protected: {bad}")


def auto_workers():
    logical = os.cpu_count() or 4
    if logical >= 20:
        return 16
    return max(1, logical - 2)


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
    return dict(mean=mean, lo=lo, hi=hi, wins=wins, ties=ties, losses=losses)


def print_summary(label, s):
    print(
        f"{label:<20} "
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
            "V8.8 development-only checkpoint sweep on seeds 4301~4320. "
            "Use this block only to choose ONE V8.8 checkpoint for later untouched qualification."
        )
    )
    parser.add_argument("--seed-start", type=int, default=4301)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-pieces", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--cache",
        default="data/v8_8_checkpoint_dev_4301_4320.json",
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.games))
    protect_seeds(seeds)

    missing_files = [path for _, path, _, _ in MODEL_SPECS if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing_files))

    workers = args.workers if args.workers > 0 else auto_workers()

    policies = [("teacher", "teacher", None)] + [
        (label, gate_kind, gate)
        for label, _, gate_kind, gate in MODEL_SPECS
    ]

    meta = {
        "purpose": "DEVELOPMENT_ONLY_CHECKPOINT_SELECTION",
        "seed_start": args.seed_start,
        "games": args.games,
        "max_pieces": args.max_pieces,
        "models": MODEL_SPECS,
    }

    cache_path = Path(args.cache)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("meta") != meta:
            raise RuntimeError("Cache metadata mismatch; use a new --cache path.")
        cache.setdefault("results", {})
    else:
        cache = {"meta": meta, "results": {}}

    requested = [
        dict(
            seed=seed,
            policy=policy,
            gate_kind=gate_kind,
            gate=gate,
            max_pieces=args.max_pieces,
        )
        for policy, gate_kind, gate in policies
        for seed in seeds
    ]
    missing = [
        t for t in requested
        if result_key(t["seed"], t["policy"]) not in cache["results"]
    ]

    print("=" * 80)
    print("V8.8 DEVELOPMENT CHECKPOINT SWEEP — 4301..4320")
    print("=" * 80)
    print("Purpose: choose ONE V8.8 checkpoint; NOT qualification.")
    print("Frozen normalized gate for V8.7/V8.8:", 0.600)
    print("Formal Champion raw gate:", 0.060)
    print("Seeds:", seeds)
    print("Workers:", workers)
    print("Policies:", [p[0] for p in policies])
    print("Already cached:", len(requested) - len(missing))
    print("Games to simulate:", len(missing))
    print("Permanent seeds 6~20: PROTECTED")
    print()

    start = time.perf_counter()
    if missing:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(MODEL_SPECS,),
        ) as executor:
            futures = [executor.submit(_run_game, t) for t in missing]
            done = 0
            for future in as_completed(futures):
                result = future.result()
                cache["results"][result_key(result["seed"], result["policy"])] = result
                atomic_save(args.cache, cache)
                done += 1
                if done == 1 or done % max(1, len(missing) // 20) == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"new={done:>3}/{len(missing)} "
                        f"elapsed={elapsed:7.1f}s "
                        f"games/s={done/max(elapsed,1e-9):.3f}"
                    )

    rows = {}
    for policy, _, _ in policies:
        rr = [cache["results"][result_key(seed, policy)] for seed in seeds]
        rr.sort(key=lambda x: x["seed"])
        rows[policy] = rr

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    summaries = {}
    for policy, _, _ in policies:
        summaries[policy] = summarize(rows[policy])
        print_summary(policy, summaries[policy])

    base_label = "v8_7_100k_norm"
    champion_label = "champion_30k_raw"

    print()
    print("PAIRED VALUE VS 100K NORMALIZED BASE")
    for label, _, _, _ in MODEL_SPECS:
        if label in (base_label, champion_label):
            continue
        p = paired_stats(rows[label], rows[base_label])
        print(
            f"{label:<20} mean={p['mean']:+9.1f} "
            f"95%CI=[{p['lo']:+9.1f},{p['hi']:+9.1f}] "
            f"W/T/L={p['wins']}/{p['ties']}/{p['losses']}"
        )

    print()
    print("PAIRED VALUE VS FORMAL 30K CHAMPION")
    for label, _, _, _ in MODEL_SPECS:
        if label == champion_label:
            continue
        p = paired_stats(rows[label], rows[champion_label])
        print(
            f"{label:<20} mean={p['mean']:+9.1f} "
            f"95%CI=[{p['lo']:+9.1f},{p['hi']:+9.1f}] "
            f"W/T/L={p['wins']}/{p['ties']}/{p['losses']}"
        )

    # Development ranking only. Safety first, then R/1000, then value.
    v88_labels = [f"v8_8_{k}k_norm" for k in (110, 120, 130, 140, 150)]
    ranked = sorted(
        v88_labels,
        key=lambda label: (
            summaries[label]["gameovers"],
            -summaries[label]["reward_per_1000"],
            -summaries[label]["value"],
        ),
    )

    selected = ranked[0]
    print()
    print("=" * 80)
    print("DEVELOPMENT SELECTION")
    print("=" * 80)
    print("Selected V8.8 checkpoint:", selected)
    print(
        "Selection rule: lowest gameovers, then highest R/1000, then highest average value."
    )
    print(
        "IMPORTANT: 4301~4320 are now DEVELOPMENT-CONSUMED. "
        "Do not use them for formal promotion qualification."
    )
    print(
        "Next: qualify only the selected checkpoint against the formal Champion "
        "on one untouched predeclared block (recommended 3101~3120)."
    )


if __name__ == "__main__":
    main()
