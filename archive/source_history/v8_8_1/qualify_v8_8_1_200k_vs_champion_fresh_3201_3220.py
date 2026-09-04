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
QUAL_SEED_START = 3201
QUAL_GAMES = 20
CHAMPION_LABEL = "champion_150k_norm"
CHALLENGER_LABEL = "v8_8_1_200k_norm"

MODEL_SPECS = [
    (CHAMPION_LABEL, "models/v8_8_jax_vectorized_td_150k.pt", "norm", 0.600),
    (CHALLENGER_LABEL, "models/v8_8_1_longtraj_gpu_replay_td_200k.pt", "norm", 0.600),
]

_W_MODELS = {}
_W_TEACHER = None
_W_ADAPTER = None

def _close_worker():
    global _W_ADAPTER
    if _W_ADAPTER is not None:
        try:
            _W_ADAPTER.close()
        except Exception:
            pass
        _W_ADAPTER = None

def _load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ObservableSafeQNetwork().cpu()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def _worker_init(model_specs):
    global _W_MODELS, _W_TEACHER, _W_ADAPTER
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _W_MODELS = {label: _load_model(path) for label, path, _, _ in model_specs}
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
        state=torch.from_numpy(np.asarray(state_features, dtype=np.float32)).unsqueeze(0),
        candidates=torch.from_numpy(candidates).unsqueeze(0),
        rewards=torch.from_numpy(rewards).unsqueeze(0),
        teacher_scores=torch.from_numpy(scores).unsqueeze(0),
        teacher_ranks=torch.from_numpy(ranks).unsqueeze(0),
    )[0]
    return q.detach().numpy()

def _run_game(task):
    seed = int(task["seed"])
    policy = task["policy"]
    gate_kind = task["gate_kind"]
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
        successors = preview_top_k_successors(adapter=adapter, teacher=teacher, state=state, top_k=TOP_K)
        if not successors:
            game_over = True
            break

        if policy == "teacher":
            chosen_index = 0
        else:
            q = _q_values(model, state_features, successors)
            if gate_kind == "raw":
                chosen_index, _ = conservative_choice(q, float(gate))
            else:
                chosen_index, _ = normalized_margin_choice(q, float(gate))
            if chosen_index != 0:
                interventions += 1

        chosen = successors[chosen_index]
        result = execute_placement(adapter, chosen.action)
        state = result["state"]
        state_features = np.asarray(chosen.next_state_features, dtype=np.float32).copy()
        pieces += 1

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
        "seed": seed, "policy": policy, "pieces": pieces, "lines": total_lines,
        "tetrises": counts[4], "value": value, "interventions": interventions,
        "max_height": max_height, "game_over": game_over,
    }

def _key(seed, policy):
    return f"{seed}|{policy}"

def _save(path, payload):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)

def _summary(rows):
    total_pieces = sum(r["pieces"] for r in rows)
    total_value = sum(r["value"] for r in rows)
    total_switch = sum(r["interventions"] for r in rows)
    return {
        "pieces": float(np.mean([r["pieces"] for r in rows])),
        "lines": float(np.mean([r["lines"] for r in rows])),
        "tetrises": float(np.mean([r["tetrises"] for r in rows])),
        "value": float(np.mean([r["value"] for r in rows])),
        "r1000": total_value / max(total_pieces, 1) * 1000.0,
        "avgH": float(np.mean([r["max_height"] for r in rows])),
        "worstH": int(max(r["max_height"] for r in rows)),
        "GO": int(sum(bool(r["game_over"]) for r in rows)),
        "switch": total_switch / max(total_pieces, 1) * 100.0,
    }

def _paired(a_rows, b_rows):
    a = {r["seed"]: r for r in a_rows}
    b = {r["seed"]: r for r in b_rows}
    seeds = sorted(a)
    d = np.asarray([float(a[s]["value"] - b[s]["value"]) for s in seeds], dtype=np.float64)
    mean = float(np.mean(d))
    if len(d) > 1:
        se = float(np.std(d, ddof=1) / math.sqrt(len(d)))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
    else:
        lo = hi = mean
    return mean, lo, hi, int(np.sum(d > 0)), int(np.sum(d == 0)), int(np.sum(d < 0))

def _print_summary(label, s):
    print(
        f"{label:<20} pieces={s['pieces']:7.2f} lines={s['lines']:7.2f} "
        f"Tetris={s['tetrises']:6.2f} value={s['value']:9.2f} "
        f"R/1000={s['r1000']:9.2f} avgH={s['avgH']:5.2f} "
        f"worstH={s['worstH']:>2} GO={s['GO']} switch={s['switch']:5.2f}%"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-pieces", type=int, default=2000)
    ap.add_argument("--cache", default="data/v8_8_1_200k_qualification_3201_3220.json")
    args = ap.parse_args()

    seeds = list(range(3201, 3221))
    for _, path, _, _ in MODEL_SPECS:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    policies = [
        ("teacher", "teacher", None),
        (CHAMPION_LABEL, "norm", 0.600),
        (CHALLENGER_LABEL, "norm", 0.600),
    ]
    meta = {
        "purpose": "ONE_TIME_FRESH_FORMAL_QUALIFICATION",
        "seeds": seeds,
        "max_pieces": args.max_pieces,
        "champion": MODEL_SPECS[0],
        "challenger": MODEL_SPECS[1],
    }

    cache_path = Path(args.cache)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("meta") != meta:
            raise RuntimeError("Qualification cache metadata mismatch.")
    else:
        cache = {"meta": meta, "results": {}}

    tasks = [
        {"seed": seed, "policy": p, "gate_kind": gk, "gate": g, "max_pieces": args.max_pieces}
        for p, gk, g in policies for seed in seeds
    ]
    missing = [t for t in tasks if _key(t["seed"], t["policy"]) not in cache["results"]]

    print("=" * 80)
    print("V8.8.1 200K FRESH FORMAL QUALIFICATION — 3201..3220")
    print("=" * 80)
    print("Formal Champion : V8.8 150K @ NORMALIZED gate 0.600")
    print("Challenger      : V8.8.1 200K @ NORMALIZED gate 0.600")
    print("Teacher         : reference only")
    print("Seeds           :", seeds)
    print("Workers         :", args.workers)
    print("Already cached  :", len(tasks) - len(missing))
    print("Games to run    :", len(missing))
    print("Permanent seeds 6~20: PROTECTED")

    if missing:
        ctx = mp.get_context("spawn")
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
                                 initializer=_worker_init, initargs=(MODEL_SPECS,)) as ex:
            futs = [ex.submit(_run_game, t) for t in missing]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                cache["results"][_key(r["seed"], r["policy"])] = r
                _save(args.cache, cache)
                if i == 1 or i % max(1, len(missing)//20) == 0:
                    elapsed = time.perf_counter() - start
                    print(f"new={i:>3}/{len(missing)} elapsed={elapsed:7.1f}s games/s={i/max(elapsed,1e-9):.3f}")

    rows = {}
    for p, _, _ in policies:
        rows[p] = [cache["results"][_key(seed, p)] for seed in seeds]

    S = {p: _summary(rows[p]) for p, _, _ in policies}

    print()
    print("=" * 80)
    print("QUALIFICATION SUMMARY")
    print("=" * 80)
    _print_summary("teacher", S["teacher"])
    _print_summary(CHAMPION_LABEL, S[CHAMPION_LABEL])
    _print_summary(CHALLENGER_LABEL, S[CHALLENGER_LABEL])

    mean, lo, hi, w, t, l = _paired(rows[CHALLENGER_LABEL], rows[CHAMPION_LABEL])
    print()
    print(f"Paired value challenger-champion: mean={mean:+.1f} 95%CI=[{lo:+.1f},{hi:+.1f}] W/T/L={w}/{t}/{l}")
    print("CI and W/T/L are diagnostics only.")

    c = S[CHAMPION_LABEL]
    x = S[CHALLENGER_LABEL]
    gates = [
        ("Safety GO", x["GO"] <= c["GO"], f"{x['GO']} <= {c['GO']}"),
        ("Survival pieces", x["pieces"] >= c["pieces"], f"{x['pieces']:.2f} >= {c['pieces']:.2f}"),
        ("R/1000", x["r1000"] > c["r1000"], f"{x['r1000']:.2f} > {c['r1000']:.2f}"),
        ("Average value", x["value"] > c["value"], f"{x['value']:.2f} > {c['value']:.2f}"),
    ]

    print()
    print("=" * 80)
    print("PROMOTION GATES")
    print("=" * 80)
    ok = True
    for name, passed, detail in gates:
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL':<4} {name}: {detail}")

    print()
    if ok:
        print("FINAL RESULT: PROMOTION PASS")
        print("V8.8.1 200K qualifies to replace V8.8 150K as formal Champion.")
    else:
        print("FINAL RESULT: PROMOTION FAIL")
        print("Retain V8.8 150K as formal Champion.")

    print("Seeds 3201~3220 are now QUALIFICATION-CONSUMED. Do not use them for tuning.")

if __name__ == "__main__":
    main()
