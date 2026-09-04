from __future__ import annotations

import argparse
import atexit
import json
import math
import multiprocessing as mp
import os
import time
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

DEV_SEED_START = 4501
DEV_GAMES = 20

PERMANENT_BENCHMARK_FIRST = 6
PERMANENT_BENCHMARK_LAST = 20

CHAMPION_LABEL = "champion_v8_8_150k_norm"
CHAMPION_PATH = "models/v8_8_jax_vectorized_td_150k.pt"
NORMALIZED_GATE = 0.600

V886_GLOB = (
    "models/"
    "v8_8_6_affinity_sharedweight_cuda_graph_td*.pt"
)

DEFAULT_CACHE = "data/v8_8_6_checkpoint_dev_4501_4520.json"
DEFAULT_SELECTION = "data/v8_8_6_dev_selection_4501_4520.json"


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


def _checkpoint_brief(path: Path):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"Checkpoint has no model_state_dict: {path}"
        )

    env_steps = int(ckpt.get("env_steps", -1))
    grad_steps = int(ckpt.get("gradient_steps", -1))
    gate = float(
        ckpt.get(
            "normalized_gate",
            ckpt.get("target_gate", NORMALIZED_GATE),
        )
    )
    gate_semantics = str(
        ckpt.get("gate_semantics", "normalized_q_margin")
    )

    if env_steps <= 1_200_000:
        raise RuntimeError(
            f"Unexpected V8.8.6 env_steps={env_steps}: {path}"
        )
    if abs(gate - NORMALIZED_GATE) > 1e-9:
        raise RuntimeError(
            f"V8.8.6 checkpoint gate mismatch "
            f"({gate} != {NORMALIZED_GATE}): {path}"
        )
    if gate_semantics != "normalized_q_margin":
        raise RuntimeError(
            f"V8.8.6 gate semantics mismatch "
            f"({gate_semantics}): {path}"
        )

    return {
        "path": str(path).replace("\\", "/"),
        "env_steps": env_steps,
        "gradient_steps": grad_steps,
        "gate": gate,
        "gate_semantics": gate_semantics,
        "interrupted": "INTERRUPTED" in path.name.upper(),
    }


def discover_v886_checkpoints():
    paths = sorted(Path("models").glob(
        "v8_8_6_affinity_sharedweight_cuda_graph_td*.pt"
    ))
    if not paths:
        raise FileNotFoundError(
            "No V8.8.6 checkpoints found. Expected pattern: "
            + V886_GLOB
        )

    by_env_steps = {}

    for path in paths:
        info = _checkpoint_brief(path)
        env_steps = info["env_steps"]

        old = by_env_steps.get(env_steps)
        if old is None:
            by_env_steps[env_steps] = info
            continue

        # Same training point may exist as both a normal periodic/final
        # checkpoint and an INTERRUPTED checkpoint. Prefer non-interrupted.
        old_score = (
            1 if old["interrupted"] else 0,
            len(old["path"]),
        )
        new_score = (
            1 if info["interrupted"] else 0,
            len(info["path"]),
        )
        if new_score < old_score:
            by_env_steps[env_steps] = info

    infos = [
        by_env_steps[k]
        for k in sorted(by_env_steps)
    ]

    specs = []
    for info in infos:
        env_k = info["env_steps"] // 1000
        label = f"v8_8_6_{env_k}k_norm"
        specs.append(
            (
                label,
                info["path"],
                NORMALIZED_GATE,
                info["env_steps"],
                info["gradient_steps"],
                info["interrupted"],
            )
        )

    return specs


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


def _worker_init(model_specs):
    global _W_MODELS, _W_TEACHER, _W_ADAPTER

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _W_MODELS = {
        label: _load_model(path)
        for label, path, *_ in model_specs
    }
    _W_TEACHER = HeuristicTeacherV2()
    _W_ADAPTER = GymTetrisAdapter()
    atexit.register(_close_worker)


def _height_from_features(features):
    board = np.asarray(
        features[:200],
        dtype=np.float32,
    ).reshape(20, 10)
    rows = np.where(np.any(board != 0.0, axis=1))[0]
    return (
        0
        if rows.size == 0
        else int(20 - rows[0])
    )


@torch.inference_mode()
def _q_values(model, state_features, successors):
    candidates, rewards, scores, ranks = (
        compact_candidate_arrays(successors)
    )

    q = model(
        state=torch.from_numpy(
            np.asarray(
                state_features,
                dtype=np.float32,
            )
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
    model = (
        None
        if policy == "teacher"
        else _W_MODELS[policy]
    )

    state = adapter.reset(seed=seed)
    adapter.raw.gravity_enabled = False
    state_features = encode_state(
        state
    ).astype(np.float32, copy=True)

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
            chosen_index, _ = normalized_margin_choice(
                q,
                float(gate),
            )
            if chosen_index != 0:
                interventions += 1

        chosen = successors[chosen_index]
        result = execute_placement(
            adapter,
            chosen.action,
        )

        state = result["state"]
        pieces += 1
        state_features = np.asarray(
            chosen.next_state_features,
            dtype=np.float32,
        ).copy()

        lines = int(
            result["info"].get("lines_cleared", 0)
        )
        if lines in counts:
            counts[lines] += 1

        max_height = max(
            max_height,
            _height_from_features(state_features),
        )

        if bool(
            result["terminated"]
            or result["truncated"]
        ):
            game_over = True
            break

    total_lines = sum(
        k * v
        for k, v in counts.items()
    )
    value = sum(
        LINE_VALUE[k] * v
        for k, v in counts.items()
    )

    return {
        "seed": seed,
        "policy": policy,
        "pieces": int(pieces),
        "lines": int(total_lines),
        "tetrises": int(counts[4]),
        "value": int(value),
        "interventions": int(interventions),
        "max_height": int(max_height),
        "game_over": bool(game_over),
    }


def _result_key(seed, policy):
    return f"{int(seed)}|{policy}"


def _atomic_save(path, data):
    p = Path(path)
    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = p.with_suffix(
        p.suffix + ".tmp"
    )
    tmp.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def _protect_seeds(seeds):
    bad = [
        s
        for s in seeds
        if PERMANENT_BENCHMARK_FIRST
        <= s
        <= PERMANENT_BENCHMARK_LAST
    ]
    if bad:
        raise RuntimeError(
            "Permanent benchmark seeds 6~20 "
            f"are protected: {bad}"
        )


def _auto_workers():
    logical = os.cpu_count() or 4
    if logical >= 20:
        return 16
    return max(1, logical - 2)


def _summarize(rows):
    total_pieces = sum(
        r["pieces"]
        for r in rows
    )
    total_value = sum(
        r["value"]
        for r in rows
    )
    total_switch = sum(
        r["interventions"]
        for r in rows
    )

    return {
        "pieces": float(
            np.mean([r["pieces"] for r in rows])
        ),
        "lines": float(
            np.mean([r["lines"] for r in rows])
        ),
        "tetrises": float(
            np.mean([r["tetrises"] for r in rows])
        ),
        "value": float(
            np.mean([r["value"] for r in rows])
        ),
        "reward_per_1000": (
            total_value
            / max(total_pieces, 1)
            * 1000.0
        ),
        "avg_height": float(
            np.mean([r["max_height"] for r in rows])
        ),
        "worst_height": int(
            max(r["max_height"] for r in rows)
        ),
        "gameovers": int(
            sum(bool(r["game_over"]) for r in rows)
        ),
        "switch_rate": (
            total_switch
            / max(total_pieces, 1)
            * 100.0
        ),
    }


def _paired_stats(rows_a, rows_b):
    a = {
        r["seed"]: r
        for r in rows_a
    }
    b = {
        r["seed"]: r
        for r in rows_b
    }
    seeds = sorted(set(a) & set(b))

    diffs = np.asarray(
        [
            float(
                a[s]["value"]
                - b[s]["value"]
            )
            for s in seeds
        ],
        dtype=np.float64,
    )

    mean = float(np.mean(diffs))
    wins = int(np.sum(diffs > 0))
    ties = int(np.sum(diffs == 0))
    losses = int(np.sum(diffs < 0))

    if len(diffs) > 1:
        se = float(
            np.std(diffs, ddof=1)
            / math.sqrt(len(diffs))
        )
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


def _print_summary(label, s):
    print(
        f"{label:<25} "
        f"pieces={s['pieces']:7.2f} "
        f"lines={s['lines']:7.2f} "
        f"Tetris={s['tetrises']:6.2f} "
        f"value={s['value']:10.2f} "
        f"R/1000={s['reward_per_1000']:10.2f} "
        f"avgH={s['avg_height']:5.2f} "
        f"worstH={s['worst_height']:>2} "
        f"GO={s['gameovers']} "
        f"switch={s['switch_rate']:5.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "V8.8.6 development-only checkpoint sweep "
            "on fresh seeds 4501~4520. "
            "Automatically discovers actual V8.8.6 periodic/final/"
            "safe-interrupt checkpoints and chooses exactly ONE winner."
        )
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEV_SEED_START,
    )
    parser.add_argument(
        "--games",
        type=int,
        default=DEV_GAMES,
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
    )
    parser.add_argument(
        "--selection",
        default=DEFAULT_SELECTION,
    )
    args = parser.parse_args()

    seeds = list(
        range(
            args.seed_start,
            args.seed_start + args.games,
        )
    )
    _protect_seeds(seeds)

    if not Path(CHAMPION_PATH).exists():
        raise FileNotFoundError(CHAMPION_PATH)

    challengers = discover_v886_checkpoints()

    model_specs = [
        (
            CHAMPION_LABEL,
            CHAMPION_PATH,
            NORMALIZED_GATE,
            150_000,
            -1,
            False,
        )
    ] + challengers

    workers = (
        args.workers
        if args.workers > 0
        else _auto_workers()
    )

    policies = [
        ("teacher", None)
    ] + [
        (label, gate)
        for (
            label,
            _path,
            gate,
            _env_steps,
            _grad_steps,
            _interrupted,
        ) in model_specs
    ]

    model_meta = [
        {
            "label": label,
            "path": path,
            "gate": gate,
            "env_steps": env_steps,
            "gradient_steps": grad_steps,
            "interrupted": interrupted,
        }
        for (
            label,
            path,
            gate,
            env_steps,
            grad_steps,
            interrupted,
        ) in model_specs
    ]

    meta = {
        "purpose": (
            "V8_8_6_DEVELOPMENT_ONLY_CHECKPOINT_SELECTION"
        ),
        "seed_start": args.seed_start,
        "games": args.games,
        "seeds": seeds,
        "max_pieces": args.max_pieces,
        "normalized_gate": NORMALIZED_GATE,
        "champion": CHAMPION_LABEL,
        "models": model_meta,
        "selection_rule": [
            "lowest_gameovers",
            "highest_reward_per_1000",
            "highest_average_value",
        ],
    }

    cache_path = Path(args.cache)

    if cache_path.exists():
        cache = json.loads(
            cache_path.read_text(
                encoding="utf-8"
            )
        )
        if cache.get("meta") != meta:
            raise RuntimeError(
                "Development cache metadata mismatch. "
                "The checkpoint set or run configuration changed. "
                "Use a NEW --cache path rather than mixing results."
            )
        cache.setdefault("results", {})
    else:
        cache = {
            "meta": meta,
            "results": {},
        }

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
        if _result_key(
            task["seed"],
            task["policy"],
        )
        not in cache["results"]
    ]

    print("=" * 88)
    print(
        "V8.8.6 DEVELOPMENT CHECKPOINT SWEEP "
        f"— {seeds[0]}..{seeds[-1]}"
    )
    print("=" * 88)
    print(
        "Purpose: choose exactly ONE V8.8.6 checkpoint; "
        "NOT formal qualification."
    )
    print(
        "Formal Champion: V8.8 150K @ normalized gate 0.600"
    )
    print(
        "V8.8.6 gate: normalized 0.600"
    )
    print("Seeds:", seeds)
    print("Workers:", workers)
    print(
        "Discovered V8.8.6 checkpoints:",
        len(challengers),
    )

    for (
        label,
        path,
        _gate,
        env_steps,
        grad_steps,
        interrupted,
    ) in challengers:
        marker = (
            " [SAFE-INTERRUPT]"
            if interrupted
            else ""
        )
        print(
            f"  {label:<25} "
            f"env={env_steps:>9} "
            f"grad={grad_steps:>7} "
            f"{path}{marker}"
        )

    print(
        "Already cached:",
        len(requested) - len(missing),
    )
    print(
        "Games to simulate:",
        len(missing),
    )
    print(
        "Permanent seeds 6~20: PROTECTED"
    )
    print()

    start = time.perf_counter()

    if missing:
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(model_specs,),
        ) as executor:
            futures = [
                executor.submit(
                    _run_game,
                    task,
                )
                for task in missing
            ]

            done = 0

            for future in as_completed(futures):
                result = future.result()

                cache["results"][
                    _result_key(
                        result["seed"],
                        result["policy"],
                    )
                ] = result

                _atomic_save(
                    args.cache,
                    cache,
                )

                done += 1

                if (
                    done == 1
                    or done
                    % max(
                        1,
                        len(missing) // 20,
                    )
                    == 0
                ):
                    elapsed = (
                        time.perf_counter()
                        - start
                    )
                    print(
                        f"new={done:>4}/{len(missing)} "
                        f"elapsed={elapsed:8.1f}s "
                        f"games/s="
                        f"{done/max(elapsed,1e-9):.3f}"
                    )

    rows = {}

    for policy, _gate in policies:
        rr = [
            cache["results"][
                _result_key(seed, policy)
            ]
            for seed in seeds
        ]
        rr.sort(
            key=lambda x: x["seed"]
        )
        rows[policy] = rr

    summaries = {
        policy: _summarize(rows[policy])
        for policy, _gate in policies
    }

    print()
    print("=" * 88)
    print("DEVELOPMENT SUMMARY")
    print("=" * 88)

    for policy, _gate in policies:
        _print_summary(
            policy,
            summaries[policy],
        )

    challenger_labels = [
        spec[0]
        for spec in challengers
    ]

    print()
    print(
        "PAIRED VALUE VS FORMAL V8.8 150K CHAMPION "
        "(diagnostic only)"
    )

    paired_vs_champion = {}

    for label in challenger_labels:
        paired = _paired_stats(
            rows[label],
            rows[CHAMPION_LABEL],
        )
        paired_vs_champion[label] = paired

        print(
            f"{label:<25} "
            f"mean={paired['mean']:+10.1f} "
            f"95%CI=["
            f"{paired['lo']:+10.1f},"
            f"{paired['hi']:+10.1f}] "
            f"W/T/L="
            f"{paired['wins']}/"
            f"{paired['ties']}/"
            f"{paired['losses']}"
        )

    ranked = sorted(
        challenger_labels,
        key=lambda label: (
            summaries[label]["gameovers"],
            -summaries[label]["reward_per_1000"],
            -summaries[label]["value"],
        ),
    )

    selected_label = ranked[0]
    challenger_by_label = {
        spec[0]: spec
        for spec in challengers
    }
    selected_spec = challenger_by_label[
        selected_label
    ]

    selection_payload = {
        "purpose": (
            "V8_8_6_DEVELOPMENT_SELECTION_HANDOFF"
        ),
        "development_seeds": seeds,
        "development_cache": str(args.cache),
        "selection_rule": [
            "lowest_gameovers",
            "highest_reward_per_1000",
            "highest_average_value",
        ],
        "normalized_gate": NORMALIZED_GATE,
        "formal_champion": {
            "label": CHAMPION_LABEL,
            "path": CHAMPION_PATH,
            "gate": NORMALIZED_GATE,
            "summary": summaries[CHAMPION_LABEL],
        },
        "selected": {
            "label": selected_label,
            "path": selected_spec[1],
            "gate": NORMALIZED_GATE,
            "env_steps": selected_spec[3],
            "gradient_steps": selected_spec[4],
            "interrupted": selected_spec[5],
            "summary": summaries[selected_label],
            "paired_vs_champion": paired_vs_champion[
                selected_label
            ],
        },
        "ranking": [
            {
                "rank": rank,
                "label": label,
                "path": challenger_by_label[label][1],
                "env_steps": challenger_by_label[label][3],
                "gradient_steps": challenger_by_label[label][4],
                "interrupted": challenger_by_label[label][5],
                "summary": summaries[label],
                "paired_vs_champion": paired_vs_champion[label],
            }
            for rank, label in enumerate(
                ranked,
                1,
            )
        ],
        "status": "DEVELOPMENT_CONSUMED",
        "next_predeclared_qualification_seeds": (
            list(range(3301, 3321))
        ),
    }

    _atomic_save(
        args.selection,
        selection_payload,
    )

    print()
    print("=" * 88)
    print("DEVELOPMENT SELECTION")
    print("=" * 88)
    print(
        "Selected V8.8.6 checkpoint:",
        selected_label,
    )
    print(
        "Path:",
        selected_spec[1],
    )
    print(
        "env_steps:",
        selected_spec[3],
    )
    print(
        "gradient_steps:",
        selected_spec[4],
    )
    print(
        "Selection rule: lowest gameovers, "
        "then highest R/1000, "
        "then highest average value."
    )
    print(
        "Selection handoff:",
        args.selection,
    )
    print(
        f"IMPORTANT: {seeds[0]}~{seeds[-1]} are now "
        "DEVELOPMENT-CONSUMED."
    )
    print(
        "Do NOT use these development seeds for "
        "formal qualification or later tuning."
    )
    print(
        "Next: run only the selected checkpoint on "
        "fresh predeclared qualification seeds 3301~3320."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
