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

QUAL_SEED_START = 3301
QUAL_GAMES = 20

PERMANENT_BENCHMARK_FIRST = 6
PERMANENT_BENCHMARK_LAST = 20

NORMALIZED_GATE = 0.600

CHAMPION_LABEL = "champion_v8_8_150k_norm"
CHAMPION_PATH = "models/v8_8_jax_vectorized_td_150k.pt"

DEFAULT_SELECTION = "data/v8_8_6_dev_selection_4501_4520.json"
DEFAULT_CACHE = "data/v8_8_6_qualification_3301_3320.json"
DEFAULT_RESULT = "data/v8_8_6_qualification_result_3301_3320.json"


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
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    model = ObservableSafeQNetwork().cpu()
    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    model.eval()
    return model


def _checkpoint_guard(path):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"No model_state_dict in checkpoint: {path}"
        )

    gate = float(
        ckpt.get(
            "normalized_gate",
            ckpt.get("target_gate", NORMALIZED_GATE),
        )
    )
    semantics = str(
        ckpt.get(
            "gate_semantics",
            "normalized_q_margin",
        )
    )

    if abs(gate - NORMALIZED_GATE) > 1e-9:
        raise RuntimeError(
            f"Normalized gate mismatch in {path}: "
            f"{gate} != {NORMALIZED_GATE}"
        )

    if semantics != "normalized_q_margin":
        raise RuntimeError(
            f"Gate semantics mismatch in {path}: "
            f"{semantics}"
        )

    return {
        "env_steps": int(
            ckpt.get("env_steps", -1)
        ),
        "gradient_steps": int(
            ckpt.get("gradient_steps", -1)
        ),
        "gate": gate,
        "gate_semantics": semantics,
    }


def _worker_init(model_specs):
    global _W_MODELS, _W_TEACHER, _W_ADAPTER

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _W_MODELS = {
        label: _load_model(path)
        for label, path, _gate in model_specs
    }
    _W_TEACHER = HeuristicTeacherV2()
    _W_ADAPTER = GymTetrisAdapter()
    atexit.register(_close_worker)


def _height(features):
    board = np.asarray(
        features[:200],
        dtype=np.float32,
    ).reshape(20, 10)
    rows = np.where(
        np.any(board != 0.0, axis=1)
    )[0]
    return (
        0
        if rows.size == 0
        else int(20 - rows[0])
    )


@torch.inference_mode()
def _q_values(
    model,
    state_features,
    successors,
):
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
    counts = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
    }
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
            q = _q_values(
                model,
                state_features,
                successors,
            )
            chosen_index, _ = (
                normalized_margin_choice(
                    q,
                    float(gate),
                )
            )
            if chosen_index != 0:
                interventions += 1

        chosen = successors[chosen_index]
        result = execute_placement(
            adapter,
            chosen.action,
        )

        state = result["state"]
        state_features = np.asarray(
            chosen.next_state_features,
            dtype=np.float32,
        ).copy()
        pieces += 1

        lines = int(
            result["info"].get(
                "lines_cleared",
                0,
            )
        )
        if lines in counts:
            counts[lines] += 1

        max_height = max(
            max_height,
            _height(state_features),
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


def _key(seed, policy):
    return f"{int(seed)}|{policy}"


def _atomic_save(path, payload):
    p = Path(path)
    p.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = p.with_suffix(
        p.suffix + ".tmp"
    )
    tmp.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, p)


def _protect_seeds(seeds):
    bad = [
        seed
        for seed in seeds
        if PERMANENT_BENCHMARK_FIRST
        <= seed
        <= PERMANENT_BENCHMARK_LAST
    ]
    if bad:
        raise RuntimeError(
            "Permanent benchmark seeds 6~20 "
            f"are protected: {bad}"
        )


def _summary(rows):
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
            np.mean(
                [r["pieces"] for r in rows]
            )
        ),
        "lines": float(
            np.mean(
                [r["lines"] for r in rows]
            )
        ),
        "tetrises": float(
            np.mean(
                [r["tetrises"] for r in rows]
            )
        ),
        "value": float(
            np.mean(
                [r["value"] for r in rows]
            )
        ),
        "r1000": (
            total_value
            / max(total_pieces, 1)
            * 1000.0
        ),
        "avgH": float(
            np.mean(
                [r["max_height"] for r in rows]
            )
        ),
        "worstH": int(
            max(
                r["max_height"]
                for r in rows
            )
        ),
        "GO": int(
            sum(
                bool(r["game_over"])
                for r in rows
            )
        ),
        "switch": (
            total_switch
            / max(total_pieces, 1)
            * 100.0
        ),
    }


def _paired(a_rows, b_rows):
    a = {
        r["seed"]: r
        for r in a_rows
    }
    b = {
        r["seed"]: r
        for r in b_rows
    }

    seeds = sorted(
        set(a) & set(b)
    )

    d = np.asarray(
        [
            float(
                a[s]["value"]
                - b[s]["value"]
            )
            for s in seeds
        ],
        dtype=np.float64,
    )

    mean = float(np.mean(d))

    if len(d) > 1:
        se = float(
            np.std(d, ddof=1)
            / math.sqrt(len(d))
        )
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se
    else:
        lo = hi = mean

    return {
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "wins": int(np.sum(d > 0)),
        "ties": int(np.sum(d == 0)),
        "losses": int(np.sum(d < 0)),
    }


def _print_summary(label, s):
    print(
        f"{label:<25} "
        f"pieces={s['pieces']:7.2f} "
        f"lines={s['lines']:7.2f} "
        f"Tetris={s['tetrises']:6.2f} "
        f"value={s['value']:10.2f} "
        f"R/1000={s['r1000']:10.2f} "
        f"avgH={s['avgH']:5.2f} "
        f"worstH={s['worstH']:>2} "
        f"GO={s['GO']} "
        f"switch={s['switch']:5.2f}%"
    )


def _load_selection(path):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Development selection handoff not found: {path}\n"
            "Run evaluate_v8_8_6_checkpoints_dev_4501_4520.py first."
        )

    payload = json.loads(
        p.read_text(encoding="utf-8")
    )

    if payload.get("purpose") != (
        "V8_8_6_DEVELOPMENT_SELECTION_HANDOFF"
    ):
        raise RuntimeError(
            "Selection file purpose mismatch."
        )

    expected_dev = list(
        range(4501, 4521)
    )

    if payload.get("development_seeds") != expected_dev:
        raise RuntimeError(
            "Development selection did not use "
            "the predeclared 4501~4520 block."
        )

    if payload.get("status") != "DEVELOPMENT_CONSUMED":
        raise RuntimeError(
            "Development selection status mismatch."
        )

    selected = payload.get("selected")
    if not selected:
        raise RuntimeError(
            "No selected challenger in selection handoff."
        )

    if abs(
        float(
            selected.get("gate", -1.0)
        )
        - NORMALIZED_GATE
    ) > 1e-9:
        raise RuntimeError(
            "Selected challenger gate mismatch."
        )

    return payload, selected


def main():
    parser = argparse.ArgumentParser(
        description=(
            "One-time fresh formal qualification of the "
            "ONE V8.8.6 checkpoint chosen on development "
            "seeds 4501~4520, against the formal V8.8 150K Champion, "
            "using untouched seeds 3301~3320."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--selection",
        default=DEFAULT_SELECTION,
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE,
    )
    parser.add_argument(
        "--result",
        default=DEFAULT_RESULT,
    )
    args = parser.parse_args()

    seeds = list(
        range(
            QUAL_SEED_START,
            QUAL_SEED_START + QUAL_GAMES,
        )
    )
    _protect_seeds(seeds)

    selection_payload, selected = (
        _load_selection(args.selection)
    )

    challenger_label = str(
        selected["label"]
    )
    challenger_path = str(
        selected["path"]
    )

    for path in (
        CHAMPION_PATH,
        challenger_path,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    challenger_meta = _checkpoint_guard(
        challenger_path
    )

    expected_env_steps = int(
        selected.get("env_steps", -1)
    )

    if (
        expected_env_steps > 0
        and challenger_meta["env_steps"]
        != expected_env_steps
    ):
        raise RuntimeError(
            "Selected checkpoint env_steps changed "
            "since development selection."
        )

    model_specs = [
        (
            CHAMPION_LABEL,
            CHAMPION_PATH,
            NORMALIZED_GATE,
        ),
        (
            challenger_label,
            challenger_path,
            NORMALIZED_GATE,
        ),
    ]

    policies = [
        ("teacher", None),
        (
            CHAMPION_LABEL,
            NORMALIZED_GATE,
        ),
        (
            challenger_label,
            NORMALIZED_GATE,
        ),
    ]

    meta = {
        "purpose": (
            "ONE_TIME_FRESH_FORMAL_QUALIFICATION_V8_8_6"
        ),
        "seeds": seeds,
        "max_pieces": args.max_pieces,
        "normalized_gate": NORMALIZED_GATE,
        "champion": {
            "label": CHAMPION_LABEL,
            "path": CHAMPION_PATH,
        },
        "challenger": {
            "label": challenger_label,
            "path": challenger_path,
            "env_steps": challenger_meta["env_steps"],
            "gradient_steps": challenger_meta[
                "gradient_steps"
            ],
        },
        "selection_handoff": str(args.selection),
        "development_seeds": list(
            range(4501, 4521)
        ),
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
                "Qualification cache metadata mismatch. "
                "Do not mix qualification challengers or seeds. "
                "Use a new --cache path only if this run has never started."
            )
    else:
        cache = {
            "meta": meta,
            "results": {},
        }

    tasks = [
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
        for task in tasks
        if _key(
            task["seed"],
            task["policy"],
        )
        not in cache["results"]
    ]

    print("=" * 88)
    print(
        "V8.8.6 FRESH FORMAL QUALIFICATION "
        "— 3301..3320"
    )
    print("=" * 88)
    print(
        "Formal Champion : "
        "V8.8 150K @ NORMALIZED gate 0.600"
    )
    print(
        "Challenger      : "
        f"{challenger_label} @ NORMALIZED gate 0.600"
    )
    print(
        "Challenger path :",
        challenger_path,
    )
    print(
        "Challenger env  :",
        challenger_meta["env_steps"],
    )
    print(
        "Development used: 4501..4520 "
        "(selection only; now consumed)"
    )
    print(
        "Qualification   : 3301..3320 "
        "(fresh; one-time formal block)"
    )
    print(
        "Teacher         : reference only"
    )
    print(
        "Workers         :",
        args.workers,
    )
    print(
        "Already cached  :",
        len(tasks) - len(missing),
    )
    print(
        "Games to run    :",
        len(missing),
    )
    print(
        "Permanent seeds 6~20: PROTECTED"
    )
    print()

    if missing:
        ctx = mp.get_context("spawn")
        start = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=args.workers,
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

            for i, future in enumerate(
                as_completed(futures),
                1,
            ):
                result = future.result()

                cache["results"][
                    _key(
                        result["seed"],
                        result["policy"],
                    )
                ] = result

                _atomic_save(
                    args.cache,
                    cache,
                )

                if (
                    i == 1
                    or i
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
                        f"new={i:>3}/{len(missing)} "
                        f"elapsed={elapsed:8.1f}s "
                        f"games/s="
                        f"{i/max(elapsed,1e-9):.3f}"
                    )

    rows = {}

    for policy, _gate in policies:
        rows[policy] = [
            cache["results"][
                _key(seed, policy)
            ]
            for seed in seeds
        ]

    summaries = {
        policy: _summary(rows[policy])
        for policy, _gate in policies
    }

    print()
    print("=" * 88)
    print("QUALIFICATION SUMMARY")
    print("=" * 88)

    _print_summary(
        "teacher",
        summaries["teacher"],
    )
    _print_summary(
        CHAMPION_LABEL,
        summaries[CHAMPION_LABEL],
    )
    _print_summary(
        challenger_label,
        summaries[challenger_label],
    )

    paired = _paired(
        rows[challenger_label],
        rows[CHAMPION_LABEL],
    )

    print()
    print(
        "Paired value challenger-champion: "
        f"mean={paired['mean']:+.1f} "
        f"95%CI=["
        f"{paired['lo']:+.1f},"
        f"{paired['hi']:+.1f}] "
        f"W/T/L="
        f"{paired['wins']}/"
        f"{paired['ties']}/"
        f"{paired['losses']}"
    )
    print(
        "CI and W/T/L are diagnostics only."
    )

    champion = summaries[CHAMPION_LABEL]
    challenger = summaries[challenger_label]

    gates = [
        {
            "name": "Safety GO",
            "passed": (
                challenger["GO"]
                <= champion["GO"]
            ),
            "detail": (
                f"{challenger['GO']} "
                f"<= {champion['GO']}"
            ),
        },
        {
            "name": "Survival pieces",
            "passed": (
                challenger["pieces"]
                >= champion["pieces"]
            ),
            "detail": (
                f"{challenger['pieces']:.2f} "
                f">= {champion['pieces']:.2f}"
            ),
        },
        {
            "name": "R/1000",
            "passed": (
                challenger["r1000"]
                > champion["r1000"]
            ),
            "detail": (
                f"{challenger['r1000']:.2f} "
                f"> {champion['r1000']:.2f}"
            ),
        },
        {
            "name": "Average value",
            "passed": (
                challenger["value"]
                > champion["value"]
            ),
            "detail": (
                f"{challenger['value']:.2f} "
                f"> {champion['value']:.2f}"
            ),
        },
    ]

    passed_all = all(
        gate["passed"]
        for gate in gates
    )

    print()
    print("=" * 88)
    print("PROMOTION GATES")
    print("=" * 88)

    for gate in gates:
        print(
            f"{'PASS' if gate['passed'] else 'FAIL':<4} "
            f"{gate['name']}: "
            f"{gate['detail']}"
        )

    result_payload = {
        "purpose": (
            "V8_8_6_FRESH_FORMAL_QUALIFICATION_RESULT"
        ),
        "qualification_seeds": seeds,
        "qualification_status": (
            "QUALIFICATION_CONSUMED"
        ),
        "champion": {
            "label": CHAMPION_LABEL,
            "path": CHAMPION_PATH,
            "gate": NORMALIZED_GATE,
            "summary": champion,
        },
        "challenger": {
            "label": challenger_label,
            "path": challenger_path,
            "gate": NORMALIZED_GATE,
            "env_steps": challenger_meta[
                "env_steps"
            ],
            "gradient_steps": challenger_meta[
                "gradient_steps"
            ],
            "summary": challenger,
        },
        "paired_diagnostics": paired,
        "promotion_gates": gates,
        "promotion_pass": passed_all,
        "formal_result": (
            "PROMOTE_CHALLENGER"
            if passed_all
            else "RETAIN_V8_8_150K_CHAMPION"
        ),
    }

    _atomic_save(
        args.result,
        result_payload,
    )

    print()
    if passed_all:
        print(
            "FINAL RESULT: PROMOTION PASS"
        )
        print(
            f"{challenger_label} qualifies to replace "
            "V8.8 150K as formal Champion."
        )
    else:
        print(
            "FINAL RESULT: PROMOTION FAIL"
        )
        print(
            "Retain V8.8 150K as formal Champion."
        )

    print(
        "Qualification result:",
        args.result,
    )
    print(
        "Seeds 3301~3320 are now "
        "QUALIFICATION-CONSUMED. "
        "Do not use them for tuning."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
