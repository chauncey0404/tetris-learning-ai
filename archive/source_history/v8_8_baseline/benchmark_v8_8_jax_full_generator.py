
from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from v8_8_jax_vector_backend import reset_batch
from v8_8_jax_teacher import topk_batch


def block_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def bench(batch_size: int, repeats: int):
    keys = jax.random.split(
        jax.random.PRNGKey(880000 + batch_size),
        batch_size,
    )
    states = reset_batch(keys)
    block_tree(states)

    # First call compiles; excluded from timing.
    bundle = topk_batch(states)
    block_tree(bundle)

    samples = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        bundle = topk_batch(states)
        block_tree(bundle)
        samples.append(time.perf_counter() - t0)

    sec = float(np.median(samples))

    return {
        "batch": int(batch_size),
        "step_ms": sec * 1000.0,
        "states_per_sec": batch_size / sec,
        "candidate_sims_per_sec": batch_size * 80 / sec,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the production V8.8 JAX generator core: "
            "80 placement simulations + HeuristicTeacherV2 scoring + reachable top-4."
        )
    )
    parser.add_argument(
        "--batches",
        default="32,64,128,256,512",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--old-baseline",
        type=float,
        default=370.82,
        help="Previous V8.7 online rollout transitions/s.",
    )
    args = parser.parse_args()

    batches = [
        int(x.strip())
        for x in args.batches.split(",")
        if x.strip()
    ]

    print("=" * 80)
    print("V8.8 FULL JAX GENERATOR CORE BENCHMARK")
    print("=" * 80)
    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())
    print("Per state: 80 fixed candidate slots + TeacherV2.1 + top4")
    print("Old V8.7 rollout baseline:", args.old_baseline, "transitions/s")
    print()

    results = []

    for b in batches:
        try:
            result = bench(b, args.repeats)
        except Exception as exc:
            print(f"batch={b:5d} FAILED: {exc}")
            continue

        result["speedup_vs_old"] = (
            result["states_per_sec"]
            / max(args.old_baseline, 1e-9)
        )
        results.append(result)

        print(
            f"batch={b:5d} "
            f"step={result['step_ms']:9.3f} ms "
            f"states/s={result['states_per_sec']:10.1f} "
            f"candidate-sims/s={result['candidate_sims_per_sec']:12.1f} "
            f"vs-old={result['speedup_vs_old']:7.2f}x"
        )

    if not results:
        raise SystemExit("No benchmark batch completed.")

    best = max(
        results,
        key=lambda x: x["states_per_sec"],
    )

    print()
    print(
        "BEST:",
        f"batch={best['batch']}",
        f"states/s={best['states_per_sec']:.1f}",
        f"candidate-sims/s={best['candidate_sims_per_sec']:.1f}",
        f"vs-old={best['speedup_vs_old']:.2f}x",
    )

    print()
    print(
        "Choose --vector-envs near the best stable batch. "
        "256 is the V8.8 trainer default until this benchmark says otherwise."
    )


if __name__ == "__main__":
    main()
