from __future__ import annotations

import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np

from v8_8_jax_vector_backend import (
    reset_batch,
    all_candidates_batch,
    N_CANDIDATES,
)


def block_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def bench(batch_size: int, repeats: int):
    keys = jax.random.split(jax.random.PRNGKey(12345 + batch_size), batch_size)
    states = reset_batch(keys)
    block_tree(states)

    # Compile.
    out = all_candidates_batch(states)
    block_tree(out)

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = all_candidates_batch(states)
        block_tree(out)
        samples.append(time.perf_counter() - t0)

    sec = float(np.median(samples))
    decisions_s = batch_size / sec
    candidates_s = batch_size * N_CANDIDATES / sec
    return sec, decisions_s, candidates_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batches",
        default="32,64,128,256,512",
        help="Comma-separated parallel environment counts.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-tps", type=float, default=370.82)
    args = parser.parse_args()

    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]

    print("=" * 80)
    print("V8.8 JAX VECTORIZED PLACEMENT BENCHMARK")
    print("=" * 80)
    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())
    print("Fixed candidates/state:", N_CANDIDATES)
    print("Old online rollout baseline:", args.baseline_tps, "transitions/s")
    print()
    print("NOTE: first compile is excluded from timing.")
    print()

    best = None
    for b in batches:
        try:
            sec, dps, cps = bench(b, args.repeats)
        except Exception as exc:
            print(f"batch={b:5d} FAILED: {exc}")
            continue

        speedup = dps / max(args.baseline_tps, 1e-9)
        print(
            f"batch={b:5d} "
            f"step={sec*1000:9.3f} ms "
            f"decisions/s={dps:10.1f} "
            f"candidate-sims/s={cps:12.1f} "
            f"vs-old-rollout={speedup:7.2f}x"
        )

        if best is None or dps > best[1]:
            best = (b, dps, cps, sec)

    if best is None:
        raise SystemExit("No benchmark batch completed.")

    print()
    print(
        "BEST:",
        f"batch={best[0]}",
        f"decisions/s={best[1]:.1f}",
        f"candidate-sims/s={best[2]:.1f}",
    )
    print()
    print(
        "This benchmark measures the JAX placement-generation core only. "
        "Teacher ranking and PyTorch learner integration are separate stages."
    )


if __name__ == "__main__":
    main()
