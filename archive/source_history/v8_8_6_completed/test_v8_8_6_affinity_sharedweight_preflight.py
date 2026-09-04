from __future__ import annotations

import multiprocessing as mp
import os
import time

import torch

from ai.observable_q_network import ObservableSafeQNetwork
from train_v8_8_6_affinity_sharedweight_1200k_to_31200k import (
    _build_affinity_plan,
    _create_shared_weight_bank,
    _poll_shared_weights,
    _publish_shared_weights,
    _set_process_affinity,
)


def worker(
    worker_id,
    cpus,
    bank,
    version,
    lock,
    result_queue,
):
    actual = _set_process_affinity(cpus) if cpus else []

    model = ObservableSafeQNetwork().cpu()
    seen = _poll_shared_weights(
        model,
        bank,
        version,
        lock,
        -1,
        force=True,
    )

    first = float(
        next(model.parameters()).detach().view(-1)[0]
    )

    deadline = time.time() + 15.0
    while time.time() < deadline:
        now = _poll_shared_weights(
            model,
            bank,
            version,
            lock,
            seen,
        )
        if now != seen:
            seen = now
            break
        time.sleep(0.01)

    second = float(
        next(model.parameters()).detach().view(-1)[0]
    )

    result_queue.put(
        {
            "worker": worker_id,
            "cpus": actual,
            "version": seen,
            "first": first,
            "second": second,
        }
    )


def main():
    plan = _build_affinity_plan(3, 2)

    print("=" * 80)
    print("V8.8.6 AFFINITY + SHARED-WEIGHT PREFLIGHT")
    print("=" * 80)
    print("OS:", os.name)
    print("logical CPUs:", os.cpu_count())
    print("affinity source:", plan["source"])
    for i, cpus in enumerate(plan["producer_cpus"]):
        print(f"producer {i}: {cpus}")
    print("main/learner:", plan["main_cpus"])
    print()

    model = ObservableSafeQNetwork().cpu()
    bank = _create_shared_weight_bank(model)

    ctx = mp.get_context("spawn")
    version = ctx.Value("q", 1, lock=False)
    lock = ctx.Lock()
    result_queue = ctx.Queue()

    workers = []
    for i in range(3):
        process = ctx.Process(
            target=worker,
            args=(
                i,
                plan["producer_cpus"][i],
                bank,
                version,
                lock,
                result_queue,
            ),
        )
        process.start()
        workers.append(process)

    time.sleep(0.5)

    with torch.no_grad():
        next(model.parameters()).view(-1)[0].add_(1.0)

    _publish_shared_weights(
        model,
        bank,
        version,
        lock,
        2,
    )

    results = [
        result_queue.get(timeout=30.0)
        for _ in workers
    ]

    for process in workers:
        process.join(timeout=5.0)
        if process.exitcode != 0:
            raise RuntimeError(
                f"worker {process.pid} exitcode={process.exitcode}"
            )

    results.sort(key=lambda x: x["worker"])

    for result in results:
        delta = result["second"] - result["first"]
        print(
            f"worker {result['worker']} "
            f"cpus={result['cpus']} "
            f"version={result['version']} "
            f"delta={delta:+.6f}"
        )
        if result["version"] != 2:
            raise AssertionError(
                "Shared weight version was not observed."
            )
        if abs(delta - 1.0) > 1e-5:
            raise AssertionError(
                "Shared weight contents did not update exactly."
            )

    print()
    print("Affinity plan              : PASS")
    print("Shared-memory weight bank  : PASS")
    print("Cross-process weight update: PASS")
    print("V8.8.6 PREFLIGHT: PASS")


if __name__ == "__main__":
    mp.freeze_support()
    main()
