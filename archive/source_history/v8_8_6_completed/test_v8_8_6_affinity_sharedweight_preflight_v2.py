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
    ready_queue,
    result_queue,
):
    actual = _set_process_affinity(cpus) if cpus else []

    model = ObservableSafeQNetwork().cpu()

    # Initial forced read. This MUST observe version=1 before the parent
    # publishes version=2. The ready_queue handshake below removes the race
    # that existed in the original preflight.
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

    ready_queue.put(
        {
            "worker": worker_id,
            "cpus": actual,
            "initial_version": int(seen),
            "initial_value": float(first),
        }
    )

    deadline = time.time() + 30.0
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
        time.sleep(0.005)

    second = float(
        next(model.parameters()).detach().view(-1)[0]
    )

    result_queue.put(
        {
            "worker": worker_id,
            "cpus": actual,
            "version": int(seen),
            "first": float(first),
            "second": float(second),
        }
    )


def main():
    producer_count = 3
    plan = _build_affinity_plan(
        producer_count,
        2,
    )

    print("=" * 80)
    print("V8.8.6 AFFINITY + SHARED-WEIGHT PREFLIGHT V2")
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
    ready_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for i in range(producer_count):
        process = ctx.Process(
            target=worker,
            args=(
                i,
                plan["producer_cpus"][i],
                bank,
                version,
                lock,
                ready_queue,
                result_queue,
            ),
            name=f"v8_8_6_preflight_worker_{i}",
        )
        process.start()
        workers.append(process)

    # Critical handshake:
    # wait until ALL workers have loaded version=1 before changing the bank.
    ready = [
        ready_queue.get(timeout=60.0)
        for _ in workers
    ]
    ready.sort(key=lambda x: x["worker"])

    for item in ready:
        print(
            f"worker {item['worker']} initial "
            f"cpus={item['cpus']} "
            f"version={item['initial_version']} "
            f"value={item['initial_value']:+.6f}"
        )
        if item["initial_version"] != 1:
            raise AssertionError(
                "Preflight synchronization failed: worker did not observe "
                f"initial version=1 (worker={item['worker']}, "
                f"version={item['initial_version']})."
            )

    print()
    print("All workers initialized at version 1: PASS")

    # Now mutate exactly one known scalar by +1 and publish version=2.
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
        result_queue.get(timeout=60.0)
        for _ in workers
    ]

    for process in workers:
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            raise RuntimeError(
                f"worker {process.name} failed to exit."
            )
        if process.exitcode != 0:
            raise RuntimeError(
                f"worker {process.name} exitcode={process.exitcode}"
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
                "Shared weight version was not observed: "
                f"worker={result['worker']} "
                f"version={result['version']}"
            )

        if abs(delta - 1.0) > 1e-5:
            raise AssertionError(
                "Shared weight contents did not update exactly: "
                f"worker={result['worker']} delta={delta:+.9f}"
            )

    print()
    print("Affinity plan              : PASS")
    print("Initial-version handshake  : PASS")
    print("Shared-memory weight bank  : PASS")
    print("Cross-process weight update: PASS")
    print("V8.8.6 PREFLIGHT V2: PASS")


if __name__ == "__main__":
    mp.freeze_support()
    main()
