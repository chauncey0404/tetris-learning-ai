from __future__ import annotations

import argparse
import time

import torch

from ai.observable_q_network import ObservableSafeQNetwork
from v8_8_1_packed_replay import V881PackedReplayBuffer
from v8_8_1_train_common import train_batch_device_replay


def run_one(checkpoint, batch_size, replay_size, warmup, iters, device):
    model = ObservableSafeQNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    target = ObservableSafeQNetwork().to(device)
    target.load_state_dict(
        checkpoint.get("target_model_state_dict", checkpoint["model_state_dict"])
    )
    target.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    replay = V881PackedReplayBuffer(
        capacity=replay_size,
        device=device,
        seed=88123 + batch_size,
    )

    # Synthetic resident data. This benchmark is for replay gather + learner
    # throughput, not environment semantics.
    with torch.no_grad():
        replay.data.normal_(0.0, 0.5)
        lo, hi = replay.layout.offsets["next_mask"]
        replay.data[:, lo:hi].fill_(1.0)
        lo, hi = replay.layout.offsets["done"]
        replay.data[:, lo:hi].zero_()
        replay.size = replay.capacity
        replay.position = 0

    for _ in range(warmup):
        train_batch_device_replay(
            model=model,
            target_model=target,
            optimizer=optimizer,
            replay=replay,
            batch_size=batch_size,
            gamma=0.99,
            target_gate=0.600,
            terminal_penalty=1.0,
            collect_metrics=False,
        )

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        train_batch_device_replay(
            model=model,
            target_model=target,
            optimizer=optimizer,
            replay=replay,
            batch_size=batch_size,
            gamma=0.99,
            target_gate=0.600,
            terminal_penalty=1.0,
            collect_metrics=False,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "batch": batch_size,
        "step_ms": elapsed / iters * 1000.0,
        "grad_s": iters / elapsed,
        "samples_s": iters * batch_size / elapsed,
        "replay_mib": replay.nbytes / (1024.0 ** 2),
        "peak_mib": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/v8_8_jax_vectorized_td_150k.pt")
    ap.add_argument("--batches", default="4096,8192")
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    print("=" * 80)
    print("V8.8.1 PACKED GPU REPLAY + LEARNER BENCHMARK")
    print("=" * 80)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)
    print("Synthetic replay rows:", args.replay_size)
    print()

    results = []
    for token in args.batches.split(","):
        batch = int(token.strip())
        if batch > args.replay_size:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        result = run_one(
            checkpoint,
            batch,
            args.replay_size,
            args.warmup,
            args.iters,
            device,
        )
        results.append(result)
        print(
            f"batch={batch:>5} step={result['step_ms']:7.3f}ms "
            f"grad/s={result['grad_s']:7.1f} samples/s={result['samples_s']:10.0f} "
            f"replay={result['replay_mib']:6.1f}MiB peak={result['peak_mib']:7.1f}MiB"
        )

    best = max(results, key=lambda x: x["samples_s"])
    print()
    print("BEST SAMPLES/S:", best["batch"])
    print("NOTE: the production trainer still applies the 10K-window freshness guard.")


if __name__ == "__main__":
    main()
