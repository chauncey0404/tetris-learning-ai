from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ai.observable_q_network import ObservableSafeQNetwork
from v8_7_scale_invariant_policy import normalized_margin_actions_torch
from v8_8_1_packed_replay import V881PackedReplayBuffer


def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model_state_dict"]


def build_replay(capacity, device, seed):
    replay = V881PackedReplayBuffer(
        capacity=capacity,
        device=device,
        seed=seed,
    )
    with torch.no_grad():
        replay.data.normal_(0.0, 0.25)
        lo, hi = replay.layout.offsets["next_mask"]
        replay.data[:, lo:hi].fill_(1.0)
        lo, hi = replay.layout.offsets["done"]
        replay.data[:, lo:hi].bernoulli_(0.01)
        lo, hi = replay.layout.offsets["reward"]
        replay.data[:, lo:hi].uniform_(-0.05, 1.0)
        lo, hi = replay.layout.offsets["next_rewards"]
        replay.data[:, lo:hi].uniform_(-0.05, 1.0)
        lo, hi = replay.layout.offsets["teacher_score"]
        replay.data[:, lo:hi].normal_(0.0, 500.0)
        lo, hi = replay.layout.offsets["next_teacher_scores"]
        replay.data[:, lo:hi].normal_(0.0, 500.0)
        replay.size = replay.capacity
        replay.position = 0
    return replay


def views_from_packed(replay, packed):
    return replay._views(packed)


def train_from_batch(
    *,
    model,
    target,
    optimizer,
    batch,
    gamma,
    target_gate,
    terminal_penalty,
    set_to_none,
):
    state = batch["state"]
    candidate = batch["candidate"]
    reward = batch["reward"]
    teacher_score = batch["teacher_score"]
    teacher_rank = batch["teacher_rank"]
    done = batch["done"]

    next_state = batch["next_state"]
    next_candidates = batch["next_candidates"]
    next_rewards = batch["next_rewards"]
    next_teacher_scores = batch["next_teacher_scores"]
    next_teacher_ranks = batch["next_teacher_ranks"]
    next_mask = batch["next_mask"]

    current_q = model(
        state=state,
        candidates=candidate.unsqueeze(1),
        rewards=reward.unsqueeze(1),
        teacher_scores=teacher_score.unsqueeze(1),
        teacher_ranks=teacher_rank.unsqueeze(1),
    )[:, 0]

    with torch.no_grad():
        online_next_q = model(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )
        online_next_q = online_next_q.masked_fill(~next_mask, -1e9)

        next_action, _, _, _ = normalized_margin_actions_torch(
            online_next_q,
            next_mask,
            target_gate,
        )

        target_next_q_all = target(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )
        rows = torch.arange(
            online_next_q.shape[0],
            device=online_next_q.device,
        )
        selected_next_q = target_next_q_all[rows, next_action]
        has_next = next_mask.any(dim=1)
        bootstrap = (1.0 - done) * has_next.float() * selected_next_q
        learning_reward = reward - terminal_penalty * done
        td_target = learning_reward + gamma * bootstrap

    loss = F.smooth_l1_loss(current_q, td_target)
    optimizer.zero_grad(set_to_none=set_to_none)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
        foreach=True,
    )
    optimizer.step()
    return loss


class StaticCudaGraphLearner:
    def __init__(
        self,
        *,
        model_state,
        replay,
        batch_size,
        gamma,
        target_gate,
        terminal_penalty,
        lr,
        weight_decay,
    ):
        self.device = replay.device
        self.replay = replay
        self.batch_size = int(batch_size)
        self.gamma = float(gamma)
        self.target_gate = float(target_gate)
        self.terminal_penalty = float(terminal_penalty)

        self.model = ObservableSafeQNetwork().to(self.device)
        self.model.load_state_dict(model_state)
        self.model.train()

        self.target = ObservableSafeQNetwork().to(self.device)
        self.target.load_state_dict(model_state)
        self.target.eval()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            capturable=True,
        )

        self.static_packed = torch.empty(
            (self.batch_size, replay.packed_width),
            dtype=torch.float32,
            device=self.device,
        )
        self.static_batch = views_from_packed(
            replay,
            self.static_packed,
        )

        # Warm up AdamW state and gradient storage on a side stream.
        self._sample_into_static()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(4):
                train_from_batch(
                    model=self.model,
                    target=self.target,
                    optimizer=self.optimizer,
                    batch=self.static_batch,
                    gamma=self.gamma,
                    target_gate=self.target_gate,
                    terminal_penalty=self.terminal_penalty,
                    set_to_none=False,
                )
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()

        # Capture only learner math. Replay sampling remains dynamic so replay
        # freshness and no-replacement semantics are unchanged.
        with torch.cuda.graph(self.graph):
            train_from_batch(
                model=self.model,
                target=self.target,
                optimizer=self.optimizer,
                batch=self.static_batch,
                gamma=self.gamma,
                target_gate=self.target_gate,
                terminal_penalty=self.terminal_penalty,
                set_to_none=False,
            )

    @torch.no_grad()
    def _sample_into_static(self):
        # Preserve V8.8.1 no-replacement sampling.
        idx = torch.randperm(
            self.replay.size,
            device=self.device,
            generator=self.replay.generator,
        )[: self.batch_size]

        try:
            torch.index_select(
                self.replay.data,
                0,
                idx,
                out=self.static_packed,
            )
        except TypeError:
            # Compatibility fallback if this torch build rejects out=.
            self.static_packed.copy_(
                self.replay.data.index_select(0, idx)
            )

    def step(self):
        self._sample_into_static()
        self.graph.replay()


def eager_step(
    *,
    model,
    target,
    optimizer,
    replay,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
):
    batch = replay.sample(batch_size)
    train_from_batch(
        model=model,
        target=target,
        optimizer=optimizer,
        batch=batch,
        gamma=gamma,
        target_gate=target_gate,
        terminal_penalty=terminal_penalty,
        set_to_none=False,
    )


def benchmark_case(fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    cuda_ms = float(start_event.elapsed_time(end_event))

    return {
        "iterations": iterations,
        "wall_ms_per_grad": wall / iterations * 1000.0,
        "cuda_window_ms_per_grad": cuda_ms / iterations,
        "gradients_per_sec": iterations / wall,
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "V8.8.2 experimental CUDA Graph benchmark. "
            "No checkpoint is overwritten or saved."
        )
    )
    ap.add_argument(
        "--checkpoint",
        default="models/v8_8_1_longtraj_gpu_replay_td_200k.pt",
    )
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--replay-size", type=int, default=50000)
    ap.add_argument("--iterations", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--target-gate", type=float, default=0.600)
    ap.add_argument("--terminal-penalty", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--report",
        default="data/v8_8_2_cuda_graph_benchmark.json",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")

    if not Path(args.checkpoint).exists():
        fallback = Path("models/v8_8_jax_vectorized_td_150k.pt")
        if fallback.exists():
            args.checkpoint = str(fallback)
        else:
            raise FileNotFoundError(args.checkpoint)

    print("=" * 80)
    print("V8.8.2 CUDA GRAPH LEARNER BENCHMARK")
    print("=" * 80)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Checkpoint:", args.checkpoint)
    print("Batch:", args.batch_size)
    print()

    state = load_state(args.checkpoint)
    replay = build_replay(
        args.replay_size,
        device,
        seed=882200,
    )

    # Eager baseline. lr=0 keeps weights fixed while retaining optimizer work.
    eager_model = ObservableSafeQNetwork().to(device)
    eager_model.load_state_dict(state)
    eager_model.train()

    eager_target = ObservableSafeQNetwork().to(device)
    eager_target.load_state_dict(state)
    eager_target.eval()

    eager_optimizer = torch.optim.AdamW(
        eager_model.parameters(),
        lr=0.0,
        weight_decay=args.weight_decay,
        capturable=True,
    )

    eager = benchmark_case(
        lambda: eager_step(
            model=eager_model,
            target=eager_target,
            optimizer=eager_optimizer,
            replay=replay,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_gate=args.target_gate,
            terminal_penalty=args.terminal_penalty,
        ),
        iterations=args.iterations,
        warmup=args.warmup,
    )

    print(
        "EAGER      "
        f"wall={eager['wall_ms_per_grad']:.3f}ms "
        f"CUDA-window={eager['cuda_window_ms_per_grad']:.3f}ms "
        f"grad/s={eager['gradients_per_sec']:.2f}"
    )

    del eager_model, eager_target, eager_optimizer
    torch.cuda.empty_cache()

    graph_error = None
    graph_result = None

    try:
        graph = StaticCudaGraphLearner(
            model_state=state,
            replay=replay,
            batch_size=args.batch_size,
            gamma=args.gamma,
            target_gate=args.target_gate,
            terminal_penalty=args.terminal_penalty,
            lr=0.0,
            weight_decay=args.weight_decay,
        )

        graph_result = benchmark_case(
            graph.step,
            iterations=args.iterations,
            warmup=args.warmup,
        )

        print(
            "CUDA GRAPH "
            f"wall={graph_result['wall_ms_per_grad']:.3f}ms "
            f"CUDA-window={graph_result['cuda_window_ms_per_grad']:.3f}ms "
            f"grad/s={graph_result['gradients_per_sec']:.2f}"
        )

    except Exception as exc:
        graph_error = repr(exc)
        print("CUDA GRAPH: UNSUPPORTED/FAILED")
        print(graph_error)

    report = {
        "version": "V8_8_2_CUDA_GRAPH_BENCHMARK",
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch_size": args.batch_size,
        "eager": eager,
        "cuda_graph": graph_result,
        "cuda_graph_error": graph_error,
    }

    if graph_result is not None:
        speedup = (
            graph_result["gradients_per_sec"]
            / max(eager["gradients_per_sec"], 1e-12)
        )
        report["speedup"] = speedup
        print()
        print("CUDA Graph speedup:", f"{speedup:.3f}x")
        if speedup >= 1.05:
            print(
                "RESULT: CUDA Graph is promising enough for production integration."
            )
        else:
            print(
                "RESULT: CUDA Graph does not provide >=5% throughput gain; "
                "do not integrate it merely to raise GPU utilization."
            )
    else:
        print()
        print(
            "RESULT: Keep eager learner. This Windows/PyTorch build did not "
            "accept the graph path."
        )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("Report:", report_path)
    print()
    print("V8.8.2 CUDA GRAPH BENCHMARK: PASS")


if __name__ == "__main__":
    main()
