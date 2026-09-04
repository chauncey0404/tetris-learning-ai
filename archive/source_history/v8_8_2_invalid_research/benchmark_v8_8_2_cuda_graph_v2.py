from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from ai.observable_q_network import ObservableSafeQNetwork
from v8_8_1_packed_replay import V881PackedReplayBuffer
from v8_8_2_graph_safe_policy import (
    normalized_margin_actions_top4_graphsafe,
)


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


def graphsafe_clip_grad_norm_(
    parameters,
    max_norm_tensor,
    eps_tensor,
):
    grads = [
        p.grad
        for p in parameters
        if p.grad is not None
    ]
    if not grads:
        return

    norms = torch._foreach_norm(grads, 2.0)
    total_norm = torch.linalg.vector_norm(
        torch.stack(norms),
        2.0,
    )
    coef = max_norm_tensor / (total_norm + eps_tensor)
    coef = torch.clamp(coef, max=1.0)
    torch._foreach_mul_(grads, coef)


def train_math(
    *,
    model,
    target,
    optimizer,
    batch,
    rows,
    gamma_tensor,
    gate,
    terminal_penalty_tensor,
    clip_max_tensor,
    clip_eps_tensor,
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
        online_next_q = online_next_q.masked_fill(
            ~next_mask,
            float("-inf"),
        )

        next_action, _, _, _ = (
            normalized_margin_actions_top4_graphsafe(
                online_next_q,
                next_mask,
                gate,
            )
        )

        target_next_q_all = target(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )

        selected_next_q = target_next_q_all[
            rows,
            next_action,
        ]

        has_next = next_mask.any(dim=1)
        bootstrap = (
            (1.0 - done)
            * has_next.float()
            * selected_next_q
        )
        learning_reward = (
            reward
            - terminal_penalty_tensor * done
        )
        td_target = (
            learning_reward
            + gamma_tensor * bootstrap
        )

    loss = F.smooth_l1_loss(
        current_q,
        td_target,
    )

    optimizer.zero_grad(set_to_none=False)
    loss.backward()

    graphsafe_clip_grad_norm_(
        model.parameters(),
        clip_max_tensor,
        clip_eps_tensor,
    )

    optimizer.step()


class StaticCudaGraphLearner:
    def __init__(
        self,
        *,
        model_state,
        replay,
        batch_size,
        gamma,
        gate,
        terminal_penalty,
        lr,
        weight_decay,
    ):
        self.device = replay.device
        self.replay = replay
        self.batch_size = int(batch_size)
        self.gate = float(gate)

        self.model = ObservableSafeQNetwork().to(
            self.device
        )
        self.model.load_state_dict(model_state)
        self.model.train()

        self.target = ObservableSafeQNetwork().to(
            self.device
        )
        self.target.load_state_dict(model_state)
        self.target.eval()

        self.lr_tensor = torch.tensor(
            float(lr),
            dtype=torch.float32,
            device=self.device,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr_tensor,
            weight_decay=weight_decay,
            capturable=True,
            foreach=False,
        )

        self.static_packed = torch.empty(
            (
                self.batch_size,
                replay.packed_width,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.static_batch = replay._views(
            self.static_packed
        )

        self.rows = torch.arange(
            self.batch_size,
            device=self.device,
        )

        self.gamma_tensor = torch.tensor(
            float(gamma),
            dtype=torch.float32,
            device=self.device,
        )
        self.terminal_penalty_tensor = torch.tensor(
            float(terminal_penalty),
            dtype=torch.float32,
            device=self.device,
        )
        self.clip_max_tensor = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.clip_eps_tensor = torch.tensor(
            1.0e-6,
            dtype=torch.float32,
            device=self.device,
        )

        self._sample_into_static()

        for _ in range(5):
            train_math(
                model=self.model,
                target=self.target,
                optimizer=self.optimizer,
                batch=self.static_batch,
                rows=self.rows,
                gamma_tensor=self.gamma_tensor,
                gate=self.gate,
                terminal_penalty_tensor=(
                    self.terminal_penalty_tensor
                ),
                clip_max_tensor=(
                    self.clip_max_tensor
                ),
                clip_eps_tensor=(
                    self.clip_eps_tensor
                ),
            )

        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            train_math(
                model=self.model,
                target=self.target,
                optimizer=self.optimizer,
                batch=self.static_batch,
                rows=self.rows,
                gamma_tensor=self.gamma_tensor,
                gate=self.gate,
                terminal_penalty_tensor=(
                    self.terminal_penalty_tensor
                ),
                clip_max_tensor=(
                    self.clip_max_tensor
                ),
                clip_eps_tensor=(
                    self.clip_eps_tensor
                ),
            )

    @torch.no_grad()
    def _sample_into_static(self):
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
            self.static_packed.copy_(
                self.replay.data.index_select(
                    0,
                    idx,
                )
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
    rows,
    gamma_tensor,
    gate,
    terminal_penalty_tensor,
    clip_max_tensor,
    clip_eps_tensor,
):
    batch = replay.sample(batch_size)

    train_math(
        model=model,
        target=target,
        optimizer=optimizer,
        batch=batch,
        rows=rows,
        gamma_tensor=gamma_tensor,
        gate=gate,
        terminal_penalty_tensor=(
            terminal_penalty_tensor
        ),
        clip_max_tensor=clip_max_tensor,
        clip_eps_tensor=clip_eps_tensor,
    )


def benchmark_case(fn, iterations, warmup):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(
        enable_timing=True
    )
    end_event = torch.cuda.Event(
        enable_timing=True
    )

    wall_start = time.perf_counter()
    start_event.record()

    for _ in range(iterations):
        fn()

    end_event.record()
    torch.cuda.synchronize()

    wall = time.perf_counter() - wall_start

    return {
        "iterations": int(iterations),
        "wall_ms_per_grad": (
            wall / iterations * 1000.0
        ),
        "cuda_window_ms_per_grad": (
            float(
                start_event.elapsed_time(
                    end_event
                )
            )
            / iterations
        ),
        "gradients_per_sec": (
            iterations / wall
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        default=(
            "models/"
            "v8_8_1_longtraj_gpu_replay_td_200k.pt"
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )
    ap.add_argument(
        "--replay-size",
        type=int,
        default=50000,
    )
    ap.add_argument(
        "--iterations",
        type=int,
        default=256,
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )
    ap.add_argument(
        "--gate",
        type=float,
        default=0.600,
    )
    ap.add_argument(
        "--terminal-penalty",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--report",
        default=(
            "data/"
            "v8_8_2_cuda_graph_v2_benchmark.json"
        ),
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device("cuda")

    print("=" * 80)
    print(
        "V8.8.2 CUDA GRAPH V2 — "
        "GRAPH-SAFE CAPTURE"
    )
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "Checkpoint:",
        args.checkpoint,
    )
    print(
        "Batch:",
        args.batch_size,
    )
    print()

    if not Path(args.checkpoint).exists():
        fallback = Path(
            "models/"
            "v8_8_jax_vectorized_td_150k.pt"
        )

        if fallback.exists():
            args.checkpoint = str(fallback)
            print(
                "Requested checkpoint missing; using:",
                args.checkpoint,
            )
        else:
            raise FileNotFoundError(
                args.checkpoint
            )

    state = load_state(
        args.checkpoint
    )

    replay = build_replay(
        args.replay_size,
        device,
        882202,
    )

    rows = torch.arange(
        args.batch_size,
        device=device,
    )

    gamma_tensor = torch.tensor(
        args.gamma,
        dtype=torch.float32,
        device=device,
    )

    terminal_penalty_tensor = torch.tensor(
        args.terminal_penalty,
        dtype=torch.float32,
        device=device,
    )

    clip_max_tensor = torch.tensor(
        1.0,
        dtype=torch.float32,
        device=device,
    )

    clip_eps_tensor = torch.tensor(
        1e-6,
        dtype=torch.float32,
        device=device,
    )

    eager_model = ObservableSafeQNetwork().to(
        device
    )
    eager_model.load_state_dict(state)
    eager_model.train()

    eager_target = ObservableSafeQNetwork().to(
        device
    )
    eager_target.load_state_dict(state)
    eager_target.eval()

    eager_lr = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=device,
    )

    eager_optimizer = torch.optim.AdamW(
        eager_model.parameters(),
        lr=eager_lr,
        weight_decay=args.weight_decay,
        capturable=True,
        foreach=False,
    )

    eager = benchmark_case(
        lambda: eager_step(
            model=eager_model,
            target=eager_target,
            optimizer=eager_optimizer,
            replay=replay,
            batch_size=args.batch_size,
            rows=rows,
            gamma_tensor=gamma_tensor,
            gate=args.gate,
            terminal_penalty_tensor=(
                terminal_penalty_tensor
            ),
            clip_max_tensor=(
                clip_max_tensor
            ),
            clip_eps_tensor=(
                clip_eps_tensor
            ),
        ),
        args.iterations,
        args.warmup,
    )

    print(
        "EAGER      "
        f"wall={eager['wall_ms_per_grad']:.3f}ms "
        f"CUDA-window="
        f"{eager['cuda_window_ms_per_grad']:.3f}ms "
        f"grad/s="
        f"{eager['gradients_per_sec']:.2f}"
    )

    del (
        eager_model,
        eager_target,
        eager_optimizer,
    )

    torch.cuda.empty_cache()

    graph_result = None
    graph_error = None

    try:
        graph = StaticCudaGraphLearner(
            model_state=state,
            replay=replay,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gate=args.gate,
            terminal_penalty=(
                args.terminal_penalty
            ),
            lr=0.0,
            weight_decay=(
                args.weight_decay
            ),
        )

        graph_result = benchmark_case(
            graph.step,
            args.iterations,
            args.warmup,
        )

        print(
            "CUDA GRAPH "
            f"wall="
            f"{graph_result['wall_ms_per_grad']:.3f}ms "
            f"CUDA-window="
            f"{graph_result['cuda_window_ms_per_grad']:.3f}ms "
            f"grad/s="
            f"{graph_result['gradients_per_sec']:.2f}"
        )

    except Exception:
        graph_error = traceback.format_exc()
        print(
            "CUDA GRAPH: FAILED"
        )
        print(
            graph_error
        )

    report = {
        "version": (
            "V8_8_2_CUDA_GRAPH_V2"
        ),
        "checkpoint": args.checkpoint,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "eager": eager,
        "cuda_graph": graph_result,
        "cuda_graph_error": graph_error,
    }

    if graph_result is not None:
        speedup = (
            graph_result[
                "gradients_per_sec"
            ]
            / max(
                eager[
                    "gradients_per_sec"
                ],
                1e-12,
            )
        )

        report["speedup"] = speedup

        print()
        print(
            "CUDA Graph speedup:",
            f"{speedup:.3f}x",
        )

        if speedup >= 1.05:
            print(
                "RESULT: >=5% real gain. "
                "Production integration is justified."
            )
        else:
            print(
                "RESULT: <5% gain. "
                "Keep eager production learner."
            )
    else:
        print()
        print(
            "RESULT: graph capture still fails; "
            "full traceback is saved."
        )

    out = Path(
        args.report
    )
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Report:",
        out,
    )
    print()
    print(
        "V8.8.2 CUDA GRAPH V2 BENCHMARK: PASS"
    )


if __name__ == "__main__":
    main()
