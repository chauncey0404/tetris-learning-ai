from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from singleplayer.network.q_network import (
    ObservableSafeQNetwork,
    STATE_SIZE,
    CANDIDATE_SIZE,
)
from tetris_ai.policy.confidence import normalized_margin_actions_torch

TOP_K = 4


class LowSyncMetricTracker:
    """
    Keep learner diagnostics on-device and cross the CUDA synchronization
    boundary only periodically. This removes the four .item() synchronizations
    that V8.8 performed on every gradient step.
    """

    def __init__(self, collect_every: int = 8, sync_every: int = 64):
        self.collect_every = max(1, int(collect_every))
        self.sync_every = max(self.collect_every, int(sync_every))
        self.pending = []
        self.steps_since_sync = 0

    def should_collect(self, gradient_step: int) -> bool:
        return int(gradient_step) % self.collect_every == 0

    def add(self, metric_tensor):
        if metric_tensor is not None:
            self.pending.append(metric_tensor.detach())
        self.steps_since_sync += 1

    def should_sync(self) -> bool:
        return self.steps_since_sync >= self.sync_every

    def flush(self):
        self.steps_since_sync = 0
        if not self.pending:
            return None
        # One host synchronization returns all diagnostics gathered since the
        # previous flush.
        values = torch.stack(self.pending, dim=0).mean(dim=0).float().cpu().numpy()
        self.pending.clear()
        result = {
            "loss": float(values[0]),
            "q_mean": float(values[1]),
            "target_mean": float(values[2]),
            "td_abs": float(values[3]),
        }
        for key, value in result.items():
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite learner metric: {key}={value}")
        return result


def train_batch_device_replay(
    *,
    model,
    target_model,
    optimizer,
    replay,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
    collect_metrics=False,
):
    """DDQN step using tensors that are already resident on learner device."""
    batch = replay.sample(batch_size)

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

        target_next_q_all = target_model(
            state=next_state,
            candidates=next_candidates,
            rewards=next_rewards,
            teacher_scores=next_teacher_scores,
            teacher_ranks=next_teacher_ranks,
        )

        rows = torch.arange(online_next_q.shape[0], device=online_next_q.device)
        selected_next_q = target_next_q_all[rows, next_action]
        has_next = next_mask.any(dim=1)
        bootstrap = (1.0 - done) * has_next.float() * selected_next_q

        learning_reward = reward - terminal_penalty * done
        td_target = learning_reward + gamma * bootstrap

    loss = F.smooth_l1_loss(current_q, td_target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if not collect_metrics:
        return None

    with torch.no_grad():
        td_error = td_target - current_q.detach()
        return torch.stack(
            (
                loss.detach(),
                current_q.detach().mean(),
                td_target.detach().mean(),
                td_error.detach().abs().mean(),
            )
        )


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def parse_int_list(text):
    values = []
    seen = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("Batch candidates must be positive integers.")
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("At least one batch candidate is required.")
    return values


def _benchmark_one_batch(
    state_dict,
    batch_size,
    device,
    lr,
    weight_decay,
    terminal_penalty,
    target_gate,
    gamma,
    warmup_iters,
    timed_iters,
):
    """Benchmark the REAL packed-replay sample + DDQN optimizer path."""
    if device.type != "cuda":
        return None

    from singleplayer.replay.packed import V881PackedReplayBuffer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = ObservableSafeQNetwork().to(device)
    model.load_state_dict(state_dict)
    model.train()
    target_model = ObservableSafeQNetwork().to(device)
    target_model.load_state_dict(state_dict)
    target_model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Large enough to reflect the device-side randperm/index_select cost while
    # staying modest compared with the production 100K replay allocation.
    benchmark_replay_size = max(50000, int(batch_size) * 2)
    replay = V881PackedReplayBuffer(
        capacity=benchmark_replay_size,
        device=device,
        seed=881810 + int(batch_size),
    )
    with torch.no_grad():
        replay.data.normal_(0.0, 0.5)
        lo, hi = replay.layout.offsets["next_mask"]
        replay.data[:, lo:hi].fill_(1.0)
        lo, hi = replay.layout.offsets["done"]
        replay.data[:, lo:hi].zero_()
        replay.size = replay.capacity
        replay.position = 0

    def step():
        train_batch_device_replay(
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            replay=replay,
            batch_size=int(batch_size),
            gamma=gamma,
            target_gate=target_gate,
            terminal_penalty=terminal_penalty,
            collect_metrics=False,
        )

    for _ in range(int(warmup_iters)):
        step()

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(int(timed_iters)):
        step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    result = {
        "batch_size": int(batch_size),
        "step_ms": elapsed / timed_iters * 1000.0,
        "steps_per_sec": timed_iters / max(elapsed, 1e-9),
        "samples_per_sec": timed_iters * int(batch_size) / max(elapsed, 1e-9),
        "peak_mb": peak_mb,
        "benchmark_replay_size": benchmark_replay_size,
        "benchmark_replay_mb": replay.nbytes / (1024.0 ** 2),
        "path": "packed_device_replay_plus_ddqn",
    }

    del replay, model, target_model, optimizer
    torch.cuda.empty_cache()
    return result


def benchmark_and_choose_batch(
    *,
    state_dict,
    candidates,
    device,
    lr,
    weight_decay,
    terminal_penalty,
    target_gate,
    gamma,
    warmup_iters,
    timed_iters,
    within_best_ratio,
):
    if device.type != "cuda":
        return min(candidates), []

    results = []
    print()
    print("=" * 80)
    print("CUDA PACKED-REPLAY + LEARNER AUTOTUNE")
    print("=" * 80)
    print()

    for batch_size in candidates:
        try:
            result = _benchmark_one_batch(
                state_dict=state_dict,
                batch_size=batch_size,
                device=device,
                lr=lr,
                weight_decay=weight_decay,
                terminal_penalty=terminal_penalty,
                target_gate=target_gate,
                gamma=gamma,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
            )
            results.append(result)
            print(
                f"batch={batch_size:>5} "
                f"step={result['step_ms']:7.3f}ms "
                f"samples/s={result['samples_per_sec']:>10.0f} "
                f"peakVRAM={result['peak_mb']:7.1f}MB"
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"batch={batch_size:>5} OOM")

    if not results:
        raise RuntimeError("No CUDA batch candidate succeeded.")

    best = max(x["samples_per_sec"] for x in results)
    threshold = best * float(within_best_ratio)
    eligible = [x for x in results if x["samples_per_sec"] >= threshold]
    chosen = min(eligible, key=lambda x: x["batch_size"])
    print()
    print(
        "AUTOTUNE CHOICE:",
        f"batch={chosen['batch_size']}",
        f"(smallest batch within {within_best_ratio*100:.1f}% of best samples/s)",
    )
    return int(chosen["batch_size"]), results


def save_checkpoint(
    *,
    path,
    model,
    target_model,
    optimizer,
    inherited_env_steps,
    new_env_steps,
    inherited_gradient_steps,
    new_gradient_steps,
    replay_size,
    unique_training_seeds,
    args,
    runtime_meta,
):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    total_env_steps = inherited_env_steps + new_env_steps
    total_gradient_steps = inherited_gradient_steps + new_gradient_steps

    payload = {
        "version": "V8_8_1_LONG_TRAJECTORY_PACKED_DEVICE_REPLAY_NORMALIZED_DDQN",
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "env_steps": int(total_env_steps),
        "gradient_steps": int(total_gradient_steps),
        "inherited_env_steps": int(inherited_env_steps),
        "new_env_steps": int(new_env_steps),
        "inherited_gradient_steps": int(inherited_gradient_steps),
        "new_gradient_steps": int(new_gradient_steps),
        "replay_size": int(replay_size),
        "state_size": int(STATE_SIZE),
        "candidate_size": int(CANDIDATE_SIZE),
        "top_k": int(TOP_K),
        "vector_envs": int(args.vector_envs),
        "risk_streams": int(args.risk_streams),
        "risk_fraction": float(args.risk_streams / max(args.vector_envs, 1)),
        "segment_pieces": int(args.segment_pieces),
        "unique_training_seeds_this_run": int(unique_training_seeds),
        "behavior_gate": float(args.behavior_gate),
        "risk_behavior_gate": float(args.risk_behavior_gate),
        "target_gate": float(args.target_gate),
        "normalized_gate": float(args.target_gate),
        "gate_semantics": "normalized_q_margin",
        "exploration": float(args.exploration),
        "gamma": float(args.gamma),
        "batch_size": int(args.batch_size),
        "warmup": int(args.warmup),
        "sample_budget": int(args.sample_budget),
        "terminal_penalty": float(args.terminal_penalty),
        "terminal_replay_copies": int(args.terminal_replay_copies),
        "target_update_samples": int(args.target_update_samples),
        "sync_every": int(args.sync_every),
        "queue_batches": int(args.queue_batches),
        "metric_collect_every": int(args.metric_collect_every),
        "metric_sync_every": int(args.metric_sync_every),
        "input_checkpoint": args.checkpoint,
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_prefix": args.checkpoint_prefix,
        "optimizer_resumed": bool(args.resume_optimizer),
        "batch_benchmark": getattr(args, "batch_benchmark", []),
        "generator_backend": "jax_cpu_vectorized_v4",
        "teacher_backend": "jax_vectorized_heuristic_v2_1",
        "replay_backend": "packed_device_resident_float32_v1",
        "qualification_status": "UNQUALIFIED_CHALLENGER",
        "policy_observation_rule": (
            "Q sees current state243 + observable candidate215 only; "
            "preview successor queue/current/hold tail is never an action input"
        ),
        "runtime_meta": dict(runtime_meta),
        "performance_design": {
            "jax_vectorized_generator_process": True,
            "long_trajectory_vector_envs": int(args.vector_envs),
            "packed_device_replay": True,
            "single_device_gather_per_gradient": True,
            "repeated_cpu_replay_gather_removed": True,
            "repeated_cpu_to_cuda_batch_copy_removed": True,
            "low_sync_metrics": True,
            "generator_learner_overlap": True,
            "fixed_sample_budget": int(args.sample_budget),
        },
    }
    torch.save(payload, path)
