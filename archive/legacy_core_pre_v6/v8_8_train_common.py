
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ai.observable_q_network import (
    ObservableSafeQNetwork,
    STATE_SIZE,
    CANDIDATE_SIZE,
)
from v8_7_scale_invariant_policy import normalized_margin_actions_torch


TOP_K = 4


def tensor(value, device, dtype=torch.float32):
    return torch.as_tensor(
        value,
        dtype=dtype,
        device=device,
    )


def train_batch(
    *,
    model,
    target_model,
    optimizer,
    replay,
    rng,
    batch_size,
    gamma,
    target_gate,
    terminal_penalty,
    device,
):
    batch = replay.sample(batch_size, rng)

    state = tensor(batch["state"], device)
    candidate = tensor(batch["candidate"], device)
    reward = tensor(batch["reward"], device)
    teacher_score = tensor(batch["teacher_score"], device)
    teacher_rank = tensor(batch["teacher_rank"], device)
    done = tensor(batch["done"], device)

    next_state = tensor(batch["next_state"], device)
    next_candidates = tensor(batch["next_candidates"], device)
    next_rewards = tensor(batch["next_rewards"], device)
    next_teacher_scores = tensor(batch["next_teacher_scores"], device)
    next_teacher_ranks = tensor(batch["next_teacher_ranks"], device)
    next_mask = tensor(batch["next_mask"], device, dtype=torch.bool)

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

        rows = torch.arange(
            online_next_q.shape[0],
            device=device,
        )

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

        selected_next_q = target_next_q_all[rows, next_action]
        has_next = next_mask.any(dim=1)

        bootstrap = (
            (1.0 - done)
            * has_next.float()
            * selected_next_q
        )

        # Preserve V8.7 semantics:
        # - raw observable reward stays as the Q-network input;
        # - terminal/no-next supervision gets an additional TD-only penalty.
        learning_reward = reward - terminal_penalty * done
        td_target = learning_reward + gamma * bootstrap

    loss = F.smooth_l1_loss(current_q, td_target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    td_error = td_target - current_q.detach()

    return {
        "loss": float(loss.item()),
        "q_mean": float(current_q.detach().mean().item()),
        "target_mean": float(td_target.mean().item()),
        "td_abs": float(td_error.abs().mean().item()),
    }


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
    if device.type != "cuda":
        return None

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

    b = int(batch_size)
    k = int(TOP_K)

    state = torch.randn(b, STATE_SIZE, device=device)
    candidate = torch.randn(b, CANDIDATE_SIZE, device=device)
    reward = torch.rand(b, device=device) * 0.25
    teacher_score = torch.randn(b, device=device)
    teacher_rank = torch.rand(b, device=device) * 4.0
    done = (torch.rand(b, device=device) < 0.02).float()

    next_state = torch.randn(b, STATE_SIZE, device=device)
    next_candidates = torch.randn(b, k, CANDIDATE_SIZE, device=device)
    next_rewards = torch.rand(b, k, device=device) * 0.25
    next_teacher_scores = torch.randn(b, k, device=device)
    next_teacher_ranks = torch.rand(b, k, device=device) * 4.0
    next_mask = torch.ones(b, k, dtype=torch.bool, device=device)

    rows = torch.arange(b, device=device)

    def step():
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

            next_action, _, _, _ = normalized_margin_actions_torch(
                online_next_q,
                next_mask,
                target_gate,
            )

            target_next_q = target_model(
                state=next_state,
                candidates=next_candidates,
                rewards=next_rewards,
                teacher_scores=next_teacher_scores,
                teacher_ranks=next_teacher_ranks,
            )

            selected_next_q = target_next_q[rows, next_action]
            learning_reward = reward - terminal_penalty * done
            td_target = (
                learning_reward
                + gamma * (1.0 - done) * selected_next_q
            )

        loss = F.smooth_l1_loss(current_q, td_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for _ in range(int(warmup_iters)):
        step()

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(int(timed_iters)):
        step()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    steps_per_sec = timed_iters / max(elapsed, 1e-9)
    samples_per_sec = timed_iters * b / max(elapsed, 1e-9)

    del model, target_model, optimizer
    del state, candidate, reward, teacher_score, teacher_rank, done
    del next_state, next_candidates, next_rewards
    del next_teacher_scores, next_teacher_ranks, next_mask, rows
    torch.cuda.empty_cache()

    return {
        "batch_size": b,
        "step_ms": elapsed / timed_iters * 1000.0,
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec,
        "peak_mb": peak_mb,
    }


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
    print("CUDA BATCH AUTOTUNE")
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
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"batch={batch_size:>5}: OOM -> skipped")
                torch.cuda.empty_cache()
                continue
            raise

        results.append(result)
        print(
            f"batch={batch_size:>5} "
            f"step={result['step_ms']:7.3f}ms "
            f"samples/s={result['samples_per_sec']:10.0f} "
            f"peakVRAM={result['peak_mb']:7.1f}MB"
        )

    if not results:
        raise RuntimeError("CUDA batch autotune produced no valid batch size.")

    best_rate = max(item["samples_per_sec"] for item in results)
    eligible = [
        item
        for item in results
        if item["samples_per_sec"] >= best_rate * float(within_best_ratio)
    ]

    chosen = min(
        eligible,
        key=lambda item: item["batch_size"],
    )["batch_size"]

    print()
    print(
        f"AUTOTUNE CHOICE: batch={chosen} "
        f"(smallest batch within {within_best_ratio * 100:.1f}% "
        "of best samples/s)"
    )

    return int(chosen), results


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
        "version": (
            "V8_8_JAX_VECTORIZED_OBSERVABLE_SAFE_"
            "NORMALIZED_GATE_DDQN"
        ),
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
        "q_gate": float(args.target_gate),
        "gate_semantics": "normalized_q_margin",
        "gate_type": "normalized_q_span_v1",
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
        "input_checkpoint": args.checkpoint,
        "checkpoint_every": int(args.checkpoint_every),
        "checkpoint_prefix": args.checkpoint_prefix,
        "max_batch_fraction": float(args.max_batch_fraction),
        "optimizer_resumed": bool(args.resume_optimizer),
        "batch_benchmark": getattr(args, "batch_benchmark", []),
        "generator_backend": "jax_cpu_vectorized_v4",
        "teacher_backend": "jax_vectorized_heuristic_v2_1",
        "generator_candidate_slots": 80,
        "jax_backend_parity": "REQUIRED_PASS_BEFORE_TRAINING",
        "jax_teacher_topk_parity": "REQUIRED_PASS_BEFORE_TRAINING",
        "qualification_status": "UNQUALIFIED_CHALLENGER",
        "policy_observation_rule": (
            "Q sees current state243 + observable candidate215 only; "
            "preview successor queue/current/hold tail is never an action input"
        ),
        "runtime_meta": dict(runtime_meta),
        "performance_design": {
            "scalar_gym_actors_removed": True,
            "jax_vectorized_generator_process": True,
            "vector_envs": int(args.vector_envs),
            "batched_cpu_q_inference": True,
            "array_replay_add_batch": True,
            "generator_learner_overlap": True,
            "fixed_sample_budget": int(args.sample_budget),
            "weight_sync_interval": int(args.sync_every),
            "queue_batches": int(args.queue_batches),
        },
    }

    torch.save(payload, path)
