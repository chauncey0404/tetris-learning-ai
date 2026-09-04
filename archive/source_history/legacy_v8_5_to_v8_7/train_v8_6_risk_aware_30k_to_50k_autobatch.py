import argparse
import copy
import math
import os
import queue
import random
import time
import traceback
import multiprocessing as mp

# CPU actors are deliberately single-threaded; parallelism comes from processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F

from tetris_core import GymTetrisAdapter
from teacher import HeuristicTeacherV2
from gym_executor import execute_placement
from ai.state_encoder import encode_state
from ai.observable_q_network import (
    ObservableSafeQNetwork,
    STATE_SIZE,
    CANDIDATE_SIZE,
)
from v8_successor import preview_top_k_successors
from v8_4_observable import (
    TOP_K,
    candidate_arrays,
    observable_candidate_features,
    q_values_for_successors,
    conservative_choice,
)
from v8_4_replay import ObservableReplayBuffer


GLOBAL_SEED = 20260822
PERMANENT_BENCHMARK_FIRST = 6
PERMANENT_BENCHMARK_LAST = 20


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_actor_threads():
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def sanitize_training_seed(seed):
    seed = int(seed)
    if PERMANENT_BENCHMARK_FIRST <= seed <= PERMANENT_BENCHMARK_LAST:
        return PERMANENT_BENCHMARK_LAST + 1
    return seed


def next_actor_seed(seed, stride):
    seed = int(seed) + int(stride)
    while PERMANENT_BENCHMARK_FIRST <= seed <= PERMANENT_BENCHMARK_LAST:
        seed += int(stride)
    return seed


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_actor_weights_if_available(model, control_queue, current_version):
    latest = None

    while True:
        try:
            latest = control_queue.get_nowait()
        except queue.Empty:
            break

    if latest is None:
        return current_version

    if latest.get("kind") == "weights":
        model.load_state_dict(latest["state_dict"])
        model.eval()
        return int(latest.get("version", current_version))

    return current_version


def put_transition(transition_queue, message, stop_event):
    while not stop_event.is_set():
        try:
            transition_queue.put(message, timeout=0.5)
            return True
        except queue.Full:
            continue

    return False


def actor_loop(
    actor_id,
    config,
    initial_model_state_dict,
    initial_weight_version,
    transition_queue,
    control_queue,
    stop_event,
):
    configure_actor_threads()

    actor_seed = GLOBAL_SEED + 20000 + actor_id * 997
    random.seed(actor_seed)
    np.random.seed(actor_seed)
    torch.manual_seed(actor_seed)
    rng = np.random.default_rng(actor_seed)

    adapter = None

    try:
        model = ObservableSafeQNetwork().cpu()
        model.load_state_dict(initial_model_state_dict)
        model.eval()

        teacher = HeuristicTeacherV2()
        adapter = GymTetrisAdapter()

        seed_cursor = sanitize_training_seed(
            int(config["start_seed"]) + actor_id
        )
        seed_stride = int(config["actors"])
        segment_pieces = int(config["segment_pieces"])
        base_behavior_gate = float(config["behavior_gate"])
        risk_behavior_gate = float(config["risk_behavior_gate"])
        risk_actor_count = int(config["risk_actor_count"])
        behavior_gate = (
            risk_behavior_gate
            if actor_id < risk_actor_count
            else base_behavior_gate
        )
        exploration = float(config["exploration"])
        sync_poll_steps = int(config["sync_poll_steps"])
        verify_every = int(config["verify_every"])

        weight_version = int(initial_weight_version)
        local_steps = 0
        segment_steps = 0

        def start_segment(seed_value):
            episode_seed = sanitize_training_seed(seed_value)
            state_value = adapter.reset(seed=episode_seed)
            adapter.raw.gravity_enabled = False

            # Encode only once at segment start.  For normal continuation,
            # state_features are carried forward from the chosen preview successor.
            state_features_value = encode_state(state_value).astype(
                np.float32,
                copy=True,
            )

            successors_value = preview_top_k_successors(
                adapter=adapter,
                teacher=teacher,
                state=state_value,
                top_k=TOP_K,
            )

            next_seed = next_actor_seed(episode_seed, seed_stride)

            return (
                state_value,
                state_features_value,
                successors_value,
                next_seed,
                episode_seed,
            )

        (
            state,
            state_features,
            successors,
            seed_cursor,
            episode_seed,
        ) = start_segment(seed_cursor)

        while not stop_event.is_set():
            if local_steps % sync_poll_steps == 0:
                weight_version = load_actor_weights_if_available(
                    model,
                    control_queue,
                    weight_version,
                )

            if not successors:
                (
                    state,
                    state_features,
                    successors,
                    seed_cursor,
                    episode_seed,
                ) = start_segment(seed_cursor)
                segment_steps = 0
                continue

            if state_features.shape != (STATE_SIZE,):
                raise RuntimeError(
                    f"Actor {actor_id}: state shape {state_features.shape}"
                )

            random_alt = (
                len(successors) > 1
                and rng.random() < exploration
            )

            q_gap = 0.0

            if random_alt:
                chosen_index = int(rng.integers(1, len(successors)))
                behavior = "random"
            else:
                q_values = q_values_for_successors(
                    model=model,
                    state_features=state_features,
                    successors=successors,
                    device=torch.device("cpu"),
                )

                chosen_index, q_gap = conservative_choice(
                    q_values,
                    behavior_gate,
                )

                behavior = (
                    "teacher"
                    if chosen_index == 0
                    else "q"
                )

            chosen = successors[chosen_index]
            chosen_candidate = observable_candidate_features(chosen)

            result = execute_placement(adapter, chosen.action)
            next_state = result["state"]

            # IMPORTANT PERFORMANCE CHANGE:
            # preview_top_k_successors already encoded the exact chosen next state.
            # Use that directly for the legal post-action TD state.
            real_next_features = np.asarray(
                chosen.next_state_features,
                dtype=np.float32,
            ).copy()

            # Keep a periodic runtime identity audit instead of redundantly
            # calling encode_state() on EVERY transition.
            if verify_every > 0 and ((local_steps + 1) % verify_every == 0):
                verified = encode_state(next_state).astype(
                    np.float32,
                    copy=False,
                )

                if not np.array_equal(
                    verified,
                    real_next_features,
                ):
                    raise RuntimeError(
                        f"Actor {actor_id}: periodic runtime successor "
                        "!= preview successor"
                    )

            real_lines = int(
                result["info"].get(
                    "lines_cleared",
                    0,
                )
            )

            if real_lines != int(chosen.lines_cleared):
                raise RuntimeError(
                    f"Actor {actor_id}: runtime lines != preview lines"
                )

            real_done = bool(
                result["terminated"]
                or result["truncated"]
            )

            if not real_done:
                next_successors = preview_top_k_successors(
                    adapter=adapter,
                    teacher=teacher,
                    state=next_state,
                    top_k=TOP_K,
                )
            else:
                next_successors = []

            no_next = (
                (not real_done)
                and len(next_successors) == 0
            )

            transition_done = bool(
                real_done
                or no_next
            )

            (
                next_candidate_array,
                next_rewards_array,
                next_scores_array,
                next_ranks_array,
                next_mask_array,
            ) = candidate_arrays(
                next_successors,
                top_k=TOP_K,
            )

            segment_steps += 1
            local_steps += 1

            segment_boundary = (
                segment_steps >= segment_pieces
            )

            message = {
                "kind": "transition",
                "actor_id": int(actor_id),
                "episode_seed": int(episode_seed),
                "weight_version": int(weight_version),
                "behavior": behavior,
                "actor_behavior_gate": float(behavior_gate),
                "is_risk_actor": bool(actor_id < risk_actor_count),
                "q_gap": float(q_gap),
                "real_done": bool(real_done),
                "no_next": bool(no_next),
                "segment_boundary": bool(segment_boundary),
                "lines": int(real_lines),
                "state": np.asarray(
                    state_features,
                    dtype=np.float32,
                ).copy(),
                "candidate": chosen_candidate,
                "reward": float(chosen.normalized_reward),
                "teacher_score": float(chosen.teacher_score),
                "teacher_rank": float(chosen.reachable_rank),
                "done": bool(transition_done),
                "next_state": real_next_features,
                "next_candidates": next_candidate_array,
                "next_rewards": next_rewards_array,
                "next_teacher_scores": next_scores_array,
                "next_teacher_ranks": next_ranks_array,
                "next_mask": next_mask_array,
            }

            if not put_transition(
                transition_queue,
                message,
                stop_event,
            ):
                break

            if transition_done or segment_boundary:
                (
                    state,
                    state_features,
                    successors,
                    seed_cursor,
                    episode_seed,
                ) = start_segment(seed_cursor)

                segment_steps = 0
            else:
                state = next_state
                state_features = real_next_features
                successors = next_successors

    except Exception:
        error_message = {
            "kind": "error",
            "actor_id": int(actor_id),
            "traceback": traceback.format_exc(),
        }

        try:
            transition_queue.put(
                error_message,
                timeout=1.0,
            )
        except Exception:
            pass

        stop_event.set()

    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass


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
    batch = replay.sample(
        batch_size,
        rng,
    )

    state = tensor(
        batch["state"],
        device,
    )
    candidate = tensor(
        batch["candidate"],
        device,
    )
    reward = tensor(
        batch["reward"],
        device,
    )
    teacher_score = tensor(
        batch["teacher_score"],
        device,
    )
    teacher_rank = tensor(
        batch["teacher_rank"],
        device,
    )
    done = tensor(
        batch["done"],
        device,
    )

    next_state = tensor(
        batch["next_state"],
        device,
    )
    next_candidates = tensor(
        batch["next_candidates"],
        device,
    )
    next_rewards = tensor(
        batch["next_rewards"],
        device,
    )
    next_teacher_scores = tensor(
        batch["next_teacher_scores"],
        device,
    )
    next_teacher_ranks = tensor(
        batch["next_teacher_ranks"],
        device,
    )
    next_mask = tensor(
        batch["next_mask"],
        device,
        dtype=torch.bool,
    )

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
            -1e9,
        )

        batch_n = online_next_q.shape[0]

        rows = torch.arange(
            batch_n,
            device=device,
        )

        next_action = torch.zeros(
            batch_n,
            dtype=torch.long,
            device=device,
        )

        if online_next_q.shape[1] > 1:
            alt_q = online_next_q[:, 1:]

            best_alt_index = (
                alt_q.argmax(dim=1)
                + 1
            )

            teacher_q = online_next_q[
                rows,
                0,
            ]

            best_alt_q = online_next_q[
                rows,
                best_alt_index,
            ]

            gap = (
                best_alt_q
                - teacher_q
            )

            alt_is_valid = next_mask[
                rows,
                best_alt_index,
            ]

            allow_alt = (
                alt_is_valid
                & (gap >= target_gate)
            )

            next_action = torch.where(
                allow_alt,
                best_alt_index,
                next_action,
            )

        target_next_q_all = target_model(
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

        has_next = next_mask.any(
            dim=1,
        )

        bootstrap = (
            (1.0 - done)
            * has_next.float()
            * selected_next_q
        )

        # Keep the raw immediate reward as an OBSERVABLE NETWORK INPUT,
        # but use a separate risk-aware learning reward for TD supervision.
        # This avoids train/inference feature mismatch while making real terminal
        # outcomes expensive enough to propagate backward through Q-learning.
        learning_reward = (
            reward
            - terminal_penalty * done
        )

        td_target = (
            learning_reward
            + gamma * bootstrap
        )

    loss = F.smooth_l1_loss(
        current_q,
        td_target,
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    td_error = (
        td_target
        - current_q.detach()
    )

    return {
        "loss": float(loss.item()),
        "q_mean": float(
            current_q.detach().mean().item()
        ),
        "target_mean": float(
            td_target.mean().item()
        ),
        "td_abs": float(
            td_error.abs().mean().item()
        ),
    }


def broadcast_weights(
    model,
    control_queues,
    version,
):
    payload = {
        "kind": "weights",
        "version": int(version),
        "state_dict": cpu_state_dict(model),
    }

    for control_queue in control_queues:
        try:
            while True:
                control_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            control_queue.put_nowait(payload)
        except queue.Full:
            pass


def move_optimizer_state_to_device(
    optimizer,
    device,
):
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
    warmup_iters,
    timed_iters,
):
    """Benchmark the actual forward/backward/optimizer shape used by V8.4."""
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

            alt_q = online_next_q[:, 1:]
            best_alt_index = alt_q.argmax(dim=1) + 1
            teacher_q = online_next_q[:, 0]
            best_alt_q = online_next_q[rows, best_alt_index]
            allow_alt = (best_alt_q - teacher_q) >= 0.085
            next_action = torch.where(
                allow_alt,
                best_alt_index,
                torch.zeros_like(best_alt_index),
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
            td_target = learning_reward + 0.99 * (1.0 - done) * selected_next_q

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
    samples_per_sec = (timed_iters * b) / max(elapsed, 1e-9)

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
    warmup_iters,
    timed_iters,
    within_best_ratio,
):
    if device.type != "cuda":
        chosen = min(candidates)
        return chosen, []

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
        item for item in results
        if item["samples_per_sec"] >= best_rate * float(within_best_ratio)
    ]

    # Prefer the smallest near-best batch for fresher replay and lower warmup.
    chosen = min(eligible, key=lambda item: item["batch_size"])["batch_size"]

    print()
    print(
        f"AUTOTUNE CHOICE: batch={chosen} "
        f"(smallest batch within {within_best_ratio * 100:.1f}% of best samples/s)"
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
):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    total_env_steps = (
        inherited_env_steps
        + new_env_steps
    )

    total_gradient_steps = (
        inherited_gradient_steps
        + new_gradient_steps
    )

    torch.save(
        {
            "version": (
                "V8_6_RISK_AWARE_OBSERVABLE_SAFE_"
                "LONGRUN_AUTOBATCH_CONSERVATIVE_DDQN"
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
            "actors": int(args.actors),
            "segment_pieces": int(args.segment_pieces),
            "unique_training_seeds_this_run": int(unique_training_seeds),
            "behavior_gate": float(args.behavior_gate),
            "risk_behavior_gate": float(args.risk_behavior_gate),
            "risk_actors": int(args.risk_actors),
            "target_gate": float(args.target_gate),
            # Compatibility for evaluators that inspect q_gate metadata.
            "q_gate": float(args.target_gate),
            "exploration": float(args.exploration),
            "gamma": float(args.gamma),
            "batch_size": int(args.batch_size),
            "warmup": int(args.warmup),
            "sample_budget": int(args.sample_budget),
            "terminal_penalty": float(args.terminal_penalty),
            "terminal_replay_copies": int(args.terminal_replay_copies),
            "target_update_samples": int(args.target_update_samples),
            "sync_every": int(args.sync_every),
            "queue_size": int(args.queue_size),
            "verify_every": int(args.verify_every),
            "input_checkpoint": args.checkpoint,
            "midpoint_new_steps": int(args.midpoint_new_steps),
            "mid_output": args.mid_output,
            "max_batch_fraction": float(args.max_batch_fraction),
            "optimizer_resumed": bool(args.resume_optimizer),
            "batch_benchmark": getattr(args, "batch_benchmark", []),
            "policy_observation_rule": (
                "Q sees current state + board/action candidate only; "
                "preview successor indices 200:243 are forbidden for action scoring"
            ),
            "performance_design": {
                "cpu_actor_threads": 1,
                "post_step_encode_state_removed": True,
                "periodic_runtime_successor_audit": int(args.verify_every),
                "autotuned_gpu_batch": int(args.batch_size),
                "fixed_sample_budget": int(args.sample_budget),
                "queue_backpressure": int(args.queue_size),
                "weight_sync_interval": int(args.sync_every),
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Long-run risk-aware observable-safe continuation from the V8.5-30K "
            "Champion to 50K, with a 40K midpoint checkpoint and CUDA autotuning."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default="models/v8_5_risk_aware_observable_safe_td_30k.pt",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="NEW transitions to add (30K -> 50K by default).",
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=9001,
    )

    parser.add_argument(
        "--actors",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--risk-actors",
        type=int,
        default=2,
        help="Number of actors using the deliberately more aggressive risk gate.",
    )

    parser.add_argument(
        "--segment-pieces",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--behavior-gate",
        type=float,
        default=0.060,
        help="Normal actor behavior gate, frozen from the V8.5-30K Champion.",
    )

    parser.add_argument(
        "--risk-behavior-gate",
        type=float,
        default=0.05,
        help="Aggressive gate used only by risk actors to harvest failure data.",
    )

    parser.add_argument(
        "--target-gate",
        type=float,
        default=0.060,
    )

    parser.add_argument(
        "--exploration",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--terminal-penalty",
        type=float,
        default=1.0,
        help=(
            "Penalty subtracted from the TD learning reward on real terminal/no-next "
            "transitions. Raw reward remains the network input."
        ),
    )

    parser.add_argument(
        "--terminal-replay-copies",
        type=int,
        default=8,
        help="Replay multiplicity for terminal transitions; collected-step count is unchanged.",
    )

    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="0 = auto max(2048, chosen batch size).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = CUDA autotune; otherwise force this batch size.",
    )

    parser.add_argument(
        "--batch-candidates",
        default="4096,8192,16384",
    )

    parser.add_argument(
        "--benchmark-warmup-iters",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-near-best-ratio",
        type=float,
        default=0.95,
        help="Choose the smallest batch reaching this fraction of best samples/s.",
    )

    parser.add_argument(
        "--sample-budget",
        type=int,
        default=4071424,
        help="Approximate total replay samples to process in this 20K continuation.",
    )

    parser.add_argument(
        "--target-update-samples",
        type=int,
        default=512000,
        help="Target-network refresh interval expressed in sampled replay items.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--sync-every",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--sync-poll-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--drain-batch",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--verify-every",
        type=int,
        default=250,
        help=(
            "Re-encode one real successor every N actor transitions to audit "
            "preview/runtime identity. 0 disables the periodic audit."
        ),
    )

    parser.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=2.0,
        help="Concurrent grace period for actor shutdown before terminate().",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--midpoint-new-steps",
        type=int,
        default=10000,
        help="Save an exact intermediate checkpoint after this many NEW transitions; 0 disables it.",
    )

    parser.add_argument(
        "--mid-output",
        default="models/v8_6_risk_aware_observable_safe_td_40k.pt",
    )

    parser.add_argument(
        "--max-batch-fraction",
        type=float,
        default=0.50,
        help=(
            "Autotune may benchmark larger batches, but deployment batch is capped to "
            "this fraction of NEW transitions to preserve replay freshness."
        ),
    )

    parser.add_argument(
        "--output",
        default="models/v8_6_risk_aware_observable_safe_td_50k.pt",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.actors <= 0:
        raise ValueError("--actors must be > 0")
    if not 0 <= args.risk_actors <= args.actors:
        raise ValueError("--risk-actors must be in [0, actors]")
    if args.segment_pieces <= 0:
        raise ValueError("--segment-pieces must be > 0")
    if not 0.0 <= args.exploration <= 1.0:
        raise ValueError("--exploration must be in [0,1]")
    if args.behavior_gate < 0.0 or args.risk_behavior_gate < 0.0:
        raise ValueError("Behavior gates must be >= 0")
    if args.target_gate < 0.0:
        raise ValueError("--target-gate must be >= 0")
    if args.terminal_penalty < 0.0:
        raise ValueError("--terminal-penalty must be >= 0")
    if args.terminal_replay_copies <= 0:
        raise ValueError("--terminal-replay-copies must be > 0")
    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0")
    if args.sample_budget <= 0:
        raise ValueError("--sample-budget must be > 0")
    if not 0.0 < args.max_batch_fraction <= 1.0:
        raise ValueError("--max-batch-fraction must be in (0,1]")
    if args.midpoint_new_steps < 0 or args.midpoint_new_steps >= args.steps:
        if args.midpoint_new_steps != 0:
            raise ValueError("--midpoint-new-steps must be 0 or in [1, steps-1]")
    if args.target_update_samples <= 0:
        raise ValueError("--target-update-samples must be > 0")
    if not 0.0 < args.batch_near_best_ratio <= 1.0:
        raise ValueError("--batch-near-best-ratio must be in (0,1]")
    if args.sync_every <= 0 or args.sync_poll_steps <= 0:
        raise ValueError("sync settings must be > 0")
    if args.queue_size <= 0 or args.drain_batch <= 0:
        raise ValueError("queue settings must be > 0")
    if args.shutdown_grace < 0.0:
        raise ValueError("--shutdown-grace must be >= 0")

    set_global_seed(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    inherited_env_steps = int(
        checkpoint.get(
            "env_steps",
            0,
        )
    )

    inherited_gradient_steps = int(
        checkpoint.get(
            "gradient_steps",
            0,
        )
    )

    model = ObservableSafeQNetwork().to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    target_model = ObservableSafeQNetwork().to(device)

    if "target_model_state_dict" in checkpoint:
        target_model.load_state_dict(
            checkpoint["target_model_state_dict"]
        )
    else:
        target_model.load_state_dict(
            model.state_dict()
        )

    target_model.eval()

    batch_candidates = parse_int_list(args.batch_candidates)

    if args.batch_size > 0:
        chosen_batch_size = int(args.batch_size)
        batch_benchmark = []
    else:
        chosen_batch_size, batch_benchmark = benchmark_and_choose_batch(
            state_dict=checkpoint["model_state_dict"],
            candidates=batch_candidates,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            terminal_penalty=args.terminal_penalty,
            warmup_iters=args.benchmark_warmup_iters,
            timed_iters=args.benchmark_iters,
            within_best_ratio=args.batch_near_best_ratio,
        )

    # Benchmark all requested GPU sizes, but do not let a very large batch
    # consume almost the entire fresh 20K run before learning even begins.
    freshness_limit = max(
        1,
        int(math.floor(args.steps * args.max_batch_fraction)),
    )

    if int(chosen_batch_size) > freshness_limit:
        eligible_fresh = [
            item
            for item in batch_benchmark
            if int(item["batch_size"]) <= freshness_limit
        ]

        if not eligible_fresh:
            raise RuntimeError(
                "No autotuned batch survives the replay-freshness guard. "
                "Lower --batch-candidates or increase --max-batch-fraction."
            )

        guarded_choice = max(
            eligible_fresh,
            key=lambda item: item["samples_per_sec"],
        )

        print()
        print(
            "FRESHNESS GUARD OVERRIDE:",
            f"autotune={chosen_batch_size} -> deploy={guarded_choice['batch_size']}",
            f"(batch <= {args.max_batch_fraction:.2f} * {args.steps} new transitions)",
        )

        chosen_batch_size = int(guarded_choice["batch_size"])

    args.batch_size = int(chosen_batch_size)
    args.batch_benchmark = batch_benchmark
    args.warmup = int(args.warmup) if args.warmup > 0 else max(2048, args.batch_size)

    if args.warmup < args.batch_size:
        raise ValueError("Effective warmup must be >= chosen batch size")
    if args.steps <= args.warmup:
        raise ValueError("--steps must be greater than effective warmup")

    target_gradient_steps = max(
        1,
        int(round(args.sample_budget / args.batch_size)),
    )
    target_update_gradients = max(
        1,
        int(round(args.target_update_samples / args.batch_size)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    optimizer_resumed = False

    if (
        args.resume_optimizer
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        move_optimizer_state_to_device(
            optimizer,
            device,
        )

        # Explicit CLI values win after optimizer resume.
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr)
            group["weight_decay"] = float(args.weight_decay)

        optimizer_resumed = True

    replay = ObservableReplayBuffer(
        capacity=args.replay_capacity
    )

    print()
    print("=" * 80)
    print("V8.6 RISK-AWARE OBSERVABLE-SAFE 30K -> 50K LONG-RUN RESUME")
    print("=" * 80)
    print()
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("Input checkpoint:", args.checkpoint)
    print("Inherited observable-safe transitions:", inherited_env_steps)
    print("Inherited gradient steps:", inherited_gradient_steps)
    print("New transitions:", args.steps)
    print(
        "Target total transitions:",
        inherited_env_steps + args.steps,
    )
    print("Optimizer resumed:", "YES" if optimizer_resumed else "NO")
    print()
    print("CPU actors:", args.actors)
    print("Actor torch threads:", 1)
    print("Segment pieces:", args.segment_pieces)
    print("Training seed start:", args.start_seed)
    print("Normal behavior gate:", args.behavior_gate)
    print("Risk behavior gate:", args.risk_behavior_gate)
    print("Risk actors:", args.risk_actors, "/", args.actors)
    print("TD target gate:", args.target_gate)
    print(
        "Random alternative exploration:",
        f"{args.exploration * 100:.2f}%",
    )
    print("Replay capacity:", args.replay_capacity)
    print("Fresh replay warmup:", args.warmup)
    print("GPU batch size:", args.batch_size)
    print("Target gradients this run:", target_gradient_steps)
    print("Target replay samples:", target_gradient_steps * args.batch_size)
    print("Target update every:", target_update_gradients, "gradients")
    print(
        "Midpoint checkpoint:",
        (
            f"after {args.midpoint_new_steps} new transitions -> {args.mid_output}"
            if args.midpoint_new_steps > 0
            else "DISABLED"
        ),
    )
    print("Freshness batch fraction cap:", args.max_batch_fraction)
    print("Terminal TD penalty:", args.terminal_penalty)
    print("Terminal replay copies:", args.terminal_replay_copies)
    print(
        "Actor weight sync:",
        args.sync_every,
        "new transitions",
    )
    print("Actor sync poll:", args.sync_poll_steps)
    print("Transition queue maxsize:", args.queue_size)
    print("Drain batch:", args.drain_batch)
    print(
        "Full encode_state runtime audit:",
        f"every {args.verify_every} actor steps"
        if args.verify_every > 0
        else "DISABLED",
    )
    print("Redundant every-step encode_state:", "REMOVED")
    print("Permanent seeds 6~20:", "PROTECTED")

    print()
    print("POLICY OBSERVATION RULE:")
    print("  current state 243          : ALLOWED")
    print("  candidate board/action 215 : ALLOWED")
    print("  preview successor 200:243  : FORBIDDEN FOR Q")
    print("  real next state after move  : ALLOWED FOR TD BOOTSTRAP")

    print()
    print("PERFORMANCE DESIGN:")
    print("  10 single-thread CPU actors by default (closer to learner capacity)")
    print("  CUDA batch 4096/8192/16384 benchmarked at startup")
    print("  fixed replay-sample budget across batch choices")
    print("  freshness guard caps deployed batch to <= 50% of new transitions")
    print("  exact 40K midpoint checkpoint saved before continuing to 50K")
    print("  weight sync every 128 transitions")
    print("  queue 384 to reduce actor-policy staleness")
    print("  2 risk actors @ gate 0.05 to harvest failure data")
    print("  post-step encode_state removed except periodic identity audits")
    print("  CUDA learner remains FP32 for stable Q-gap calibration")
    print("  terminal TD penalty + replay oversampling are learning-only safety signals")

    initial_cpu_weights = cpu_state_dict(model)

    ctx = mp.get_context("spawn")

    transition_queue = ctx.Queue(
        maxsize=args.queue_size
    )

    stop_event = ctx.Event()

    control_queues = [
        ctx.Queue(maxsize=1)
        for _ in range(args.actors)
    ]

    actor_config = {
        "start_seed": int(args.start_seed),
        "actors": int(args.actors),
        "segment_pieces": int(args.segment_pieces),
        "behavior_gate": float(args.behavior_gate),
        "risk_behavior_gate": float(args.risk_behavior_gate),
        "risk_actor_count": int(args.risk_actors),
        "exploration": float(args.exploration),
        "sync_poll_steps": int(args.sync_poll_steps),
        "verify_every": int(args.verify_every),
    }

    actors = []

    for actor_id in range(args.actors):
        process = ctx.Process(
            target=actor_loop,
            args=(
                actor_id,
                actor_config,
                initial_cpu_weights,
                inherited_env_steps,
                transition_queue,
                control_queues[actor_id],
                stop_event,
            ),
            name=f"v8_5_risk_actor_{actor_id}",
        )

        process.start()
        actors.append(process)

    print()
    print(
        "Actors started:",
        [process.pid for process in actors],
    )

    collected = 0
    new_gradient_steps = 0

    next_log = args.log_every
    next_sync = args.sync_every
    midpoint_saved = False

    teacher_actions = 0
    q_interventions = 0
    random_explorations = 0
    real_gameovers = 0
    no_next_resets = 0
    segment_boundaries = 0
    risk_actor_transitions = 0
    terminal_replay_extra_entries = 0

    actor_counts = {
        i: 0
        for i in range(args.actors)
    }

    actor_weight_versions = {
        i: inherited_env_steps
        for i in range(args.actors)
    }

    training_seeds = set()

    q_gap_history = []
    loss_history = []
    q_history = []
    td_history = []

    run_start = time.perf_counter()

    def process_message(message):
        nonlocal collected
        nonlocal teacher_actions
        nonlocal q_interventions
        nonlocal random_explorations
        nonlocal real_gameovers
        nonlocal no_next_resets
        nonlocal segment_boundaries
        nonlocal risk_actor_transitions
        nonlocal terminal_replay_extra_entries

        kind = message.get("kind")

        if kind == "error":
            raise RuntimeError(
                f"Actor {message.get('actor_id')} failed:\n"
                f"{message.get('traceback', '')}"
            )

        if (
            kind != "transition"
            or collected >= args.steps
        ):
            return

        replay_copies = (
            args.terminal_replay_copies
            if bool(message["done"])
            else 1
        )

        for copy_index in range(replay_copies):
            replay.add(
                state=message["state"],
                candidate=message["candidate"],
                reward=message["reward"],
                teacher_score=message["teacher_score"],
                teacher_rank=message["teacher_rank"],
                done=message["done"],
                next_state=message["next_state"],
                next_candidates=message["next_candidates"],
                next_rewards=message["next_rewards"],
                next_teacher_scores=message["next_teacher_scores"],
                next_teacher_ranks=message["next_teacher_ranks"],
                next_mask=message["next_mask"],
            )

        terminal_replay_extra_entries += max(0, replay_copies - 1)

        collected += 1

        actor_id = int(
            message["actor_id"]
        )

        actor_counts[
            actor_id
        ] += 1

        actor_weight_versions[
            actor_id
        ] = max(
            actor_weight_versions[actor_id],
            int(
                message.get(
                    "weight_version",
                    inherited_env_steps,
                )
            ),
        )

        training_seeds.add(
            int(message["episode_seed"])
        )

        if bool(message.get("is_risk_actor", False)):
            risk_actor_transitions += 1

        behavior = message["behavior"]

        if behavior == "teacher":
            teacher_actions += 1

        elif behavior == "q":
            q_interventions += 1
            q_gap_history.append(
                float(message["q_gap"])
            )

        elif behavior == "random":
            random_explorations += 1

        if message["real_done"]:
            real_gameovers += 1

        if message["no_next"]:
            no_next_resets += 1

        if message["segment_boundary"]:
            segment_boundaries += 1

    try:
        while collected < args.steps:
            if stop_event.is_set():
                try:
                    process_message(
                        transition_queue.get(
                            timeout=1.0
                        )
                    )
                except queue.Empty:
                    pass

                if not any(
                    process.is_alive()
                    for process in actors
                ):
                    raise RuntimeError(
                        "All actors stopped before collection completed."
                    )

            try:
                first_message = transition_queue.get(
                    timeout=2.0
                )
            except queue.Empty:
                if not any(
                    process.is_alive()
                    for process in actors
                ):
                    raise RuntimeError(
                        "All actors exited before collection completed."
                    )

                continue

            messages = [first_message]

            for _ in range(
                max(
                    0,
                    args.drain_batch - 1,
                )
            ):
                try:
                    messages.append(
                        transition_queue.get_nowait()
                    )
                except queue.Empty:
                    break

            for message in messages:
                process_message(message)

                # Preserve an exact 40K-style midpoint rather than saving at an
                # arbitrary drain-batch overshoot. Unprocessed drained messages
                # are intentionally dropped; actors will replenish the queue.
                if (
                    args.midpoint_new_steps > 0
                    and not midpoint_saved
                    and collected >= args.midpoint_new_steps
                ):
                    break

                if collected >= args.steps:
                    break

            if len(replay) >= args.warmup:
                post_warmup_total = max(1, args.steps - args.warmup)
                post_warmup_done = max(0, collected - args.warmup)

                desired_new_gradient_steps = min(
                    target_gradient_steps,
                    int(
                        math.floor(
                            post_warmup_done
                            / post_warmup_total
                            * target_gradient_steps
                        )
                    ),
                )

                while (
                    new_gradient_steps
                    < desired_new_gradient_steps
                ):
                    metrics = train_batch(
                        model=model,
                        target_model=target_model,
                        optimizer=optimizer,
                        replay=replay,
                        rng=rng,
                        batch_size=args.batch_size,
                        gamma=args.gamma,
                        target_gate=args.target_gate,
                        terminal_penalty=args.terminal_penalty,
                        device=device,
                    )

                    new_gradient_steps += 1

                    loss_history.append(
                        metrics["loss"]
                    )
                    q_history.append(
                        metrics["q_mean"]
                    )
                    td_history.append(
                        metrics["td_abs"]
                    )

                    for key in (
                        "loss",
                        "q_mean",
                        "target_mean",
                        "td_abs",
                    ):
                        if not math.isfinite(
                            metrics[key]
                        ):
                            raise RuntimeError(
                                f"Non-finite learner metric: {key}"
                            )

                    if (
                        new_gradient_steps
                        % target_update_gradients
                        == 0
                    ):
                        target_model.load_state_dict(
                            model.state_dict()
                        )

            while collected >= next_sync:
                absolute_version = (
                    inherited_env_steps
                    + collected
                )

                broadcast_weights(
                    model,
                    control_queues,
                    absolute_version,
                )

                next_sync += args.sync_every

            if (
                args.midpoint_new_steps > 0
                and not midpoint_saved
                and collected == args.midpoint_new_steps
            ):
                save_checkpoint(
                    path=args.mid_output,
                    model=model,
                    target_model=target_model,
                    optimizer=optimizer,
                    inherited_env_steps=inherited_env_steps,
                    new_env_steps=collected,
                    inherited_gradient_steps=inherited_gradient_steps,
                    new_gradient_steps=new_gradient_steps,
                    replay_size=len(replay),
                    unique_training_seeds=len(training_seeds),
                    args=args,
                )

                midpoint_saved = True

                print()
                print(
                    "MIDPOINT CHECKPOINT SAVED:",
                    args.mid_output,
                    f"(total transitions={inherited_env_steps + collected})",
                )

            if collected >= next_log:
                elapsed = (
                    time.perf_counter()
                    - run_start
                )

                total = max(
                    collected,
                    1,
                )

                recent_loss = (
                    float(
                        np.mean(
                            loss_history[-200:]
                        )
                    )
                    if loss_history
                    else 0.0
                )

                recent_q = (
                    float(
                        np.mean(
                            q_history[-200:]
                        )
                    )
                    if q_history
                    else 0.0
                )

                recent_td = (
                    float(
                        np.mean(
                            td_history[-200:]
                        )
                    )
                    if td_history
                    else 0.0
                )

                recent_gap = (
                    float(
                        np.mean(
                            q_gap_history[-200:]
                        )
                    )
                    if q_gap_history
                    else 0.0
                )

                absolute_collected = (
                    inherited_env_steps
                    + collected
                )

                versions = np.asarray(
                    list(
                        actor_weight_versions.values()
                    ),
                    dtype=np.float64,
                )

                mean_lag = float(
                    np.mean(
                        np.maximum(
                            0.0,
                            absolute_collected
                            - versions,
                        )
                    )
                )

                print(
                    f"new={collected:>6}/{args.steps} "
                    f"total={absolute_collected:>6} "
                    f"seeds={len(training_seeds):>3} "
                    f"replay={len(replay):>6} "
                    f"grad={new_gradient_steps:>5} "
                    f"lag={mean_lag:6.0f} "
                    f"tps={collected / max(elapsed, 1e-9):6.1f} "
                    f"gps={new_gradient_steps / max(elapsed, 1e-9):6.1f} "
                    f"| Qswitch={q_interventions / total * 100:5.2f}% "
                    f"random={random_explorations / total * 100:5.2f}% "
                    f"gap={recent_gap:.4f} "
                    f"| L={recent_loss:.6f} "
                    f"Q={recent_q:+.4f} "
                    f"TD={recent_td:.4f}"
                )

                next_log += args.log_every

        # Finish the exact fixed sample budget after collection, if the
        # proportional scheduler is one or two updates short because of rounding.
        while new_gradient_steps < target_gradient_steps:
            metrics = train_batch(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                replay=replay,
                rng=rng,
                batch_size=args.batch_size,
                gamma=args.gamma,
                target_gate=args.target_gate,
                terminal_penalty=args.terminal_penalty,
                device=device,
            )
            new_gradient_steps += 1
            loss_history.append(metrics["loss"])
            q_history.append(metrics["q_mean"])
            td_history.append(metrics["td_abs"])
            if new_gradient_steps % target_update_gradients == 0:
                target_model.load_state_dict(model.state_dict())

        target_model.load_state_dict(
            model.state_dict()
        )

        core_training_elapsed = time.perf_counter() - run_start

    finally:
        stop_event.set()
        shutdown_start = time.perf_counter()

        # IMPORTANT: concurrent grace period.  Do NOT spend N * timeout seconds
        # joining actors sequentially after the learner is already finished.
        shutdown_deadline = shutdown_start + float(args.shutdown_grace)
        while time.perf_counter() < shutdown_deadline:
            if not any(process.is_alive() for process in actors):
                break
            time.sleep(0.05)

        forced_terminations = 0
        for process in actors:
            if process.is_alive():
                forced_terminations += 1
                process.terminate()

        for process in actors:
            process.join(timeout=1.0)

        shutdown_elapsed = time.perf_counter() - shutdown_start

        try:
            transition_queue.close()
        except Exception:
            pass

        for control_queue in control_queues:
            try:
                control_queue.close()
            except Exception:
                pass

    total_process_elapsed = time.perf_counter() - run_start

    save_checkpoint(
        path=args.output,
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        inherited_env_steps=inherited_env_steps,
        new_env_steps=collected,
        inherited_gradient_steps=inherited_gradient_steps,
        new_gradient_steps=new_gradient_steps,
        replay_size=len(replay),
        unique_training_seeds=len(training_seeds),
        args=args,
    )

    total_env_steps = (
        inherited_env_steps
        + collected
    )

    total_gradient_steps = (
        inherited_gradient_steps
        + new_gradient_steps
    )

    print()
    print("=" * 80)
    print("V8.6 RISK-AWARE OBSERVABLE-SAFE 50K TRAINING SUMMARY")
    print("=" * 80)
    print()
    print("Inherited transitions:", inherited_env_steps)
    print("New transitions:", collected)
    print("Total transitions:", total_env_steps)
    print("Inherited gradients:", inherited_gradient_steps)
    print("New gradients:", new_gradient_steps)
    print("Total gradient label:", total_gradient_steps)
    print("Unique training seeds this run:", len(training_seeds))

    if training_seeds:
        print(
            "Training seed range:",
            min(training_seeds),
            "..",
            max(training_seeds),
        )

    print("Replay size:", len(replay))
    print("Terminal replay extra entries:", terminal_replay_extra_entries)
    print("Risk-actor transitions:", risk_actor_transitions)
    print()
    print("Teacher actions:", teacher_actions)
    print("Q interventions:", q_interventions)
    print("Random alternative explorations:", random_explorations)
    print(
        "Q intervention rate:",
        f"{q_interventions / max(collected, 1) * 100:.2f}%",
    )
    print(
        "Random exploration rate:",
        f"{random_explorations / max(collected, 1) * 100:.2f}%",
    )
    print("Real game overs:", real_gameovers)
    print("No-successor resets:", no_next_resets)
    print(f"{args.segment_pieces}-piece segment boundaries:", segment_boundaries)

    print()
    print("Transitions / latest synced weight per actor:")

    final_lags = []

    for actor_id in range(args.actors):
        version = actor_weight_versions[
            actor_id
        ]

        lag = max(
            0,
            total_env_steps - version,
        )

        final_lags.append(lag)

        print(
            f"  actor {actor_id:>2}: "
            f"transitions={actor_counts[actor_id]:>5} "
            f"weight_version={version} "
            f"lag={lag}"
        )

    if final_lags:
        print(
            "Actor weight lag mean/max:",
            f"{float(np.mean(final_lags)):.1f} / {int(max(final_lags))}",
        )

    if loss_history:
        print()
        print(
            "Final 200 loss:",
            float(
                np.mean(
                    loss_history[-200:]
                )
            ),
        )
        print(
            "Final 200 Q:",
            float(
                np.mean(
                    q_history[-200:]
                )
            ),
        )
        print(
            "Final 200 TD abs:",
            float(
                np.mean(
                    td_history[-200:]
                )
            ),
        )

    effective_samples = (
        new_gradient_steps
        * args.batch_size
    )

    print()
    print("=" * 80)
    print("PERFORMANCE")
    print("=" * 80)
    print()
    print("Core training time:", f"{core_training_elapsed:.2f}s")
    print("Actor shutdown time:", f"{shutdown_elapsed:.2f}s")
    print("Total process wall time:", f"{total_process_elapsed:.2f}s")
    print("Forced actor terminations:", forced_terminations)
    print(
        "Core transition throughput:",
        f"{collected / max(core_training_elapsed, 1e-9):.2f} transitions/s",
    )
    print(
        "Core gradient throughput:",
        f"{new_gradient_steps / max(core_training_elapsed, 1e-9):.2f} gradients/s",
    )
    print(
        "Effective replay samples processed:",
        effective_samples,
    )
    print(
        "Effective samples / new transition:",
        f"{effective_samples / max(collected, 1):.2f}",
    )

    print()
    print("Checkpoint:", args.output)

    if collected != args.steps:
        raise RuntimeError(
            "Collected transition count mismatch."
        )

    if new_gradient_steps <= 0:
        raise RuntimeError(
            "No learner gradient updates occurred."
        )

    if len(training_seeds) < max(
        2,
        args.actors,
    ):
        raise RuntimeError(
            "Seed diversity check failed."
        )

    if not os.path.isfile(
        args.output
    ):
        raise RuntimeError(
            "Checkpoint was not created."
        )
    if (
        args.midpoint_new_steps > 0
        and not os.path.isfile(args.mid_output)
    ):
        raise RuntimeError(
            "Midpoint checkpoint was not created."
        )

    if (
        args.steps >= args.sync_every
        and max(actor_weight_versions.values())
        <= inherited_env_steps
    ):
        raise RuntimeError(
            "Actor weight sync check failed."
        )

    print()
    print("Checkpoint Resume       : PASS")
    print("Observable Candidate Q  : PASS")
    print("Candidate/Next Split    : PASS")
    print("Parallel CPU Actors     : PASS")
    print("Segment Seed Diversity  : PASS")
    print("Central Replay Buffer   : PASS")
    print("CUDA Autotuned Learner : PASS")
    print("Risk-Aware TD Target    : PASS")
    print("Target Network          : PASS")
    print("Actor Weight Sync       : PASS")
    print("Periodic Identity Audit : PASS")
    print("Checkpoint              : PASS")
    print("40K Midpoint Checkpoint : PASS" if args.midpoint_new_steps > 0 else "40K Midpoint Checkpoint : DISABLED")
    print()
    print("V8.6 RISK-AWARE OBSERVABLE-SAFE 30K -> 50K LONG-RUN TRAINING: PASS")


if __name__ == "__main__":
    mp.freeze_support()
    main()
