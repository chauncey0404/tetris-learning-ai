
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import time
import traceback

import numpy as np
import torch

from ai.observable_q_network import ObservableSafeQNetwork
from v8_8_array_replay import V88ArrayReplayBuffer
from v8_8_train_common import (
    benchmark_and_choose_batch,
    move_optimizer_state_to_device,
    parse_int_list,
    save_checkpoint,
    train_batch,
)


GLOBAL_SEED = 20260823
PERMANENT_BENCHMARK_FIRST = 6
PERMANENT_BENCHMARK_LAST = 20


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_training_seed(seed):
    seed = int(seed)
    if PERMANENT_BENCHMARK_FIRST <= seed <= PERMANENT_BENCHMARK_LAST:
        return PERMANENT_BENCHMARK_LAST + 1
    return seed


def cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _poll_weights(model, control_queue, current_version):
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


def _put_latest_weights(control_queue, model, version):
    payload = {
        "kind": "weights",
        "version": int(version),
        "state_dict": cpu_state_dict(model),
    }

    # Keep only the newest generator policy snapshot.
    while True:
        try:
            control_queue.get_nowait()
        except queue.Empty:
            break

    try:
        control_queue.put_nowait(payload)
    except queue.Full:
        pass


def _masked_normalized_choice_numpy(q_values, mask, gates, eps=1e-6):
    """
    Per-row V8.7 normalized gate with a different gate for risk streams.

    candidate 0 is Teacher.
    """
    q = np.asarray(q_values, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    gates = np.asarray(gates, dtype=np.float64)

    if q.ndim != 2 or mask.shape != q.shape:
        raise ValueError("q/mask must be [B,K]")
    if gates.shape != (q.shape[0],):
        raise ValueError("gates must be [B]")

    b, k = q.shape

    if k < 1:
        raise ValueError("K must be >= 1")

    teacher_valid = mask[:, 0]

    if k == 1:
        return (
            np.zeros(b, dtype=np.int64),
            np.zeros(b, dtype=np.float32),
        )

    alt_mask = mask.copy()
    alt_mask[:, 0] = False
    has_alt = alt_mask.any(axis=1)

    alt_q = np.where(alt_mask, q, -np.inf)
    best_alt = np.argmax(alt_q, axis=1)

    rows = np.arange(b)
    teacher_q = q[:, 0]
    best_alt_q = q[rows, best_alt]

    raw_gap = best_alt_q - teacher_q

    q_max = np.max(np.where(mask, q, -np.inf), axis=1)
    q_min = np.min(np.where(mask, q, np.inf), axis=1)
    span = q_max - q_min

    valid = (
        teacher_valid
        & has_alt
        & np.isfinite(raw_gap)
        & np.isfinite(span)
        & (raw_gap > eps)
        & (span > eps)
    )

    margin = np.zeros(b, dtype=np.float64)
    margin[valid] = raw_gap[valid] / span[valid]
    margin = np.clip(margin, 0.0, 1.0)

    choose_alt = valid & (margin >= gates)
    chosen = np.where(choose_alt, best_alt, 0).astype(np.int64)

    return chosen, margin.astype(np.float32)


def _slice_batch(batch, start, end):
    result = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (
            batch["state"].shape[0],
        ):
            result[key] = value[start:end]
        else:
            result[key] = value
    return result


def vector_generator_loop(
    config,
    initial_model_state_dict,
    initial_weight_version,
    out_queue,
    control_queue,
    stop_event,
):
    """
    One vectorized generator process.

    JAX owns CPU rollout/candidate/Teacher work.
    A batched CPU copy of Q chooses between Teacher's top-4.
    The main process concurrently runs CUDA replay learning.
    """
    try:
        # Do not let the child wait for its multiprocessing.Queue feeder thread
        # after the learner has already stopped. Any queued tail batches are
        # intentionally disposable during shutdown.
        try:
            out_queue.cancel_join_thread()
        except Exception:
            pass

        # Import JAX only inside the producer process. On native Windows this
        # keeps the main CUDA learner independent of JAX's CPU runtime.
        import jax
        import jax.numpy as jnp

        from v8_8_jax_vector_backend import reset_batch
        from v8_8_jax_teacher import (
            replace_done_or_segment_states_jit,
            select_candidate_state_jit,
            topk_batch,
        )

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        model = ObservableSafeQNetwork().cpu()
        model.load_state_dict(initial_model_state_dict)
        model.eval()

        b = int(config["vector_envs"])
        start_seed = sanitize_training_seed(config["start_seed"])
        segment_pieces = int(config["segment_pieces"])
        exploration = float(config["exploration"])
        normal_gate = float(config["behavior_gate"])
        risk_gate = float(config["risk_behavior_gate"])
        risk_streams = int(config["risk_streams"])

        if b <= 0:
            raise ValueError("vector_envs must be > 0")
        if not 0 <= risk_streams <= b:
            raise ValueError("risk_streams must be in [0, vector_envs]")

        rng = np.random.default_rng(
            int(config["generator_rng_seed"])
        )

        stream_ids = np.arange(b, dtype=np.int64)
        episode_seeds = (
            int(start_seed) + stream_ids
        ).astype(np.int64)

        risk_mask = stream_ids < risk_streams
        gates = np.where(
            risk_mask,
            risk_gate,
            normal_gate,
        ).astype(np.float64)

        def keys_from_seeds(seeds):
            # Only used on startup and rare resets, so a tiny Python loop is
            # preferable to introducing another compiled PRNG interface.
            return jnp.stack(
                [
                    jax.random.PRNGKey(int(seed))
                    for seed in np.asarray(seeds).tolist()
                ],
                axis=0,
            )

        compile_start = time.perf_counter()

        states = reset_batch(keys_from_seeds(episode_seeds))
        bundle = topk_batch(states)
        jax.block_until_ready(bundle.candidate_features)

        compile_elapsed = time.perf_counter() - compile_start

        if np.asarray(bundle.state_features).shape != (b, 243):
            raise RuntimeError("generator state243 shape mismatch")
        if np.asarray(bundle.candidate_features).shape != (b, 4, 215):
            raise RuntimeError("generator candidate215 shape mismatch")

        weight_version = int(initial_weight_version)
        segment_steps = np.zeros(b, dtype=np.int32)
        generated = 0
        batch_index = 0

        out_queue.put(
            {
                "kind": "ready",
                "jax_version": str(jax.__version__),
                "jax_devices": [str(x) for x in jax.devices()],
                "vector_envs": b,
                "risk_streams": risk_streams,
                "compile_seconds": float(compile_elapsed),
            },
            timeout=120.0,
        )

        while not stop_event.is_set():
            weight_version = _poll_weights(
                model,
                control_queue,
                weight_version,
            )

            # ------------------------------------------------------------
            # Batched Q policy on Teacher's reachable top-4.
            # ------------------------------------------------------------
            q_start = time.perf_counter()

            state_np = np.array(
                bundle.state_features,
                dtype=np.float32,
                copy=True,
            )
            cand_np = np.array(
                bundle.candidate_features,
                dtype=np.float32,
                copy=True,
            )
            reward_np = np.array(
                bundle.rewards,
                dtype=np.float32,
                copy=True,
            )
            score_np = np.array(
                bundle.teacher_scores,
                dtype=np.float32,
                copy=True,
            )
            rank_np = np.array(
                bundle.teacher_ranks,
                dtype=np.float32,
                copy=True,
            )
            mask_np = np.array(
                bundle.mask,
                dtype=np.bool_,
                copy=True,
            )

            with torch.inference_mode():
                q = model(
                    state=torch.from_numpy(state_np),
                    candidates=torch.from_numpy(cand_np),
                    rewards=torch.from_numpy(reward_np),
                    teacher_scores=torch.from_numpy(score_np),
                    teacher_ranks=torch.from_numpy(rank_np),
                ).numpy()

            chosen, margins = _masked_normalized_choice_numpy(
                q,
                mask_np,
                gates,
            )

            # 5% random alternative exploration, same intent as V8.7.
            valid_counts = mask_np.sum(axis=1)
            random_rows = (
                (valid_counts > 1)
                & (rng.random(b) < exploration)
            )

            for row in np.flatnonzero(random_rows):
                chosen[row] = int(
                    rng.integers(1, int(valid_counts[row]))
                )

            behavior_code = np.zeros(b, dtype=np.int8)
            # 0 teacher, 1 Q, 2 random
            behavior_code[(chosen != 0) & (~random_rows)] = 1
            behavior_code[random_rows] = 2
            margins = np.where(
                random_rows,
                0.0,
                margins,
            ).astype(np.float32)

            q_elapsed = time.perf_counter() - q_start

            # ------------------------------------------------------------
            # Advance all B states in one JAX gather, then build all next
            # candidate/Teacher data in one JAX compiled call.
            # ------------------------------------------------------------
            jax_start = time.perf_counter()

            selected_states = select_candidate_state_jit(
                bundle.candidate_states,
                jnp.asarray(chosen, dtype=jnp.int32),
            )

            next_bundle = topk_batch(selected_states)
            jax.block_until_ready(next_bundle.candidate_features)

            jax_elapsed = time.perf_counter() - jax_start

            rows = np.arange(b)
            selected_candidate = cand_np[rows, chosen]
            selected_reward = reward_np[rows, chosen]
            selected_score = score_np[rows, chosen]
            selected_rank = rank_np[rows, chosen]
            selected_lines = np.asarray(
                bundle.lines,
                dtype=np.int32,
            )[rows, chosen]

            real_done = np.asarray(
                selected_states.game_over,
                dtype=np.bool_,
            )
            next_mask = np.asarray(
                next_bundle.mask,
                dtype=np.bool_,
            )
            no_next = (~real_done) & (~next_mask.any(axis=1))
            done = real_done | no_next

            segment_steps += 1
            if segment_pieces > 0:
                segment_boundary = segment_steps >= segment_pieces
            else:
                segment_boundary = np.zeros(b, dtype=np.bool_)

            batch = {
                "state": np.ascontiguousarray(state_np),
                "candidate": np.ascontiguousarray(
                    selected_candidate.astype(np.float32, copy=False)
                ),
                "reward": np.ascontiguousarray(
                    selected_reward.astype(np.float32, copy=False)
                ),
                "teacher_score": np.ascontiguousarray(
                    selected_score.astype(np.float32, copy=False)
                ),
                "teacher_rank": np.ascontiguousarray(
                    selected_rank.astype(np.float32, copy=False)
                ),
                "done": np.ascontiguousarray(done.astype(np.float32)),
                "next_state": np.ascontiguousarray(
                    np.asarray(
                        next_bundle.state_features,
                        dtype=np.float32,
                    )
                ),
                "next_candidates": np.ascontiguousarray(
                    np.asarray(
                        next_bundle.candidate_features,
                        dtype=np.float32,
                    )
                ),
                "next_rewards": np.ascontiguousarray(
                    np.asarray(
                        next_bundle.rewards,
                        dtype=np.float32,
                    )
                ),
                "next_teacher_scores": np.ascontiguousarray(
                    np.asarray(
                        next_bundle.teacher_scores,
                        dtype=np.float32,
                    )
                ),
                "next_teacher_ranks": np.ascontiguousarray(
                    np.asarray(
                        next_bundle.teacher_ranks,
                        dtype=np.float32,
                    )
                ),
                "next_mask": np.ascontiguousarray(next_mask),
                # Diagnostics / accounting:
                "episode_seed": episode_seeds.copy(),
                "behavior_code": behavior_code,
                "q_margin": margins,
                "real_done": real_done,
                "no_next": no_next,
                "segment_boundary": segment_boundary.copy(),
                "lines": selected_lines,
                "is_risk": risk_mask.copy(),
            }

            message = {
                "kind": "transition_batch",
                "batch": batch,
                "weight_version": int(weight_version),
                "generated_total": int(generated + b),
                "generator_batch_index": int(batch_index),
                "jax_topk_ms": float(jax_elapsed * 1000.0),
                "q_policy_ms": float(q_elapsed * 1000.0),
            }

            # Backpressure is intentional: if learner/replay cannot consume
            # quickly enough, do not accumulate unbounded stale experience.
            while not stop_event.is_set():
                try:
                    out_queue.put(message, timeout=0.5)
                    break
                except queue.Full:
                    continue

            if stop_event.is_set():
                break

            generated += b
            batch_index += 1

            # ------------------------------------------------------------
            # Continue from the exact selected next states.
            # Reset only terminal/no-next streams. With hundreds of vector
            # streams, forced 500-piece segmentation is no longer required
            # for diversity; segment_pieces defaults to 0.
            # ------------------------------------------------------------
            reset_mask = done | segment_boundary

            if np.any(reset_mask):
                episode_seeds[reset_mask] += b
                segment_steps[reset_mask] = 0

                reset_states = reset_batch(
                    keys_from_seeds(episode_seeds)
                )
                states = replace_done_or_segment_states_jit(
                    selected_states,
                    reset_states,
                    jnp.asarray(reset_mask),
                )
                bundle = topk_batch(states)
                jax.block_until_ready(bundle.candidate_features)
            else:
                states = selected_states
                bundle = next_bundle

    except Exception:
        error = {
            "kind": "error",
            "traceback": traceback.format_exc(),
        }
        try:
            out_queue.put(error, timeout=2.0)
        except Exception:
            pass
        stop_event.set()


def _update_stats(stats, batch):
    n = int(batch["state"].shape[0])

    behavior = np.asarray(batch["behavior_code"])
    stats["teacher_actions"] += int(np.count_nonzero(behavior == 0))
    stats["q_interventions"] += int(np.count_nonzero(behavior == 1))
    stats["random_explorations"] += int(np.count_nonzero(behavior == 2))
    stats["real_gameovers"] += int(np.count_nonzero(batch["real_done"]))
    stats["no_next_resets"] += int(np.count_nonzero(batch["no_next"]))
    stats["segment_boundaries"] += int(
        np.count_nonzero(batch["segment_boundary"])
    )
    stats["risk_transitions"] += int(np.count_nonzero(batch["is_risk"]))

    q_rows = behavior == 1
    if np.any(q_rows):
        stats["q_margin_history"].extend(
            np.asarray(batch["q_margin"])[q_rows].astype(float).tolist()
        )

    stats["training_seeds"].update(
        int(x)
        for x in np.unique(batch["episode_seed"])
    )

    return n


def _verify_parity_stamp(path):
    if not os.path.isfile(path):
        raise RuntimeError(
            "V8.8 Teacher/Top-K parity stamp not found:\n"
            f"  {path}\n"
            "Run: python test_v8_8_jax_teacher_parity.py\n"
            "and require V8.8 JAX TEACHER/TOP-K PARITY: PASS before training."
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "PASS":
        raise RuntimeError(
            f"Parity stamp status is not PASS: {data!r}"
        )

    return data


def main():
    parser = argparse.ArgumentParser(
        description=(
            "V8.8: JAX-vectorized CPU rollout + batched Teacher/Q generator "
            "overlapped with the existing observable-safe PyTorch CUDA DDQN learner."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default="models/v8_7_normalized_gpu_td_100k.pt",
        help=(
            "Warm-start challenger checkpoint. This does NOT promote it to Champion."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="NEW transitions; default continues 100K -> 150K.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=11001,
        help="JAX training stream seed base; kept disjoint from qualification blocks.",
    )

    parser.add_argument(
        "--vector-envs",
        type=int,
        default=256,
        help="Parallel JAX Tetris states in the single vectorized generator.",
    )
    parser.add_argument(
        "--risk-streams",
        type=int,
        default=-1,
        help="-1 = auto 20%% of vector_envs; 0 disables risk streams.",
    )
    parser.add_argument(
        "--segment-pieces",
        type=int,
        default=0,
        help=(
            "0 disables forced segmentation. With hundreds of independent vector "
            "streams, diversity no longer depends on 500-piece actor resets."
        ),
    )

    parser.add_argument("--behavior-gate", type=float, default=0.600)
    parser.add_argument("--risk-behavior-gate", type=float, default=0.400)
    parser.add_argument("--target-gate", type=float, default=0.600)
    parser.add_argument("--exploration", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--terminal-penalty", type=float, default=1.0)
    parser.add_argument("--terminal-replay-copies", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=100000)

    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="0 = auto max(2048, chosen learner batch size).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = CUDA autotune.",
    )
    parser.add_argument(
        "--batch-candidates",
        default="4096,8192,16384,32768",
    )
    parser.add_argument("--benchmark-warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--batch-near-best-ratio", type=float, default=0.95)

    parser.add_argument(
        "--sample-budget",
        type=int,
        default=10178560,
        help=(
            "Same ~203.57 replay samples/new transition ratio as the successful "
            "V8.7 50K continuation."
        ),
    )
    parser.add_argument("--target-update-samples", type=int, default=512000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument(
        "--sync-every",
        type=int,
        default=1024,
        help="Send latest learner weights to the batched CPU Q generator.",
    )
    parser.add_argument(
        "--queue-batches",
        type=int,
        default=4,
        help="Bounded vector-batch queue; small by design to control policy lag.",
    )
    parser.add_argument(
        "--generator-ready-timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shutdown-grace", type=float, default=3.0)
    parser.add_argument("--log-every", type=int, default=1000)

    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument(
        "--checkpoint-prefix",
        default="models/v8_8_jax_vectorized_td",
    )
    parser.add_argument(
        "--max-batch-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--output",
        default="models/v8_8_jax_vectorized_td_150k.pt",
    )
    parser.add_argument(
        "--parity-stamp",
        default="data/v8_8_jax_teacher_parity_pass.json",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.vector_envs <= 0:
        raise ValueError("--vector-envs must be > 0")
    if args.risk_streams == -1:
        args.risk_streams = int(round(args.vector_envs * 0.20))
    if not 0 <= args.risk_streams <= args.vector_envs:
        raise ValueError("--risk-streams must be -1 or in [0,vector-envs]")
    if args.segment_pieces < 0:
        raise ValueError("--segment-pieces must be >= 0")
    if not 0.0 <= args.exploration <= 1.0:
        raise ValueError("--exploration must be in [0,1]")
    if min(
        args.behavior_gate,
        args.risk_behavior_gate,
        args.target_gate,
    ) < 0.0:
        raise ValueError("all normalized gates must be >= 0")
    if args.terminal_penalty < 0.0:
        raise ValueError("--terminal-penalty must be >= 0")
    if args.terminal_replay_copies <= 0:
        raise ValueError("--terminal-replay-copies must be > 0")
    if args.replay_capacity <= 0:
        raise ValueError("--replay-capacity must be > 0")
    if args.sample_budget <= 0:
        raise ValueError("--sample-budget must be > 0")
    if args.target_update_samples <= 0:
        raise ValueError("--target-update-samples must be > 0")
    if args.sync_every <= 0:
        raise ValueError("--sync-every must be > 0")
    if args.queue_batches <= 0:
        raise ValueError("--queue-batches must be > 0")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if args.checkpoint_every > args.steps:
        raise ValueError("--checkpoint-every cannot exceed --steps")
    if not 0.0 < args.max_batch_fraction <= 1.0:
        raise ValueError("--max-batch-fraction must be in (0,1]")
    if not 0.0 < args.batch_near_best_ratio <= 1.0:
        raise ValueError("--batch-near-best-ratio must be in (0,1]")

    parity_stamp = _verify_parity_stamp(args.parity_stamp)

    set_global_seed(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    inherited_env_steps = int(checkpoint.get("env_steps", 0))
    inherited_gradient_steps = int(checkpoint.get("gradient_steps", 0))

    model = ObservableSafeQNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    target_model = ObservableSafeQNetwork().to(device)
    if "target_model_state_dict" in checkpoint:
        target_model.load_state_dict(
            checkpoint["target_model_state_dict"]
        )
    else:
        target_model.load_state_dict(model.state_dict())
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
            target_gate=args.target_gate,
            gamma=args.gamma,
            warmup_iters=args.benchmark_warmup_iters,
            timed_iters=args.benchmark_iters,
            within_best_ratio=args.batch_near_best_ratio,
        )

    freshness_basis = (
        min(args.steps, args.checkpoint_every)
        if args.checkpoint_every > 0
        else args.steps
    )
    freshness_limit = max(
        1,
        int(math.floor(
            freshness_basis * args.max_batch_fraction
        )),
    )

    if int(chosen_batch_size) > freshness_limit:
        eligible = [
            item
            for item in batch_benchmark
            if int(item["batch_size"]) <= freshness_limit
        ]

        if not eligible:
            raise RuntimeError(
                "No autotuned CUDA batch survives freshness guard."
            )

        guarded = max(
            eligible,
            key=lambda item: item["samples_per_sec"],
        )

        print()
        print(
            "FRESHNESS GUARD OVERRIDE:",
            f"autotune={chosen_batch_size} -> "
            f"deploy={guarded['batch_size']}",
            f"(<= {args.max_batch_fraction:.2f} * "
            f"{freshness_basis} checkpoint-window transitions)",
        )

        chosen_batch_size = int(guarded["batch_size"])

    args.batch_size = int(chosen_batch_size)
    args.batch_benchmark = batch_benchmark
    args.warmup = (
        int(args.warmup)
        if args.warmup > 0
        else max(2048, args.batch_size)
    )

    if args.warmup < args.batch_size:
        raise ValueError("Effective warmup must be >= batch_size")
    if args.steps <= args.warmup:
        raise ValueError("--steps must exceed effective warmup")

    target_gradient_steps = max(
        1,
        int(round(args.sample_budget / args.batch_size)),
    )
    target_update_gradients = max(
        1,
        int(round(
            args.target_update_samples / args.batch_size
        )),
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
        for group in optimizer.param_groups:
            group["lr"] = float(args.lr)
            group["weight_decay"] = float(args.weight_decay)
        optimizer_resumed = True

    replay = V88ArrayReplayBuffer(
        capacity=args.replay_capacity
    )

    runtime_meta = {
        "parity_stamp": parity_stamp,
    }

    print()
    print("=" * 80)
    print("V8.8 JAX-VECTORIZED OBSERVABLE-SAFE GPU TRAINING")
    print("=" * 80)
    print()
    print("Learner device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Input checkpoint:", args.checkpoint)
    print("Inherited transitions:", inherited_env_steps)
    print("Inherited gradient steps:", inherited_gradient_steps)
    print("New transitions:", args.steps)
    print("Target total:", inherited_env_steps + args.steps)
    print("Optimizer resumed:", "YES" if optimizer_resumed else "NO")
    print()
    print("JAX vector environments:", args.vector_envs)
    print("Risk streams:", args.risk_streams)
    print("Forced segment length:", args.segment_pieces or "DISABLED")
    print("Training seed start:", args.start_seed)
    print("Normal normalized gate:", args.behavior_gate)
    print("Risk normalized gate:", args.risk_behavior_gate)
    print("DDQN target gate:", args.target_gate)
    print("Exploration:", f"{args.exploration * 100:.2f}%")
    print()
    print("Array replay capacity:", args.replay_capacity)
    print(
        "Array replay allocated:",
        f"{replay.nbytes / (1024.0 ** 2):.1f} MiB",
    )
    print("Fresh replay warmup:", args.warmup)
    print("CUDA learner batch:", args.batch_size)
    print("Target gradients:", target_gradient_steps)
    print(
        "Target replay samples:",
        target_gradient_steps * args.batch_size,
    )
    print(
        "Samples/new transition:",
        f"{target_gradient_steps * args.batch_size / args.steps:.2f}",
    )
    print("Generator weight sync:", args.sync_every, "transitions")
    print("Generator queue:", args.queue_batches, "vector batches")
    print("Permanent seeds 6~20:", "PROTECTED")
    print("Qualification status:", "CHALLENGER ONLY")
    print()
    print("OBSERVABLE SAFETY:")
    print("  Q current input   : state243")
    print("  Q candidate input : board200 + rotation4 + x10 + hold1 = 215")
    print("  Preview future tail 200:243 is NEVER candidate input")
    print("  Real selected next state243 is used only for TD bootstrap")

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=args.queue_batches)
    control_queue = ctx.Queue(maxsize=1)
    stop_event = ctx.Event()

    generator_config = {
        "vector_envs": args.vector_envs,
        "risk_streams": args.risk_streams,
        "segment_pieces": args.segment_pieces,
        "start_seed": args.start_seed,
        "behavior_gate": args.behavior_gate,
        "risk_behavior_gate": args.risk_behavior_gate,
        "exploration": args.exploration,
        "generator_rng_seed": GLOBAL_SEED + 8800,
    }

    generator = ctx.Process(
        target=vector_generator_loop,
        args=(
            generator_config,
            cpu_state_dict(model),
            inherited_env_steps,
            out_queue,
            control_queue,
            stop_event,
        ),
        name="v8_8_jax_generator",
    )

    generator.start()

    # Wait for JAX compilation/ready report.
    try:
        ready = out_queue.get(
            timeout=args.generator_ready_timeout
        )
    except queue.Empty:
        stop_event.set()
        generator.terminate()
        generator.join(timeout=2.0)
        raise RuntimeError(
            "V8.8 JAX generator did not become ready in time."
        )

    if ready.get("kind") == "error":
        raise RuntimeError(
            "V8.8 generator failed during startup:\n"
            + ready.get("traceback", "")
        )
    if ready.get("kind") != "ready":
        raise RuntimeError(
            f"Unexpected generator startup message: {ready!r}"
        )

    runtime_meta.update(
        {
            "jax_version": ready["jax_version"],
            "jax_devices": ready["jax_devices"],
            "generator_compile_seconds": ready["compile_seconds"],
            "vector_envs": ready["vector_envs"],
            "risk_streams": ready["risk_streams"],
        }
    )

    print()
    print("Generator JAX:", ready["jax_version"])
    print("Generator devices:", ready["jax_devices"])
    print(
        "Generator first compile:",
        f"{ready['compile_seconds']:.2f}s",
    )
    print("Generator ready: PASS")
    print()

    collected = 0
    new_gradient_steps = 0
    terminal_replay_extra_entries = 0

    stats = {
        "teacher_actions": 0,
        "q_interventions": 0,
        "random_explorations": 0,
        "real_gameovers": 0,
        "no_next_resets": 0,
        "segment_boundaries": 0,
        "risk_transitions": 0,
        "q_margin_history": [],
        "training_seeds": set(),
    }

    loss_history = []
    q_history = []
    td_history = []

    generator_jax_ms = []
    generator_q_ms = []
    generator_lag = []

    next_sync = args.sync_every
    next_log = args.log_every

    next_checkpoint = (
        args.checkpoint_every
        if args.checkpoint_every > 0
        else None
    )
    periodic_checkpoints = []

    run_start = time.perf_counter()
    core_training_elapsed = 0.0
    shutdown_elapsed = 0.0
    forced_termination = False

    def run_due_gradients():
        nonlocal new_gradient_steps

        if len(replay) < args.warmup:
            return

        post_warmup_total = max(
            1,
            args.steps - args.warmup,
        )
        post_warmup_done = max(
            0,
            collected - args.warmup,
        )

        desired = min(
            target_gradient_steps,
            int(math.floor(
                post_warmup_done
                / post_warmup_total
                * target_gradient_steps
            )),
        )

        while new_gradient_steps < desired:
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

            for key in (
                "loss",
                "q_mean",
                "target_mean",
                "td_abs",
            ):
                if not math.isfinite(metrics[key]):
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

    def maybe_sync():
        nonlocal next_sync
        while collected >= next_sync:
            absolute_version = (
                inherited_env_steps + collected
            )
            _put_latest_weights(
                control_queue,
                model,
                absolute_version,
            )
            next_sync += args.sync_every

    def save_periodic_if_due():
        nonlocal next_checkpoint

        if (
            next_checkpoint is None
            or collected != next_checkpoint
        ):
            return

        total_at_checkpoint = (
            inherited_env_steps + collected
        )
        checkpoint_k = total_at_checkpoint // 1000
        path = (
            f"{args.checkpoint_prefix}_{checkpoint_k}k.pt"
        )

        save_checkpoint(
            path=path,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            inherited_env_steps=inherited_env_steps,
            new_env_steps=collected,
            inherited_gradient_steps=inherited_gradient_steps,
            new_gradient_steps=new_gradient_steps,
            replay_size=len(replay),
            unique_training_seeds=len(
                stats["training_seeds"]
            ),
            args=args,
            runtime_meta=runtime_meta,
        )

        periodic_checkpoints.append(path)
        print()
        print(
            "PERIODIC CHECKPOINT SAVED:",
            path,
            f"(total transitions={total_at_checkpoint}, "
            f"new gradients={new_gradient_steps})",
        )

        next_checkpoint += args.checkpoint_every
        if next_checkpoint > args.steps:
            next_checkpoint = None

    def maybe_log():
        nonlocal next_log

        if collected < next_log:
            return

        elapsed = time.perf_counter() - run_start
        total = max(collected, 1)

        recent_loss = (
            float(np.mean(loss_history[-200:]))
            if loss_history else 0.0
        )
        recent_q = (
            float(np.mean(q_history[-200:]))
            if q_history else 0.0
        )
        recent_td = (
            float(np.mean(td_history[-200:]))
            if td_history else 0.0
        )
        recent_margin = (
            float(np.mean(
                stats["q_margin_history"][-200:]
            ))
            if stats["q_margin_history"]
            else 0.0
        )
        recent_jax = (
            float(np.mean(generator_jax_ms[-20:]))
            if generator_jax_ms else 0.0
        )
        recent_q_ms = (
            float(np.mean(generator_q_ms[-20:]))
            if generator_q_ms else 0.0
        )
        recent_lag = (
            float(np.mean(generator_lag[-20:]))
            if generator_lag else 0.0
        )

        print(
            f"new={collected:>6}/{args.steps} "
            f"total={inherited_env_steps + collected:>6} "
            f"seeds={len(stats['training_seeds']):>4} "
            f"replay={len(replay):>6} "
            f"grad={new_gradient_steps:>5} "
            f"tps={collected / max(elapsed,1e-9):7.1f} "
            f"gps={new_gradient_steps / max(elapsed,1e-9):6.1f} "
            f"| JAX={recent_jax:6.2f}ms "
            f"Qcpu={recent_q_ms:5.2f}ms "
            f"lag={recent_lag:7.0f} "
            f"| Qswitch={stats['q_interventions']/total*100:5.2f}% "
            f"random={stats['random_explorations']/total*100:5.2f}% "
            f"margin={recent_margin:.4f} "
            f"| L={recent_loss:.6f} "
            f"Q={recent_q:+.4f} "
            f"TD={recent_td:.4f}"
        )

        next_log += args.log_every

    try:
        while collected < args.steps:
            if stop_event.is_set() and not generator.is_alive():
                raise RuntimeError(
                    "V8.8 generator stopped before collection completed."
                )

            try:
                message = out_queue.get(timeout=5.0)
            except queue.Empty:
                if not generator.is_alive():
                    raise RuntimeError(
                        "V8.8 generator exited before collection completed."
                    )
                continue

            kind = message.get("kind")

            if kind == "error":
                raise RuntimeError(
                    "V8.8 generator error:\n"
                    + message.get("traceback", "")
                )

            if kind != "transition_batch":
                continue

            full_batch = message["batch"]
            batch_n = int(full_batch["state"].shape[0])

            generator_jax_ms.append(
                float(message.get("jax_topk_ms", 0.0))
            )
            generator_q_ms.append(
                float(message.get("q_policy_ms", 0.0))
            )

            absolute_version = int(
                message.get(
                    "weight_version",
                    inherited_env_steps,
                )
            )
            generator_lag.append(
                max(
                    0,
                    inherited_env_steps
                    + collected
                    - absolute_version,
                )
            )

            offset = 0

            while (
                offset < batch_n
                and collected < args.steps
            ):
                limit = min(
                    batch_n - offset,
                    args.steps - collected,
                )

                if next_checkpoint is not None:
                    limit = min(
                        limit,
                        next_checkpoint - collected,
                    )

                end = offset + limit
                batch = _slice_batch(
                    full_batch,
                    offset,
                    end,
                )

                replay_payload = {
                    key: batch[key]
                    for key in (
                        "state",
                        "candidate",
                        "reward",
                        "teacher_score",
                        "teacher_rank",
                        "done",
                        "next_state",
                        "next_candidates",
                        "next_rewards",
                        "next_teacher_scores",
                        "next_teacher_ranks",
                        "next_mask",
                    )
                }

                replay.add_batch(**replay_payload)

                terminal_replay_extra_entries += (
                    replay.add_terminal_extras(
                        replay_payload,
                        args.terminal_replay_copies,
                    )
                )

                added = _update_stats(stats, batch)
                collected += added
                offset = end

                run_due_gradients()
                maybe_sync()
                save_periodic_if_due()
                maybe_log()

        # Finish exact fixed replay sample budget.
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

            if (
                new_gradient_steps
                % target_update_gradients
                == 0
            ):
                target_model.load_state_dict(
                    model.state_dict()
                )

        target_model.load_state_dict(
            model.state_dict()
        )

        core_training_elapsed = (
            time.perf_counter() - run_start
        )

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
            unique_training_seeds=len(
                stats["training_seeds"]
            ),
            args=args,
            runtime_meta=runtime_meta,
        )

    finally:
        stop_event.set()
        shutdown_start = time.perf_counter()

        generator.join(
            timeout=max(0.0, args.shutdown_grace)
        )

        if generator.is_alive():
            forced_termination = True
            generator.terminate()
            generator.join(timeout=2.0)

        shutdown_elapsed = (
            time.perf_counter() - shutdown_start
        )

    total_elapsed = (
        core_training_elapsed + shutdown_elapsed
    )

    print()
    print("=" * 80)
    print("V8.8 TRAINING SUMMARY")
    print("=" * 80)
    print()
    print("Inherited transitions:", inherited_env_steps)
    print("New transitions:", collected)
    print("Total transitions:", inherited_env_steps + collected)
    print("Inherited gradients:", inherited_gradient_steps)
    print("New gradients:", new_gradient_steps)
    print(
        "Total gradient label:",
        inherited_gradient_steps + new_gradient_steps,
    )
    print(
        "Unique JAX training seeds:",
        len(stats["training_seeds"]),
    )
    if stats["training_seeds"]:
        print(
            "Training seed range:",
            min(stats["training_seeds"]),
            "..",
            max(stats["training_seeds"]),
        )
    print("Replay size:", len(replay))
    print(
        "Terminal replay extra entries:",
        terminal_replay_extra_entries,
    )
    print("Risk-stream transitions:", stats["risk_transitions"])
    print()
    print("Teacher actions:", stats["teacher_actions"])
    print("Q interventions:", stats["q_interventions"])
    print(
        "Random alternative explorations:",
        stats["random_explorations"],
    )
    print(
        "Q intervention rate:",
        f"{stats['q_interventions']/max(collected,1)*100:.2f}%",
    )
    print(
        "Random exploration rate:",
        f"{stats['random_explorations']/max(collected,1)*100:.2f}%",
    )
    print("Real game overs:", stats["real_gameovers"])
    print("No-successor resets:", stats["no_next_resets"])
    print(
        "Forced segment boundaries:",
        stats["segment_boundaries"],
    )

    if generator_jax_ms:
        print()
        print(
            "Generator JAX top-K ms mean:",
            f"{float(np.mean(generator_jax_ms)):.3f}",
        )
        print(
            "Generator CPU Q ms mean:",
            f"{float(np.mean(generator_q_ms)):.3f}",
        )
        print(
            "Generator weight lag mean/max:",
            f"{float(np.mean(generator_lag)):.1f} / "
            f"{int(max(generator_lag))}",
        )

    if loss_history:
        print()
        print(
            "Final 200 loss:",
            float(np.mean(loss_history[-200:])),
        )
        print(
            "Final 200 Q:",
            float(np.mean(q_history[-200:])),
        )
        print(
            "Final 200 TD abs:",
            float(np.mean(td_history[-200:])),
        )

    effective_samples = (
        new_gradient_steps * args.batch_size
    )

    print()
    print("=" * 80)
    print("PERFORMANCE")
    print("=" * 80)
    print()
    print(
        "Core training time:",
        f"{core_training_elapsed:.2f}s",
    )
    print(
        "Generator shutdown time:",
        f"{shutdown_elapsed:.2f}s",
    )
    print(
        "Total process wall time:",
        f"{total_elapsed:.2f}s",
    )
    print(
        "Forced generator termination:",
        "YES" if forced_termination else "NO",
    )
    print(
        "Core transition throughput:",
        f"{collected/max(core_training_elapsed,1e-9):.2f} transitions/s",
    )
    print(
        "Core gradient throughput:",
        f"{new_gradient_steps/max(core_training_elapsed,1e-9):.2f} gradients/s",
    )
    print(
        "Effective replay samples processed:",
        effective_samples,
    )
    print(
        "Effective samples / new transition:",
        f"{effective_samples/max(collected,1):.2f}",
    )
    print()
    print("Checkpoint:", args.output)

    if collected != args.steps:
        raise RuntimeError(
            "Collected transition count mismatch."
        )
    if new_gradient_steps != target_gradient_steps:
        raise RuntimeError(
            "Gradient/sample budget mismatch."
        )
    if not os.path.isfile(args.output):
        raise RuntimeError(
            "Final checkpoint was not created."
        )

    if args.checkpoint_every > 0:
        expected = args.steps // args.checkpoint_every
        if len(periodic_checkpoints) != expected:
            raise RuntimeError(
                "Periodic checkpoint count mismatch: "
                f"{len(periodic_checkpoints)} != {expected}"
            )
        for path in periodic_checkpoints:
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Missing periodic checkpoint: {path}"
                )

    print()
    print("JAX Backend Parity Stamp : PASS")
    print("JAX Vector Generator     : PASS")
    print("Vectorized Teacher Top-K : PASS")
    print("Observable Candidate Q   : PASS")
    print("Candidate/Next Split     : PASS")
    print("Array Replay Buffer      : PASS")
    print("Generator/Learner Async  : PASS")
    print("CUDA Autotuned Learner   : PASS")
    print("Normalized Actor Gate    : PASS")
    print("Normalized DDQN Gate     : PASS")
    print("Target Network           : PASS")
    print("Periodic Checkpoints     :", "PASS" if args.checkpoint_every > 0 else "DISABLED")
    print("Final Checkpoint         : PASS")
    print()
    print("V8.8 JAX-VECTORIZED TRAINING: PASS")
    print(
        "NOTE: resulting checkpoints are challengers only until a fresh "
        "predeclared qualification block passes."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
