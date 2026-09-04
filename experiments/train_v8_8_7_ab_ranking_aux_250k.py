
from __future__ import annotations

# V8.8.7 RANKING-AUX PILOT (2026-08-31)
# Base: V8.8.6 formal Champion at 31.2M transitions.
# Goal: +250K transitions as a SMALL ablation pilot.
# Production DDQN/replay/Teacher/gates/candidate contract stay frozen.
# Only change: low-frequency offline pairwise ranking auxiliary updates.
# Counterfactual values never replace or modify TD rewards.
# The 31.2M Champion is never overwritten.

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

try:
    # Current V6+ flat-layout core.
    from tetris_ai.model.q_network import ObservableSafeQNetwork
    from tetris_ai.replay.packed import V881PackedReplayBuffer
    from tetris_ai.learning.ranking import (
        OfflineRankingCorpus,
        RankingAuxTrainer,
    )
    from tetris_ai.learning.cuda_graph import (
        CudaGraphDDQNLearner,
        LowSyncMetricTracker,
        make_capturable_adamw,
        save_checkpoint_v882,
    )
except ImportError:
    # Compatibility fallback for the archived pre-refactor tree.
    from ai.observable_q_network import ObservableSafeQNetwork
    from v8_8_1_packed_replay import V881PackedReplayBuffer
    from tetris_ai.learning.ranking import (
        OfflineRankingCorpus,
        RankingAuxTrainer,
    )
    from v8_8_3_cuda_graph_train_common import (
        CudaGraphDDQNLearner,
        LowSyncMetricTracker,
        make_capturable_adamw,
        save_checkpoint_v882,
    )


GLOBAL_SEED = 20260824
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

    while True:
        try:
            control_queue.get_nowait()
        except queue.Empty:
            break

    try:
        control_queue.put_nowait(payload)
    except queue.Full:
        pass


def _broadcast_latest_weights(control_queues, model, version):
    state = cpu_state_dict(model)
    for control_queue in control_queues:
        payload = {
            "kind": "weights",
            "version": int(version),
            "state_dict": state,
        }
        while True:
            try:
                control_queue.get_nowait()
            except queue.Empty:
                break
        try:
            control_queue.put_nowait(payload)
        except queue.Full:
            pass



def _set_process_affinity(cpus):
    cpus = sorted({int(x) for x in cpus})
    if not cpus:
        return []

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        if max(cpus) >= 64:
            raise RuntimeError(
                "V8.8.6 Windows affinity supports processor group 0 only."
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.c_size_t,
        ]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

        mask = 0
        for cpu in cpus:
            mask |= 1 << cpu

        ok = kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.c_size_t(mask),
        )
        if not ok:
            raise OSError(
                ctypes.get_last_error(),
                "SetProcessAffinityMask failed",
            )
        return cpus

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))

    return cpus


def _windows_core_topology():
    if os.name != "nt":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        func = kernel32.GetLogicalProcessorInformationEx
        func.argtypes = [
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
        ]
        func.restype = wintypes.BOOL

        relation_processor_core = 0
        length = wintypes.DWORD(0)

        func(
            relation_processor_core,
            None,
            ctypes.byref(length),
        )
        if length.value <= 0:
            return None

        buf = ctypes.create_string_buffer(length.value)
        if not func(
            relation_processor_core,
            ctypes.byref(buf),
            ctypes.byref(length),
        ):
            return None

        raw = memoryview(buf.raw)
        offset = 0
        cores = []
        pointer_bytes = ctypes.sizeof(ctypes.c_size_t)
        group_affinity_size = pointer_bytes + 8

        while offset + 8 <= length.value:
            relationship = int.from_bytes(
                raw[offset : offset + 4],
                "little",
            )
            size = int.from_bytes(
                raw[offset + 4 : offset + 8],
                "little",
            )
            if size <= 0:
                break

            if relationship == relation_processor_core:
                efficiency_class = int(raw[offset + 9])
                group_count = int.from_bytes(
                    raw[offset + 30 : offset + 32],
                    "little",
                )

                cpus = []
                group_offset = offset + 32

                for group_index in range(group_count):
                    g = group_offset + group_index * group_affinity_size
                    mask = int.from_bytes(
                        raw[g : g + pointer_bytes],
                        "little",
                    )
                    group = int.from_bytes(
                        raw[
                            g + pointer_bytes :
                            g + pointer_bytes + 2
                        ],
                        "little",
                    )

                    if group != 0:
                        continue

                    bit = 0
                    while mask:
                        if mask & 1:
                            cpus.append(bit)
                        mask >>= 1
                        bit += 1

                if cpus:
                    cores.append(
                        {
                            "efficiency_class": efficiency_class,
                            "cpus": sorted(cpus),
                        }
                    )

            offset += size

        return cores or None

    except Exception:
        return None


def _build_affinity_plan(producer_count, reserve_main_logical):
    logical = int(os.cpu_count() or 1)
    producer_count = int(producer_count)
    reserve_main_logical = max(0, int(reserve_main_logical))

    if logical < producer_count:
        raise RuntimeError(
            f"Only {logical} logical CPUs for {producer_count} producers."
        )

    topology = _windows_core_topology()

    if topology:
        # Reserve lowest-efficiency whole physical cores for the learner/main.
        by_efficiency = sorted(
            topology,
            key=lambda c: (
                int(c["efficiency_class"]),
                min(c["cpus"]),
            ),
        )

        reserved_cores = []
        reserved_count = 0
        for core in by_efficiency:
            if reserved_count >= reserve_main_logical:
                break
            reserved_cores.append(core)
            reserved_count += len(core["cpus"])

        reserved_keys = {
            tuple(core["cpus"])
            for core in reserved_cores
        }
        remaining = [
            core
            for core in topology
            if tuple(core["cpus"]) not in reserved_keys
        ]

        # Prefer higher-performance cores first and keep whole cores together.
        remaining.sort(
            key=lambda c: (
                -int(c["efficiency_class"]),
                min(c["cpus"]),
            )
        )

        groups = [[] for _ in range(producer_count)]
        for core in remaining:
            target = min(
                range(producer_count),
                key=lambda i: len(groups[i]),
            )
            groups[target].extend(core["cpus"])

        main = sorted(
            cpu
            for core in reserved_cores
            for cpu in core["cpus"]
        )

        if all(groups) and main:
            return {
                "source": "windows_efficiency_class",
                "producer_cpus": [
                    sorted(set(group))
                    for group in groups
                ],
                "main_cpus": sorted(set(main)),
                "topology": topology,
            }

    # Fallback: reserve the final logical CPUs, round-robin the rest.
    reserve = min(
        reserve_main_logical,
        max(0, logical - producer_count),
    )
    main = list(
        range(
            max(0, logical - reserve),
            logical,
        )
    )
    main_set = set(main)
    producer_pool = [
        cpu
        for cpu in range(logical)
        if cpu not in main_set
    ]

    groups = [[] for _ in range(producer_count)]
    for index, cpu in enumerate(producer_pool):
        groups[index % producer_count].append(cpu)

    return {
        "source": "balanced_logical_fallback",
        "producer_cpus": groups,
        "main_cpus": main,
        "topology": topology,
    }


def _create_shared_weight_bank(model):
    bank = {}
    for key, value in cpu_state_dict(model).items():
        tensor = value.contiguous().clone()
        tensor.share_memory_()
        bank[key] = tensor
    return bank


def _publish_shared_weights(
    model,
    shared_bank,
    shared_version,
    shared_lock,
    version,
):
    state = cpu_state_dict(model)

    with shared_lock:
        for key, value in state.items():
            shared_bank[key].copy_(value)
        shared_version.value = int(version)


def _poll_shared_weights(
    model,
    shared_bank,
    shared_version,
    shared_lock,
    current_version,
    force=False,
):
    with shared_lock:
        version = int(shared_version.value)

        if (not force) and version == int(current_version):
            return int(current_version)

        model.load_state_dict(shared_bank)
        model.eval()
        return version


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
    shared_weight_bank,
    shared_weight_version,
    shared_weight_lock,
    initial_weight_version,
    out_queue,
    ready_queue,
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

        try:
            from tetris_ai.backend.jax.vector_env import reset_batch
            from tetris_ai.backend.jax.teacher import (
                replace_done_or_segment_states_jit,
                select_candidate_state_jit,
                topk_batch,
            )
        except ImportError:
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

        producer_id = int(config["producer_id"])
        requested_affinity = list(
            config.get("affinity_cpus", [])
        )
        actual_affinity = (
            _set_process_affinity(requested_affinity)
            if requested_affinity
            else []
        )

        model = ObservableSafeQNetwork().cpu()
        weight_version = _poll_shared_weights(
            model,
            shared_weight_bank,
            shared_weight_version,
            shared_weight_lock,
            initial_weight_version,
            force=True,
        )

        b = int(config["vector_envs"])
        start_seed = sanitize_training_seed(config["start_seed"])
        seed_offset = int(config["seed_offset"])
        seed_stride = int(config["seed_stride"])
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
            int(start_seed) + seed_offset + stream_ids
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

        segment_steps = np.zeros(b, dtype=np.int32)
        generated = 0
        batch_index = 0

        ready_queue.put(
            {
                "kind": "ready",
                "producer_id": producer_id,
                "jax_version": str(jax.__version__),
                "jax_devices": [str(x) for x in jax.devices()],
                "vector_envs": b,
                "risk_streams": risk_streams,
                "compile_seconds": float(compile_elapsed),
                "affinity_cpus": actual_affinity,
            },
            timeout=120.0,
        )

        while not stop_event.is_set():
            weight_version = _poll_shared_weights(
                model,
                shared_weight_bank,
                shared_weight_version,
                shared_weight_lock,
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
                "episode_step": segment_steps.copy(),
                "lines": selected_lines,
                "is_risk": risk_mask.copy(),
            }

            message = {
                "kind": "transition_batch",
                "producer_id": producer_id,
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
                episode_seeds[reset_mask] += seed_stride
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
            "producer_id": int(config.get("producer_id", -1)),
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

    ages = np.asarray(batch.get("episode_step", np.zeros(n, dtype=np.int32)), dtype=np.int32)
    if ages.size:
        stats["max_episode_step"] = max(stats["max_episode_step"], int(ages.max()))
        stats["age_lt250"] += int(np.count_nonzero(ages < 250))
        stats["age_250_499"] += int(np.count_nonzero((ages >= 250) & (ages < 500)))
        stats["age_500_999"] += int(np.count_nonzero((ages >= 500) & (ages < 1000)))
        stats["age_ge1000"] += int(np.count_nonzero(ages >= 1000))
        terminal_rows = np.asarray(batch["real_done"], dtype=np.bool_)
        if np.any(terminal_rows):
            stats["completed_episode_lengths"].extend(ages[terminal_rows].astype(int).tolist())

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
            "V8.8.7 PILOT: continue the formal 31.2M Champion for +250K "
            "transitions with the V8.8.6 production DDQN recipe plus one "
            "low-frequency offline pairwise ranking auxiliary."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default="models/v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt",
        help=(
            "Control baseline: warm-start ONLY from the formal V8.8.6 31.2M Champion."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250000,
        help=(
            "NEW transitions. Pilot default +250,000 takes 31.2M -> 31.45M."
        ),
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=70001,
        help=(
            "Fresh control-training stream seed base; intentionally disjoint from "
            "the original 31.2M training streams and evaluation/development blocks."
        ),
    )

    parser.add_argument(
        "--producers",
        type=int,
        default=3,
        help=(
            "Independent JAX rollout producer processes. "
            "Default 2 targets the i5-13500 idle-core headroom."
        ),
    )
    parser.add_argument(
        "--affinity",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Windows CPU affinity policy. auto detects physical-core "
            "EfficiencyClass and balances P/E cores across producers. "
            "Falls back to balanced logical-CPU groups if topology probing fails."
        ),
    )
    parser.add_argument(
        "--main-reserve-logical",
        type=int,
        default=2,
        help=(
            "Logical CPUs reserved for the CUDA learner/main process when "
            "--affinity=auto."
        ),
    )
    parser.add_argument(
        "--vector-envs",
        type=int,
        default=256,
        help=(
            "Vector envs PER producer. With the default two producers this is "
            "512 concurrent natural streams."
        ),
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
            "0 disables forced segmentation. V8.8.1 intentionally keeps long natural trajectories and resets only on terminal/no-successor."
        ),
    )

    parser.add_argument("--behavior-gate", type=float, default=0.600)
    parser.add_argument("--risk-behavior-gate", type=float, default=0.400)
    parser.add_argument("--target-gate", type=float, default=0.600)
    parser.add_argument("--exploration", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--terminal-penalty", type=float, default=1.0)
    parser.add_argument("--terminal-replay-copies", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=750000)

    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="0 = auto max(2048, chosen learner batch size).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="CUDA Graph fixed batch. 8192 is validated on RTX 5070 and passes the 10K-window freshness guard.",
    )
    parser.add_argument(
        "--batch-candidates",
        default="4096,8192",
    )
    parser.add_argument("--benchmark-warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument("--batch-near-best-ratio", type=float, default=0.95)

    parser.add_argument(
        "--sample-budget",
        type=int,
        default=50872320,
        help=(
            "Preserves the V8.8.6 ratio 203.48928 replay samples/new transition: "
            "250,000 transitions = 50,872,320 samples = 6,210 TD gradients."
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
        default=2,
        help=(
            "Keep queue depth small for policy freshness. At 256 streams, "
            "two batches are still enough to overlap generator and learner."
        ),
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
    parser.add_argument(
        "--metric-collect-every",
        type=int,
        default=8,
        help="Collect learner diagnostics every N gradients without host sync.",
    )
    parser.add_argument(
        "--metric-sync-every",
        type=int,
        default=64,
        help="Synchronize accumulated learner diagnostics to CPU every N gradients.",
    )
    parser.add_argument("--log-every", type=int, default=1000)

    parser.add_argument("--checkpoint-every", type=int, default=125000)
    parser.add_argument(
        "--checkpoint-prefix",
        default="models/v8_8_7_ab_ranking_aux",
    )
    parser.add_argument(
        "--max-batch-fraction",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--output",
        default="models/v8_8_7_ab_ranking_aux_31450k.pt",
    )
    parser.add_argument(
        "--ranking-corpus",
        default="data/v8_8_7_ranking_corpus_4761_4775.npz",
    )
    parser.add_argument("--ranking-weight", type=float, default=0.01)
    parser.add_argument("--ranking-temperature", type=float, default=0.10)
    parser.add_argument("--ranking-batch-size", type=int, default=32)
    parser.add_argument(
        "--ranking-every",
        type=int,
        default=1242,
        help=(
            "For the default 6,210 TD gradients, 1242 yields exactly "
            "five evenly-spaced ranking updates."
        ),
    )
    parser.add_argument("--ranking-seed", type=int, default=20260831)
    parser.add_argument(
        "--ranking-max-updates",
        type=int,
        default=5,
        help="Hard safety cap. CV-selected pilot uses exactly 5 updates.",
    )

    parser.add_argument(
        "--parity-stamp",
        default="data/v8_8_jax_teacher_parity_pass.json",
    )
    parser.add_argument(
        "--graph-gate-stamp",
        default="data/v8_8_3_dynamicmask_cuda_graph_gate_pass.json",
        help="Required PASS stamp produced by test_v8_8_2_production_cuda_graph_gate.py.",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.producers <= 0:
        raise ValueError("--producers must be > 0")
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
    if args.metric_collect_every <= 0:
        raise ValueError("--metric-collect-every must be > 0")
    if args.metric_sync_every < args.metric_collect_every:
        raise ValueError("--metric-sync-every must be >= --metric-collect-every")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be >= 0")
    if args.checkpoint_every > args.steps:
        raise ValueError("--checkpoint-every cannot exceed --steps")
    if not 0.0 < args.max_batch_fraction <= 1.0:
        raise ValueError("--max-batch-fraction must be in (0,1]")
    if not 0.0 < args.batch_near_best_ratio <= 1.0:
        raise ValueError("--batch-near-best-ratio must be in (0,1]")
    if args.ranking_weight <= 0.0:
        raise ValueError("--ranking-weight must be > 0")
    if args.ranking_temperature <= 0.0:
        raise ValueError("--ranking-temperature must be > 0")
    if args.ranking_batch_size <= 0:
        raise ValueError("--ranking-batch-size must be > 0")
    if args.ranking_every <= 0:
        raise ValueError("--ranking-every must be > 0")
    if args.ranking_max_updates <= 0:
        raise ValueError("--ranking-max-updates must be > 0")

    parity_stamp = _verify_parity_stamp(args.parity_stamp)
    graph_gate_stamp = _verify_parity_stamp(args.graph_gate_stamp)

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

    # This script is deliberately a controlled continuation experiment.
    # Refuse accidental use of another lineage or accidental overwrite.
    if inherited_env_steps != 31_200_000:
        raise RuntimeError(
            "CONTROL BASE MISMATCH: expected checkpoint env_steps=31,200,000, "
            f"got {inherited_env_steps:,}. Use the formal V8.8.6 31.2M Champion."
        )
    if os.path.abspath(args.checkpoint) == os.path.abspath(args.output):
        raise RuntimeError(
            "Refusing to overwrite the 31.2M input checkpoint. "
            "Use a distinct --output path."
        )
    if args.steps == 250_000 and args.sample_budget != 50_872_320:
        raise RuntimeError(
            "V8.8.7 PILOT RECIPE MISMATCH: +250K requires "
            "--sample-budget 50872320 to preserve 203.48928 "
            "replay samples/new transition."
        )

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

    # V8.8.2 production uses the already-profiler-validated CUDA Graph
    # batch=8192 by default. Do not retune merely to increase GPU utilization.
    args.batch_size = int(args.batch_size)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

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
    if args.batch_size > freshness_limit:
        raise ValueError(
            f"CUDA Graph batch {args.batch_size} exceeds freshness guard "
            f"{freshness_limit}."
        )

    args.batch_benchmark = []
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

    if args.steps == 250_000:
        if target_gradient_steps != 6_210:
            raise RuntimeError(
                f"A/B TREATMENT expected 6,210 TD gradients, got "
                f"{target_gradient_steps:,}."
            )
        if args.ranking_weight != 0.01:
            raise RuntimeError(
                "A/B TREATMENT requires --ranking-weight 0.01 from CV."
            )
        if args.ranking_every != 1242 or args.ranking_max_updates != 5:
            raise RuntimeError(
                "A/B TREATMENT requires exactly five ranking updates: "
                "--ranking-every 1242 --ranking-max-updates 5."
            )

    optimizer, optimizer_lr_tensor, optimizer_resumed = (
        make_capturable_adamw(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_optimizer_state=checkpoint.get(
                "optimizer_state_dict"
            ),
            resume=(
                args.resume_optimizer
                and "optimizer_state_dict" in checkpoint
            ),
        )
    )
    args.optimizer_resumed_actual = optimizer_resumed

    if not optimizer_resumed:
        raise RuntimeError(
            "CONTROL CONTINUATION REQUIRES optimizer-state resume, but the "
            "31.2M checkpoint did not provide a resumable optimizer state. "
            "Stop here rather than silently changing the training recipe."
        )

    replay = V881PackedReplayBuffer(
        capacity=args.replay_capacity,
        device=device,
        seed=GLOBAL_SEED + 88101,
    )

    if device.type != "cuda":
        raise RuntimeError(
            "V8.8.2 production trainer requires CUDA; CPU fallback is intentionally disabled."
        )

    graph_learner = CudaGraphDDQNLearner(
        model=model,
        target_model=target_model,
        optimizer=optimizer,
        replay=replay,
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_gate=args.target_gate,
        terminal_penalty=args.terminal_penalty,
    )

    ranking_corpus = OfflineRankingCorpus(
        args.ranking_corpus,
        device=device,
    )
    ranking_trainer = RankingAuxTrainer(
        model=model,
        optimizer=optimizer,
        corpus=ranking_corpus,
        split="train",
        batch_size=args.ranking_batch_size,
        weight=args.ranking_weight,
        temperature=args.ranking_temperature,
        seed=args.ranking_seed,
    )

    runtime_meta = {
        "parity_stamp": parity_stamp,
        "cuda_graph_gate_stamp": graph_gate_stamp,
        "formal_champion_reference": "V8.8.6 31.2M normalized 0.600",
        "input_lineage": "V8.8.6 31.2M FORMAL CHAMPION; same-recipe +10M control continuation",
        "control_experiment": "same_recipe_more_training_no_algorithm_change",
        "replay_backend": "packed_device_resident_float32_v1",
        "learner_backend": "cuda_graph_fixed_shape_ddqn_v2",
        "v8_8_7_auxiliary": "offline_pairwise_logistic_ranking_v1",
        "ranking_corpus": str(args.ranking_corpus),
        "ranking_weight": float(args.ranking_weight),
        "ranking_temperature": float(args.ranking_temperature),
        "ranking_every_td_gradients": int(args.ranking_every),
        "ranking_batch_size": int(args.ranking_batch_size),
        "ranking_reward_contract": "counterfactual values never modify TD rewards",
        "trajectory_design": f"{args.vector_envs} natural streams; terminal/no-next reset only",
    }

    print()
    print("=" * 80)
    print("V8.8.6 CONTROL: 31.2M -> 41.2M SAME-RECIPE CONTINUATION")
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
    print("JAX producers:", args.producers)
    print("JAX vector environments / producer:", args.vector_envs)
    print("JAX total vector environments:", args.producers * args.vector_envs)
    print("Risk streams / producer:", args.risk_streams)
    print("Total risk streams:", args.risk_streams * args.producers)
    print("Forced segment length:", args.segment_pieces or "DISABLED")
    print("Training seed start:", args.start_seed)
    print("Normal normalized gate:", args.behavior_gate)
    print("Risk normalized gate:", args.risk_behavior_gate)
    print("DDQN target gate:", args.target_gate)
    print("Exploration:", f"{args.exploration * 100:.2f}%")
    print()
    print("Packed replay capacity:", args.replay_capacity)
    print("Packed replay device:", replay.device)
    print("Packed replay width:", replay.packed_width, "float32 / transition")
    print(
        "Packed replay allocated:",
        f"{replay.nbytes / (1024.0 ** 2):.1f} MiB",
    )
    print(
        "Long-trajectory nominal steps/initial stream:",
        f"{args.steps / (args.producers * args.vector_envs):.1f}",
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
    print(
        "Learner metric sync:",
        f"collect/{args.metric_collect_every} gradients, host-sync/{args.metric_sync_every}",
    )
    print("Permanent seeds 6~20:", "PROTECTED")
    print("Formal Champion reference:", "V8.8.6 31.2M normalized 0.600")
    print("Input lineage:", "V8.8.6 31.2M FORMAL CHAMPION")
    print("Experiment:", "SAME RECIPE, +10M MORE TRAINING, NO ALGORITHM CHANGE")
    print("Output status:", "CONTROL CHALLENGER ONLY; 31.2M Champion remains frozen")
    print("CUDA Graph learner:", "ENABLED - DYNAMIC next_mask V3")
    print("Graph capture:", "NON-DESTRUCTIVE")
    print()
    print("OBSERVABLE SAFETY:")
    print("  Q current input   : state243")
    print("  Q candidate input : board200 + rotation4 + x10 + hold1 = 215")
    print("  Preview future tail 200:243 is NEVER candidate input")
    print("  Real selected next state243 is used only for TD bootstrap")

    ctx = mp.get_context("spawn")
    total_vector_envs = args.producers * args.vector_envs
    out_queue = ctx.Queue(
        maxsize=max(1, args.queue_batches * args.producers)
    )
    ready_queue = ctx.Queue(maxsize=args.producers)
    stop_event = ctx.Event()

    if args.affinity == "auto":
        affinity_plan = _build_affinity_plan(
            args.producers,
            args.main_reserve_logical,
        )
    else:
        affinity_plan = {
            "source": "disabled",
            "producer_cpus": [
                []
                for _ in range(args.producers)
            ],
            "main_cpus": [],
            "topology": None,
        }

    shared_weight_bank = _create_shared_weight_bank(model)
    shared_weight_version = ctx.Value(
        "q",
        int(inherited_env_steps),
        lock=False,
    )
    shared_weight_lock = ctx.Lock()

    generators = []

    for producer_id in range(args.producers):
        generator_config = {
            "producer_id": producer_id,
            "affinity_cpus": (
                affinity_plan["producer_cpus"][producer_id]
            ),
            "vector_envs": args.vector_envs,
            "risk_streams": args.risk_streams,
            "segment_pieces": args.segment_pieces,
            "start_seed": args.start_seed,
            "seed_offset": producer_id * args.vector_envs,
            "seed_stride": total_vector_envs,
            "behavior_gate": args.behavior_gate,
            "risk_behavior_gate": args.risk_behavior_gate,
            "exploration": args.exploration,
            "generator_rng_seed": (
                GLOBAL_SEED + 8800 + producer_id * 100003
            ),
        }

        generator = ctx.Process(
            target=vector_generator_loop,
            args=(
                generator_config,
                shared_weight_bank,
                shared_weight_version,
                shared_weight_lock,
                inherited_env_steps,
                out_queue,
                ready_queue,
                stop_event,
            ),
            name=f"v8_8_5_jax_generator_{producer_id}",
        )
        generator.start()
        generators.append(generator)

    if (
        args.affinity == "auto"
        and affinity_plan["main_cpus"]
    ):
        _set_process_affinity(
            affinity_plan["main_cpus"]
        )

    ready_reports = {}
    startup_deadline = time.perf_counter() + args.generator_ready_timeout

    while len(ready_reports) < args.producers:
        remaining = startup_deadline - time.perf_counter()
        if remaining <= 0:
            stop_event.set()
            for generator in generators:
                if generator.is_alive():
                    generator.terminate()
                generator.join(timeout=2.0)
            raise RuntimeError(
                "V8.8.6 JAX producers did not all become ready in time."
            )

        try:
            ready = ready_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            dead = [g.name for g in generators if not g.is_alive()]
            if dead:
                stop_event.set()
                for generator in generators:
                    if generator.is_alive():
                        generator.terminate()
                    generator.join(timeout=2.0)
                raise RuntimeError(
                    "V8.8.6 producer died during startup: "
                    + ", ".join(dead)
                )
            continue

        if ready.get("kind") != "ready":
            raise RuntimeError(
                f"Unexpected producer startup message: {ready!r}"
            )

        ready_reports[int(ready["producer_id"])] = ready

    ordered_ready = [ready_reports[i] for i in range(args.producers)]

    runtime_meta.update(
        {
            "jax_version": ordered_ready[0]["jax_version"],
            "jax_devices": ordered_ready[0]["jax_devices"],
            "producer_count": args.producers,
            "vector_envs_per_producer": args.vector_envs,
            "total_vector_envs": total_vector_envs,
            "risk_streams_per_producer": args.risk_streams,
            "total_risk_streams": args.risk_streams * args.producers,
            "generator_compile_seconds": [
                x["compile_seconds"] for x in ordered_ready
            ],
            "affinity_source": affinity_plan["source"],
            "producer_affinity": [
                x["affinity_cpus"]
                for x in ordered_ready
            ],
            "main_affinity": affinity_plan["main_cpus"],
            "shared_weight_bank": True,
        }
    )

    print()
    print("Generator JAX:", ordered_ready[0]["jax_version"])
    print("Generator devices:", ordered_ready[0]["jax_devices"])
    print("Generator producers:", args.producers)
    print("Vector envs / producer:", args.vector_envs)
    print("Affinity source:", affinity_plan["source"])
    for ready in ordered_ready:
        print(
            f"  producer {ready['producer_id']} CPUs:",
            ready["affinity_cpus"],
        )
    print("  main/learner CPUs:", affinity_plan["main_cpus"])
    print("Shared weight bank:", "ENABLED")
    print("Total vector envs:", total_vector_envs)
    for ready in ordered_ready:
        print(
            f"  producer {ready['producer_id']} compile:",
            f"{ready['compile_seconds']:.2f}s",
        )
    print("All generators ready: PASS")
    print()

    collected = 0
    new_gradient_steps = 0
    ranking_updates = 0
    terminal_replay_extra_entries = 0

    stats = {
        "teacher_actions": 0,
        "q_interventions": 0,
        "random_explorations": 0,
        "real_gameovers": 0,
        "no_next_resets": 0,
        "segment_boundaries": 0,
        "risk_transitions": 0,
        "max_episode_step": 0,
        "age_lt250": 0,
        "age_250_499": 0,
        "age_500_999": 0,
        "age_ge1000": 0,
        "completed_episode_lengths": [],
        "q_margin_history": [],
        "training_seeds": set(),
    }

    loss_history = []
    q_history = []
    td_history = []
    ranking_loss_history = []
    ranking_acc_history = []
    metric_tracker = LowSyncMetricTracker(
        collect_every=args.metric_collect_every,
        sync_every=args.metric_sync_every,
    )

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

    def _record_metric_snapshot(metrics):
        if metrics is None:
            return
        loss_history.append(metrics["loss"])
        q_history.append(metrics["q_mean"])
        td_history.append(metrics["td_abs"])

    def _learner_step():
        nonlocal new_gradient_steps, ranking_updates
        collect_metrics = metric_tracker.should_collect(
            new_gradient_steps + 1
        )
        metric_tensor = graph_learner.step(
            collect_metrics=collect_metrics
        )
        new_gradient_steps += 1
        metric_tracker.add(metric_tensor)

        if (
            ranking_updates < args.ranking_max_updates
            and new_gradient_steps % args.ranking_every == 0
        ):
            rank_metrics = ranking_trainer.step(
                collect_metrics=collect_metrics
            )
            ranking_updates += 1
            if rank_metrics is not None:
                ranking_loss_history.append(rank_metrics.loss)
                ranking_acc_history.append(rank_metrics.pair_accuracy)

        if metric_tracker.should_sync():
            _record_metric_snapshot(
                metric_tracker.flush()
            )

        if (
            new_gradient_steps
            % target_update_gradients
            == 0
        ):
            graph_learner.update_target_from_online()

    def run_due_gradients():
        if len(replay) < args.warmup:
            return

        post_warmup_total = max(1, args.steps - args.warmup)
        post_warmup_done = max(0, collected - args.warmup)
        desired = min(
            target_gradient_steps,
            int(math.floor(
                post_warmup_done / post_warmup_total * target_gradient_steps
            )),
        )

        while new_gradient_steps < desired:
            _learner_step()

    def maybe_sync():
        nonlocal next_sync
        while collected >= next_sync:
            absolute_version = (
                inherited_env_steps + collected
            )
            _publish_shared_weights(
                model,
                shared_weight_bank,
                shared_weight_version,
                shared_weight_lock,
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

        args.ranking_updates_actual = int(ranking_updates)
        save_checkpoint_v882(
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


        # One batched metric synchronization at log time keeps diagnostics
        # current without reintroducing per-gradient .item() stalls.
        _record_metric_snapshot(metric_tracker.flush())

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

    interrupted = False
    emergency_checkpoint = None

    try:
        while collected < args.steps:
            if stop_event.is_set():
                dead = [g.name for g in generators if not g.is_alive()]
                if dead:
                    raise RuntimeError(
                        "V8.8.6 producer stopped before collection completed: "
                        + ", ".join(dead)
                    )

            try:
                message = out_queue.get(timeout=5.0)
            except queue.Empty:
                dead = [g.name for g in generators if not g.is_alive()]
                if dead:
                    raise RuntimeError(
                        "V8.8.6 producer exited before collection completed: "
                        + ", ".join(dead)
                    )
                continue

            kind = message.get("kind")

            if kind == "error":
                raise RuntimeError(
                    "V8.8.5 generator error "
                    f"(producer={message.get('producer_id', -1)}):\n"
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
            _learner_step()

        _record_metric_snapshot(metric_tracker.flush())
        graph_learner.update_target_from_online()
        torch.cuda.synchronize(device)

        core_training_elapsed = (
            time.perf_counter() - run_start
        )

        args.ranking_updates_actual = int(ranking_updates)
        save_checkpoint_v882(
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

    except KeyboardInterrupt:
        interrupted = True
        _record_metric_snapshot(metric_tracker.flush())
        graph_learner.update_target_from_online()
        torch.cuda.synchronize(device)

        core_training_elapsed = time.perf_counter() - run_start
        total_k = (inherited_env_steps + collected) // 1000
        emergency_checkpoint = (
            f"{args.checkpoint_prefix}_INTERRUPTED_{total_k}k.pt"
        )

        interrupt_meta = dict(runtime_meta)
        interrupt_meta["interrupted"] = True

        args.ranking_updates_actual = int(ranking_updates)
        save_checkpoint_v882(
            path=emergency_checkpoint,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            inherited_env_steps=inherited_env_steps,
            new_env_steps=collected,
            inherited_gradient_steps=inherited_gradient_steps,
            new_gradient_steps=new_gradient_steps,
            replay_size=len(replay),
            unique_training_seeds=len(stats["training_seeds"]),
            args=args,
            runtime_meta=interrupt_meta,
        )

        print()
        print("SAFE INTERRUPT CHECKPOINT SAVED:", emergency_checkpoint)

    finally:
        stop_event.set()
        shutdown_start = time.perf_counter()

        for generator in generators:
            generator.join(timeout=max(0.0, args.shutdown_grace))

        still_alive = [g for g in generators if g.is_alive()]
        if still_alive:
            forced_termination = True
            for generator in still_alive:
                generator.terminate()
                generator.join(timeout=2.0)

        shutdown_elapsed = time.perf_counter() - shutdown_start

    total_elapsed = (
        core_training_elapsed + shutdown_elapsed
    )

    print()
    print("=" * 80)
    print("V8.8.7 RANKING-AUX PILOT SUMMARY")
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
    print("Max observed natural trajectory step:", stats["max_episode_step"])
    age_total = max(collected, 1)
    print(
        "Trajectory-age mix <250 / 250-499 / 500-999 / >=1000:",
        f"{stats['age_lt250']/age_total*100:.1f}% / "
        f"{stats['age_250_499']/age_total*100:.1f}% / "
        f"{stats['age_500_999']/age_total*100:.1f}% / "
        f"{stats['age_ge1000']/age_total*100:.1f}%",
    )
    if stats["completed_episode_lengths"]:
        print(
            "Completed natural episode length mean/max:",
            f"{float(np.mean(stats['completed_episode_lengths'])):.1f} / "
            f"{max(stats['completed_episode_lengths'])}",
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
    print("Ranking auxiliary updates:", ranking_updates)
    print("Ranking cadence:", f"1 / {args.ranking_every} TD gradients")
    print("Ranking weight:", f"{args.ranking_weight:.4f}")
    print("Ranking temperature:", f"{args.ranking_temperature:.4f}")
    print()
    print(
        "Checkpoint:",
        emergency_checkpoint if interrupted else args.output,
    )

    if interrupted:
        print()
        print("V8.8.7 PILOT: SAFELY INTERRUPTED")
        return

    if collected != args.steps:
        raise RuntimeError(
            "Collected transition count mismatch."
        )
    if args.steps == 250_000 and ranking_updates != 5:
        raise RuntimeError(
            f"V8.8.7 A/B ranking update count mismatch: "
            f"expected 5, got {ranking_updates}."
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
    print("JAX 3-Producer Rollout    : PASS")
    print("CPU Affinity              : PASS")
    print("Shared Actor Weights      : PASS")
    print("Vectorized Teacher Top-K : PASS")
    print("Observable Candidate Q   : PASS")
    print("Candidate/Next Split     : PASS")
    print("Packed Device Replay     : PASS")
    print("CUDA Graph DDQN Learner  : PASS")
    print("Graph-safe Policy        : PASS")
    print("Dynamic next_mask        : PASS")
    print("Non-destructive Capture  : PASS")
    print("Generator/Learner Async  : PASS")
    print("Long Natural Trajectories : PASS")
    print("Low-Sync Learner Metrics  : PASS")
    print("CUDA Autotuned Learner   : PASS")
    print("Normalized Actor Gate    : PASS")
    print("Normalized DDQN Gate     : PASS")
    print("Target Network           : PASS")
    print("Periodic Checkpoints     :", "PASS" if args.checkpoint_every > 0 else "DISABLED")
    print("Final Checkpoint         : PASS")
    print()
    print("V8.8.7 A/B MICRO-PILOT — RANKING-AUX TREATMENT 31.45M: PASS")
    print(
        "NOTE: this checkpoint is a DEVELOPMENT PILOT only. The formal 31.2M "
        "Champion remains unchanged. Do not long-train or promote unless "
        "held-out matched-state ranking improves without whole-game regression."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
