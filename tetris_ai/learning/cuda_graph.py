from __future__ import annotations

import copy
import math
import os
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from tetris_ai.model.q_network import (
    STATE_SIZE,
    CANDIDATE_SIZE,
)
from tetris_ai.policy.confidence import (
    normalized_margin_actions_top4_graphsafe,
)

TOP_K = 4


class LowSyncMetricTracker:
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
        values = (
            torch.stack(self.pending, dim=0)
            .mean(dim=0)
            .float()
            .cpu()
            .numpy()
        )
        self.pending.clear()
        result = {
            "loss": float(values[0]),
            "q_mean": float(values[1]),
            "target_mean": float(values[2]),
            "td_abs": float(values[3]),
        }
        for key, value in result.items():
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite learner metric: {key}={value}"
                )
        return result


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def make_capturable_adamw(
    *,
    model,
    lr: float,
    weight_decay: float,
    checkpoint_optimizer_state=None,
    resume: bool,
):
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(
            "V8.8.2 production CUDA Graph learner requires CUDA."
        )

    lr_tensor = torch.tensor(
        float(lr),
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_tensor,
        weight_decay=float(weight_decay),
        capturable=True,
        foreach=False,
    )

    resumed = False
    if resume and checkpoint_optimizer_state is not None:
        optimizer.load_state_dict(checkpoint_optimizer_state)
        move_optimizer_state_to_device(optimizer, device)

        # load_state_dict restores old param-group flags. Reassert graph-safe
        # production settings without changing the AdamW moments/step counts.
        for group in optimizer.param_groups:
            group["lr"] = lr_tensor
            group["weight_decay"] = float(weight_decay)
            group["capturable"] = True
            group["foreach"] = False
        resumed = True

    return optimizer, lr_tensor, resumed


def _graphsafe_clip_grad_norm_(
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



def _graph_static_views(replay, packed):
    """
    Return only true views into packed storage.

    IMPORTANT:
    V881PackedReplayBuffer._views() converts next_mask with '> 0.5', which
    materializes a separate bool tensor. That is correct for one-shot eager
    sampling, but wrong for a reusable CUDA Graph static input buffer because
    the mask would freeze at capture time.

    Here next_mask_raw remains a float32 view. _train_math converts it to bool
    inside the captured graph so every replay sees the newly sampled rows.
    """
    n = int(packed.shape[0])
    k = replay.layout.top_k
    a = replay.layout.candidate_size
    off = replay.layout.offsets

    def view(name):
        lo, hi = off[name]
        return packed[:, lo:hi]

    return {
        "state": view("state"),
        "candidate": view("candidate"),
        "reward": view("reward")[:, 0],
        "teacher_score": view("teacher_score")[:, 0],
        "teacher_rank": view("teacher_rank")[:, 0],
        "done": view("done")[:, 0],
        "next_state": view("next_state"),
        "next_candidates": view("next_candidates").view(n, k, a),
        "next_rewards": view("next_rewards").view(n, k),
        "next_teacher_scores": view("next_teacher_scores").view(n, k),
        "next_teacher_ranks": view("next_teacher_ranks").view(n, k),
        "next_mask_raw": view("next_mask").view(n, k),
    }


def _train_math(
    *,
    model,
    target_model,
    optimizer,
    batch,
    rows,
    gamma_tensor,
    target_gate,
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
    next_mask = batch["next_mask_raw"] > 0.5

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
                float(target_gate),
            )
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

    _graphsafe_clip_grad_norm_(
        model.parameters(),
        clip_max_tensor,
        clip_eps_tensor,
    )

    optimizer.step()

    return loss, current_q, td_target


def _clone_model_state_tensors(model):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _restore_model_state_in_place(model, snapshot):
    with torch.no_grad():
        current = model.state_dict()
        for key, saved in snapshot.items():
            current[key].copy_(saved)


def _snapshot_optimizer_state(optimizer):
    snap = {}
    for param, state in optimizer.state.items():
        state_copy = {}
        for key, value in state.items():
            if torch.is_tensor(value):
                state_copy[key] = value.detach().clone()
            else:
                state_copy[key] = copy.deepcopy(value)
        snap[param] = state_copy
    return snap


def _restore_optimizer_state_in_place(optimizer, snapshot):
    # AdamW state tensor addresses created during graph warmup must remain
    # stable because the captured graph references them. Restore values only.
    for param, state in optimizer.state.items():
        old = snapshot.get(param)

        if old is None:
            # Fresh optimizer: warmup created exp_avg/exp_avg_sq/step.
            # Reset them to mathematically fresh AdamW state.
            for key, value in state.items():
                if torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    state[key] = type(value)(0)
            continue

        for key, value in state.items():
            if key not in old:
                if torch.is_tensor(value):
                    value.zero_()
                continue

            saved = old[key]
            if torch.is_tensor(value):
                value.copy_(saved)
            else:
                state[key] = copy.deepcopy(saved)


class CudaGraphDDQNLearner:
    """
    Fixed-shape DDQN math captured once as a CUDA Graph.

    Dynamic replay sampling remains OUTSIDE the graph, preserving the existing
    GPU-resident no-replacement replay semantics. Graph capture is explicitly
    non-destructive: model parameters and resumed AdamW state are restored
    in-place after warmup/capture before the first counted training gradient.
    """

    def __init__(
        self,
        *,
        model,
        target_model,
        optimizer,
        replay,
        batch_size: int,
        gamma: float,
        target_gate: float,
        terminal_penalty: float,
        capture_warmup_steps: int = 4,
    ):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.replay = replay
        self.batch_size = int(batch_size)
        self.target_gate = float(target_gate)
        self.device = next(model.parameters()).device

        if self.device.type != "cuda":
            raise RuntimeError(
                "CudaGraphDDQNLearner requires CUDA."
            )
        if replay.device.type != "cuda":
            raise RuntimeError(
                "Packed replay must reside on CUDA."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.static_packed = torch.empty(
            (
                self.batch_size,
                replay.packed_width,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.static_batch = _graph_static_views(
            replay,
            self.static_packed,
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

        # Safe synthetic capture input. All four next candidates valid.
        with torch.no_grad():
            self.static_packed.zero_()
            lo, hi = replay.layout.offsets["next_mask"]
            self.static_packed[:, lo:hi].fill_(1.0)

        model_snapshot = _clone_model_state_tensors(
            self.model
        )
        optimizer_snapshot = _snapshot_optimizer_state(
            self.optimizer
        )

        # Ensure gradient and optimizer-state storage exists at stable addresses.
        for _ in range(max(1, int(capture_warmup_steps))):
            _train_math(
                model=self.model,
                target_model=self.target_model,
                optimizer=self.optimizer,
                batch=self.static_batch,
                rows=self.rows,
                gamma_tensor=self.gamma_tensor,
                target_gate=self.target_gate,
                terminal_penalty_tensor=(
                    self.terminal_penalty_tensor
                ),
                clip_max_tensor=self.clip_max_tensor,
                clip_eps_tensor=self.clip_eps_tensor,
            )

        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            (
                self._loss_ref,
                self._current_q_ref,
                self._td_target_ref,
            ) = _train_math(
                model=self.model,
                target_model=self.target_model,
                optimizer=self.optimizer,
                batch=self.static_batch,
                rows=self.rows,
                gamma_tensor=self.gamma_tensor,
                target_gate=self.target_gate,
                terminal_penalty_tensor=(
                    self.terminal_penalty_tensor
                ),
                clip_max_tensor=self.clip_max_tensor,
                clip_eps_tensor=self.clip_eps_tensor,
            )

        torch.cuda.synchronize(self.device)

        # Capture/warmup MUST NOT count as training.
        _restore_model_state_in_place(
            self.model,
            model_snapshot,
        )
        _restore_optimizer_state_in_place(
            self.optimizer,
            optimizer_snapshot,
        )

        self.optimizer.zero_grad(
            set_to_none=False
        )
        torch.cuda.synchronize(self.device)

        self.capture_non_destructive = True

    @torch.no_grad()
    def _sample_into_static(self):
        if len(self.replay) < self.batch_size:
            raise RuntimeError(
                "Replay smaller than CUDA Graph batch."
            )

        # Exact V8.8.1 sampling rule: randperm then first batch_size indices.
        idx = torch.randperm(
            len(self.replay),
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

    def step(self, collect_metrics: bool = False):
        self._sample_into_static()
        self.graph.replay()

        if not collect_metrics:
            return None

        # Enqueued after graph replay on the same stream, so these values refer
        # to the just-completed gradient. This remains low-frequency only.
        with torch.no_grad():
            td_error = (
                self._td_target_ref
                - self._current_q_ref.detach()
            )
            return torch.stack(
                (
                    self._loss_ref.detach(),
                    self._current_q_ref.detach().mean(),
                    self._td_target_ref.detach().mean(),
                    td_error.detach().abs().mean(),
                )
            )

    @torch.no_grad()
    def update_target_from_online(self):
        target_tensors = list(
            self.target_model.state_dict().values()
        )
        online_tensors = list(
            self.model.state_dict().values()
        )
        if len(target_tensors) != len(online_tensors):
            raise RuntimeError(
                "Online/target state length mismatch."
            )
        for dst, src in zip(
            target_tensors,
            online_tensors,
        ):
            dst.copy_(src)


def save_checkpoint_v882(
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

    payload = {
        "version": (
            "V8_8_3_DYNAMICMASK_CUDA_GRAPH_LONG_TRAJECTORY_"
            "PACKED_DEVICE_REPLAY_NORMALIZED_DDQN"
        ),
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": (
            target_model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "env_steps": int(total_env_steps),
        "gradient_steps": int(total_gradient_steps),
        "inherited_env_steps": int(inherited_env_steps),
        "new_env_steps": int(new_env_steps),
        "inherited_gradient_steps": int(
            inherited_gradient_steps
        ),
        "new_gradient_steps": int(
            new_gradient_steps
        ),
        "replay_size": int(replay_size),
        "state_size": int(STATE_SIZE),
        "candidate_size": int(CANDIDATE_SIZE),
        "top_k": int(TOP_K),
        "producer_count": int(getattr(args, "producers", 1)),
        "cpu_affinity_mode": str(getattr(args, "affinity", "off")),
        "vector_envs_per_producer": int(args.vector_envs),
        "vector_envs": int(
            args.vector_envs * getattr(args, "producers", 1)
        ),
        "risk_streams_per_producer": int(args.risk_streams),
        "risk_streams": int(
            args.risk_streams * getattr(args, "producers", 1)
        ),
        "risk_fraction": float(
            args.risk_streams
            / max(args.vector_envs, 1)
        ),
        "segment_pieces": int(
            args.segment_pieces
        ),
        "unique_training_seeds_this_run": int(
            unique_training_seeds
        ),
        "behavior_gate": float(
            args.behavior_gate
        ),
        "risk_behavior_gate": float(
            args.risk_behavior_gate
        ),
        "target_gate": float(
            args.target_gate
        ),
        "normalized_gate": float(
            args.target_gate
        ),
        "gate_semantics": (
            "normalized_q_margin"
        ),
        "exploration": float(
            args.exploration
        ),
        "gamma": float(args.gamma),
        "batch_size": int(
            args.batch_size
        ),
        "warmup": int(args.warmup),
        "sample_budget": int(
            args.sample_budget
        ),
        "terminal_penalty": float(
            args.terminal_penalty
        ),
        "terminal_replay_copies": int(
            args.terminal_replay_copies
        ),
        "target_update_samples": int(
            args.target_update_samples
        ),
        "sync_every": int(
            args.sync_every
        ),
        "queue_batches": int(
            args.queue_batches
        ),
        "metric_collect_every": int(
            args.metric_collect_every
        ),
        "metric_sync_every": int(
            args.metric_sync_every
        ),
        "input_checkpoint": args.checkpoint,
        "checkpoint_every": int(
            args.checkpoint_every
        ),
        "checkpoint_prefix": (
            args.checkpoint_prefix
        ),
        "optimizer_resumed": bool(
            getattr(
                args,
                "optimizer_resumed_actual",
                False,
            )
        ),
        "generator_backend": (
            "jax_cpu_vectorized_v4"
        ),
        "teacher_backend": (
            "jax_vectorized_heuristic_v2_1"
        ),
        "replay_backend": (
            "packed_device_resident_float32_v1"
        ),
        "learner_backend": (
            "cuda_graph_dynamic_next_mask_ddqn_v3"
        ),
        "qualification_status": (
            "UNQUALIFIED_CHALLENGER"
        ),
        "policy_observation_rule": (
            "Q sees current state243 + observable candidate215 only; "
            "preview successor queue/current/hold tail is never an action input"
        ),
        "runtime_meta": dict(
            runtime_meta
        ),
        "performance_design": {
            "jax_vectorized_generator_process": True,
            "long_trajectory_vector_envs": int(
                args.vector_envs
            ),
            "packed_device_replay": True,
            "cpu_affinity": str(getattr(args, "affinity", "off")),
            "shared_actor_weight_bank": True,
            "dynamic_no_replacement_sampling": True,
            "cuda_graph_ddqn_math": True,
            "graph_safe_normalized_policy": True,
            "graph_safe_grad_clip": True,
            "capturable_adamw": True,
            "non_destructive_graph_capture": True,
            "low_sync_metrics": True,
            "generator_learner_overlap": True,
            "fixed_sample_budget": int(
                args.sample_budget
            ),
        },
    }

    torch.save(
        payload,
        path,
    )


# Stable, version-free public names for successor training code.
CudaGraphLearner = CudaGraphDDQNLearner
save_training_checkpoint = save_checkpoint_v882
