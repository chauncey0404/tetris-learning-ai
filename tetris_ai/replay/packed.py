from __future__ import annotations

"""
V8.8.1 packed device-resident replay buffer.

Why this exists
---------------
V8.8 removed Python-object replay, but every gradient still gathered 12 NumPy
arrays on CPU and copied all of them to CUDA again. At 8K+ learner batches that
host gather / PCIe path dominates once the JAX generator is fast.

V8.8.1 stores each transition exactly once in one packed float32 tensor on the
learner device. Sampling performs one device-side gather; all network inputs are
views into that gathered tensor. next_mask is stored as 0/1 float32 in the pack
and exposed as bool to preserve the learner API.

No observable-state or TD semantics are changed.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass(frozen=True)
class PackedLayout:
    state_size: int = 243
    candidate_size: int = 215
    top_k: int = 4

    @property
    def offsets(self):
        o = 0
        result = {}

        def take(name, width):
            nonlocal o
            result[name] = (o, o + width)
            o += width

        take("state", self.state_size)
        take("candidate", self.candidate_size)
        take("reward", 1)
        take("teacher_score", 1)
        take("teacher_rank", 1)
        take("done", 1)
        take("next_state", self.state_size)
        take("next_candidates", self.top_k * self.candidate_size)
        take("next_rewards", self.top_k)
        take("next_teacher_scores", self.top_k)
        take("next_teacher_ranks", self.top_k)
        take("next_mask", self.top_k)
        return result

    @property
    def width(self):
        return self.offsets["next_mask"][1]


class V881PackedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device,
        *,
        state_size: int = 243,
        candidate_size: int = 215,
        top_k: int = 4,
        seed: int = 20260823,
    ):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.layout = PackedLayout(
            state_size=int(state_size),
            candidate_size=int(candidate_size),
            top_k=int(top_k),
        )
        self.position = 0
        self.size = 0

        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        self.data = torch.empty(
            (self.capacity, self.layout.width),
            dtype=torch.float32,
            device=self.device,
        )

        try:
            self.generator = torch.Generator(device=self.device)
        except TypeError:
            # Older CPU-only PyTorch accepts no device argument here.
            self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

    def __len__(self):
        return int(self.size)

    @property
    def nbytes(self):
        return int(self.data.numel() * self.data.element_size())

    @property
    def packed_width(self):
        return int(self.layout.width)

    def _pack_numpy(
        self,
        *,
        state,
        candidate,
        reward,
        teacher_score,
        teacher_rank,
        done,
        next_state,
        next_candidates,
        next_rewards,
        next_teacher_scores,
        next_teacher_ranks,
        next_mask,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        n = int(state.shape[0])
        if n <= 0:
            return np.empty((0, self.layout.width), dtype=np.float32)

        s = self.layout.state_size
        a = self.layout.candidate_size
        k = self.layout.top_k

        if state.shape != (n, s):
            raise ValueError(f"state shape mismatch: {state.shape}")
        if np.asarray(candidate).shape != (n, a):
            raise ValueError(f"candidate shape mismatch: {np.asarray(candidate).shape}")
        if np.asarray(next_state).shape != (n, s):
            raise ValueError(f"next_state shape mismatch: {np.asarray(next_state).shape}")
        if np.asarray(next_candidates).shape != (n, k, a):
            raise ValueError(
                f"next_candidates shape mismatch: {np.asarray(next_candidates).shape}"
            )
        for name, value in (
            ("next_rewards", next_rewards),
            ("next_teacher_scores", next_teacher_scores),
            ("next_teacher_ranks", next_teacher_ranks),
            ("next_mask", next_mask),
        ):
            if np.asarray(value).shape != (n, k):
                raise ValueError(f"{name} shape mismatch: {np.asarray(value).shape}")

        packed = np.empty((n, self.layout.width), dtype=np.float32)
        off = self.layout.offsets

        def put(name, value):
            lo, hi = off[name]
            packed[:, lo:hi] = np.asarray(value, dtype=np.float32).reshape(n, hi - lo)

        put("state", state)
        put("candidate", candidate)
        put("reward", reward)
        put("teacher_score", teacher_score)
        put("teacher_rank", teacher_rank)
        put("done", done)
        put("next_state", next_state)
        put("next_candidates", next_candidates)
        put("next_rewards", next_rewards)
        put("next_teacher_scores", next_teacher_scores)
        put("next_teacher_ranks", next_teacher_ranks)
        put("next_mask", np.asarray(next_mask, dtype=np.float32))
        return np.ascontiguousarray(packed)

    @torch.no_grad()
    def _write_device_rows(self, src: torch.Tensor):
        n = int(src.shape[0])
        if n <= 0:
            return
        if n > self.capacity:
            src = src[-self.capacity :]
            n = self.capacity

        first = min(n, self.capacity - self.position)
        self.data[self.position : self.position + first].copy_(src[:first])
        remaining = n - first
        if remaining:
            self.data[:remaining].copy_(src[first:])

        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    @torch.no_grad()
    def add_batch(self, **kwargs):
        packed_np = self._pack_numpy(**kwargs)
        if packed_np.shape[0] == 0:
            return
        src = torch.from_numpy(packed_np).to(self.device)
        self._write_device_rows(src)

    def add_terminal_extras(self, batch: dict, copies: int) -> int:
        copies = int(copies)
        if copies <= 1:
            return 0

        done_mask = np.asarray(batch["done"], dtype=np.bool_)
        terminal_idx = np.flatnonzero(done_mask)
        if terminal_idx.size == 0:
            return 0

        subset = {
            key: np.asarray(value)[terminal_idx]
            for key, value in batch.items()
            if key in {
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
            }
        }
        packed_np = self._pack_numpy(**subset)
        src = torch.from_numpy(packed_np).to(self.device)

        for _ in range(copies - 1):
            self._write_device_rows(src)

        return int(terminal_idx.size * (copies - 1))

    def _views(self, packed: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = int(packed.shape[0])
        k = self.layout.top_k
        a = self.layout.candidate_size
        off = self.layout.offsets

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
            "next_mask": view("next_mask").view(n, k) > 0.5,
        }

    @torch.no_grad()
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.size < batch_size:
            raise ValueError(
                f"not enough replay items: {self.size} < {batch_size}"
            )

        # Preserve V8.8's no-replacement sampling semantics.
        idx = torch.randperm(
            self.size,
            device=self.device,
            generator=self.generator,
        )[:batch_size]
        packed = self.data.index_select(0, idx)
        return self._views(packed)

    @torch.no_grad()
    def debug_unpack_rows(self, start: int = 0, count: int = 1):
        """CPU copy for parity/smoke tests only."""
        start = int(start)
        count = int(count)
        if start < 0 or count < 0 or start + count > self.size:
            raise ValueError("debug row range is outside replay size")
        packed = self.data[start : start + count]
        return {k: v.detach().cpu().numpy() for k, v in self._views(packed).items()}


# Stable, version-free public names.
PackedReplayBuffer = V881PackedReplayBuffer
ReplayLayout = PackedLayout
