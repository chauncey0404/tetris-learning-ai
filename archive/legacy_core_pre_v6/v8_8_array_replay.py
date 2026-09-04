
from __future__ import annotations

"""
Preallocated NumPy replay buffer for V8.8.

The old ObservableReplayBuffer stores Python dict objects and stacks them at
sample time. That was acceptable at ~371 transitions/s, but becomes a major
CPU/GC bottleneck once the generator is vectorized. V8.8 stores contiguous
arrays and supports add_batch().
"""

import numpy as np


class V88ArrayReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_size: int = 243,
        candidate_size: int = 215,
        top_k: int = 4,
    ):
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.candidate_size = int(candidate_size)
        self.top_k = int(top_k)

        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")

        c = self.capacity
        s = self.state_size
        a = self.candidate_size
        k = self.top_k

        self.state = np.empty((c, s), dtype=np.float32)
        self.candidate = np.empty((c, a), dtype=np.float32)
        self.reward = np.empty(c, dtype=np.float32)
        self.teacher_score = np.empty(c, dtype=np.float32)
        self.teacher_rank = np.empty(c, dtype=np.float32)
        self.done = np.empty(c, dtype=np.float32)

        self.next_state = np.empty((c, s), dtype=np.float32)
        self.next_candidates = np.empty((c, k, a), dtype=np.float32)
        self.next_rewards = np.empty((c, k), dtype=np.float32)
        self.next_teacher_scores = np.empty((c, k), dtype=np.float32)
        self.next_teacher_ranks = np.empty((c, k), dtype=np.float32)
        self.next_mask = np.empty((c, k), dtype=np.bool_)

        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    @property
    def nbytes(self) -> int:
        arrays = (
            self.state,
            self.candidate,
            self.reward,
            self.teacher_score,
            self.teacher_rank,
            self.done,
            self.next_state,
            self.next_candidates,
            self.next_rewards,
            self.next_teacher_scores,
            self.next_teacher_ranks,
            self.next_mask,
        )
        return int(sum(x.nbytes for x in arrays))

    def _write_indices(self, n: int):
        return (self.position + np.arange(n, dtype=np.int64)) % self.capacity

    def add_batch(
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
    ):
        state = np.asarray(state, dtype=np.float32)
        n = int(state.shape[0])

        if n <= 0:
            return

        if n > self.capacity:
            # Keep the newest capacity items.
            start = n - self.capacity
            return self.add_batch(
                state=state[start:],
                candidate=np.asarray(candidate)[start:],
                reward=np.asarray(reward)[start:],
                teacher_score=np.asarray(teacher_score)[start:],
                teacher_rank=np.asarray(teacher_rank)[start:],
                done=np.asarray(done)[start:],
                next_state=np.asarray(next_state)[start:],
                next_candidates=np.asarray(next_candidates)[start:],
                next_rewards=np.asarray(next_rewards)[start:],
                next_teacher_scores=np.asarray(next_teacher_scores)[start:],
                next_teacher_ranks=np.asarray(next_teacher_ranks)[start:],
                next_mask=np.asarray(next_mask)[start:],
            )

        idx = self._write_indices(n)

        self.state[idx] = state
        self.candidate[idx] = np.asarray(candidate, dtype=np.float32)
        self.reward[idx] = np.asarray(reward, dtype=np.float32)
        self.teacher_score[idx] = np.asarray(teacher_score, dtype=np.float32)
        self.teacher_rank[idx] = np.asarray(teacher_rank, dtype=np.float32)
        self.done[idx] = np.asarray(done, dtype=np.float32)
        self.next_state[idx] = np.asarray(next_state, dtype=np.float32)
        self.next_candidates[idx] = np.asarray(next_candidates, dtype=np.float32)
        self.next_rewards[idx] = np.asarray(next_rewards, dtype=np.float32)
        self.next_teacher_scores[idx] = np.asarray(
            next_teacher_scores,
            dtype=np.float32,
        )
        self.next_teacher_ranks[idx] = np.asarray(
            next_teacher_ranks,
            dtype=np.float32,
        )
        self.next_mask[idx] = np.asarray(next_mask, dtype=np.bool_)

        self.position = int((self.position + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def add_terminal_extras(self, batch: dict, copies: int) -> int:
        """
        Add (copies-1) extra copies of terminal/no-next transitions.
        Returns number of extra replay entries inserted.
        """
        copies = int(copies)
        if copies <= 1:
            return 0

        done_mask = np.asarray(batch["done"], dtype=np.bool_)
        terminal_idx = np.flatnonzero(done_mask)
        if terminal_idx.size == 0:
            return 0

        extras = 0
        for _ in range(copies - 1):
            kwargs = {
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
            self.add_batch(**kwargs)
            extras += int(terminal_idx.size)

        return extras

    def sample(self, batch_size: int, rng: np.random.Generator):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.size < batch_size:
            raise ValueError(
                f"not enough replay items: {self.size} < {batch_size}"
            )

        idx = rng.choice(
            self.size,
            size=batch_size,
            replace=False,
        )

        # Advanced indexing returns contiguous sample copies, ready for torch.
        return {
            "state": self.state[idx],
            "candidate": self.candidate[idx],
            "reward": self.reward[idx],
            "teacher_score": self.teacher_score[idx],
            "teacher_rank": self.teacher_rank[idx],
            "done": self.done[idx],
            "next_state": self.next_state[idx],
            "next_candidates": self.next_candidates[idx],
            "next_rewards": self.next_rewards[idx],
            "next_teacher_scores": self.next_teacher_scores[idx],
            "next_teacher_ranks": self.next_teacher_ranks[idx],
            "next_mask": self.next_mask[idx],
        }
