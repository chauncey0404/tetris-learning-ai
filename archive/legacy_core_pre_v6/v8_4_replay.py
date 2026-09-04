import numpy as np


class ObservableReplayBuffer:
    """Replay buffer with candidate features separated from real next state."""

    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.data = []
        self.position = 0

    def __len__(self):
        return len(self.data)

    def add(
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
        item = {
            "state": np.asarray(state, dtype=np.float32).copy(),
            "candidate": np.asarray(candidate, dtype=np.float32).copy(),
            "reward": float(reward),
            "teacher_score": float(teacher_score),
            "teacher_rank": float(teacher_rank),
            "done": float(bool(done)),
            "next_state": np.asarray(next_state, dtype=np.float32).copy(),
            "next_candidates": np.asarray(next_candidates, dtype=np.float32).copy(),
            "next_rewards": np.asarray(next_rewards, dtype=np.float32).copy(),
            "next_teacher_scores": np.asarray(
                next_teacher_scores, dtype=np.float32
            ).copy(),
            "next_teacher_ranks": np.asarray(
                next_teacher_ranks, dtype=np.float32
            ).copy(),
            "next_mask": np.asarray(next_mask, dtype=np.bool_).copy(),
        }

        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self.position] = item

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, rng):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self.data) < batch_size:
            raise ValueError(
                f"not enough replay items: {len(self.data)} < {batch_size}"
            )

        indices = rng.choice(len(self.data), size=batch_size, replace=False)
        items = [self.data[int(index)] for index in indices]

        return {
            "state": np.stack([x["state"] for x in items]),
            "candidate": np.stack([x["candidate"] for x in items]),
            "reward": np.asarray([x["reward"] for x in items], dtype=np.float32),
            "teacher_score": np.asarray(
                [x["teacher_score"] for x in items], dtype=np.float32
            ),
            "teacher_rank": np.asarray(
                [x["teacher_rank"] for x in items], dtype=np.float32
            ),
            "done": np.asarray([x["done"] for x in items], dtype=np.float32),
            "next_state": np.stack([x["next_state"] for x in items]),
            "next_candidates": np.stack(
                [x["next_candidates"] for x in items]
            ),
            "next_rewards": np.stack([x["next_rewards"] for x in items]),
            "next_teacher_scores": np.stack(
                [x["next_teacher_scores"] for x in items]
            ),
            "next_teacher_ranks": np.stack(
                [x["next_teacher_ranks"] for x in items]
            ),
            "next_mask": np.stack([x["next_mask"] for x in items]),
        }
