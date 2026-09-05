from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tetris_ai.learning.ranking import (
    RankingBatchMetrics,
    pairwise_logistic_ranking_loss,
    pairwise_ordering_accuracy_numpy,
)

class OfflineRankingCorpus:
    REQUIRED_KEYS = (
        "state",
        "candidates",
        "rewards",
        "teacher_scores",
        "teacher_ranks",
        "candidate_mask",
        "pair_targets",
        "split",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        device: torch.device,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Ranking corpus not found: {self.path}"
            )

        data = np.load(self.path, allow_pickle=False)
        missing = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing:
            raise RuntimeError(
                f"Ranking corpus missing keys: {missing}"
            )

        state = np.asarray(data["state"], dtype=np.float32)
        candidates = np.asarray(data["candidates"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32)
        teacher_scores = np.asarray(
            data["teacher_scores"], dtype=np.float32
        )
        teacher_ranks = np.asarray(
            data["teacher_ranks"], dtype=np.float32
        )
        candidate_mask = np.asarray(
            data["candidate_mask"], dtype=np.bool_
        )
        pair_targets = np.asarray(
            data["pair_targets"], dtype=np.int8
        )
        split = np.asarray(data["split"], dtype=np.int8)

        n = state.shape[0]
        expected = {
            "state": (n, 243),
            "candidates": (n, 4, 215),
            "rewards": (n, 4),
            "teacher_scores": (n, 4),
            "teacher_ranks": (n, 4),
            "candidate_mask": (n, 4),
            "pair_targets": (n, 6),
            "split": (n,),
        }
        actual = {
            "state": state.shape,
            "candidates": candidates.shape,
            "rewards": rewards.shape,
            "teacher_scores": teacher_scores.shape,
            "teacher_ranks": teacher_ranks.shape,
            "candidate_mask": candidate_mask.shape,
            "pair_targets": pair_targets.shape,
            "split": split.shape,
        }
        bad = {
            key: (actual[key], shape)
            for key, shape in expected.items()
            if actual[key] != shape
        }
        if bad:
            raise RuntimeError(
                f"Ranking corpus shape mismatch: {bad}"
            )

        if n < 2:
            raise RuntimeError(
                "Ranking corpus must contain at least 2 states."
            )
        if not np.all(np.isin(pair_targets, (-1, 0, 1))):
            raise RuntimeError(
                "pair_targets must contain only -1, 0, +1."
            )
        if not np.any(split == 0):
            raise RuntimeError("Ranking corpus has no training split.")
        if not np.any(split == 1):
            raise RuntimeError("Ranking corpus has no validation split.")

        self.device = device
        self.state = torch.from_numpy(state).to(device)
        self.candidates = torch.from_numpy(candidates).to(device)
        self.rewards = torch.from_numpy(rewards).to(device)
        self.teacher_scores = torch.from_numpy(teacher_scores).to(device)
        self.teacher_ranks = torch.from_numpy(teacher_ranks).to(device)
        self.candidate_mask = torch.from_numpy(candidate_mask).to(device)
        self.pair_targets = torch.from_numpy(pair_targets).to(device)
        self.split = torch.from_numpy(split).to(device)

        self.train_indices = torch.nonzero(
            self.split == 0,
            as_tuple=False,
        ).flatten()
        self.val_indices = torch.nonzero(
            self.split == 1,
            as_tuple=False,
        ).flatten()

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def indices_for_split(self, split: str) -> torch.Tensor:
        split = str(split).lower()
        if split == "train":
            return self.train_indices
        if split in {"val", "validation"}:
            return self.val_indices
        if split == "all":
            return torch.arange(
                len(self),
                device=self.device,
                dtype=torch.long,
            )
        raise ValueError(
            "split must be train, val/validation, or all."
        )

    def sample_indices(
        self,
        *,
        split: str,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        pool = self.indices_for_split(split)
        if pool.numel() == 0:
            raise RuntimeError(f"No examples in split={split!r}.")
        positions = torch.randint(
            0,
            int(pool.numel()),
            (int(batch_size),),
            device=self.device,
            generator=generator,
        )
        return pool[positions]


class RankingAuxTrainer:
    """
    Conservative V8.8.7 auxiliary updater.

    It uses the SAME online model and SAME optimizer as the production DDQN
    learner, but applies a low-frequency, low-weight pairwise ordering update
    from an offline counterfactual corpus.

    This class never modifies rewards, targets, replay contents, Teacher
    scores/ranks, policy gate, candidate contract, or target-network logic.
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        corpus: OfflineRankingCorpus,
        split: str = "train",
        batch_size: int = 32,
        weight: float = 0.02,
        temperature: float = 0.10,
        grad_clip: float = 1.0,
        seed: int = 20260831,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if weight < 0.0:
            raise ValueError("weight must be >= 0.")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0.")
        if grad_clip <= 0.0:
            raise ValueError("grad_clip must be > 0.")

        self.model = model
        self.optimizer = optimizer
        self.corpus = corpus
        self.split = split
        self.batch_size = int(batch_size)
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.grad_clip = float(grad_clip)

        generator_device = (
            corpus.device
            if corpus.device.type == "cuda"
            else torch.device("cpu")
        )
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(seed))

        self.updates = 0

    def _q(self, indices: torch.Tensor) -> torch.Tensor:
        return self.model(
            state=self.corpus.state[indices],
            candidates=self.corpus.candidates[indices],
            rewards=self.corpus.rewards[indices],
            teacher_scores=self.corpus.teacher_scores[indices],
            teacher_ranks=self.corpus.teacher_ranks[indices],
        )

    def step(
        self,
        *,
        collect_metrics: bool = False,
    ) -> Optional[RankingBatchMetrics]:
        indices = self.corpus.sample_indices(
            split=self.split,
            batch_size=self.batch_size,
            generator=self.generator,
        )

        q = self._q(indices)
        raw_loss, accuracy, valid_count = (
            pairwise_logistic_ranking_loss(
                q,
                self.corpus.pair_targets[indices],
                candidate_mask=self.corpus.candidate_mask[indices],
                temperature=self.temperature,
            )
        )
        scaled_loss = raw_loss * self.weight

        self.optimizer.zero_grad(set_to_none=True)
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip,
        )
        self.optimizer.step()
        self.updates += 1

        if not collect_metrics:
            return None

        with torch.no_grad():
            mask = self.corpus.candidate_mask[indices]
            q_max = q.masked_fill(~mask, -torch.inf).max(dim=1).values
            q_min = q.masked_fill(~mask, torch.inf).min(dim=1).values
            finite = torch.isfinite(q_max) & torch.isfinite(q_min)
            span = torch.where(
                finite,
                q_max - q_min,
                torch.zeros_like(q_max),
            ).mean()

        return RankingBatchMetrics(
            loss=float(raw_loss.detach().item()),
            pair_accuracy=float(accuracy.detach().item()),
            valid_pairs=int(valid_count.detach().item()),
            q_span=float(span.detach().item()),
        )


@torch.inference_mode()
def evaluate_ranking_corpus(
    model,
    corpus: OfflineRankingCorpus,
    *,
    split: str,
    batch_size: int = 256,
) -> dict:
    indices = corpus.indices_for_split(split)
    if indices.numel() == 0:
        raise RuntimeError(f"No states in split={split!r}.")

    q_rows = []
    target_rows = []
    mask_rows = []

    for start in range(0, int(indices.numel()), int(batch_size)):
        idx = indices[start : start + int(batch_size)]
        q = model(
            state=corpus.state[idx],
            candidates=corpus.candidates[idx],
            rewards=corpus.rewards[idx],
            teacher_scores=corpus.teacher_scores[idx],
            teacher_ranks=corpus.teacher_ranks[idx],
        )
        q_rows.append(q.detach().cpu().numpy())
        target_rows.append(
            corpus.pair_targets[idx].detach().cpu().numpy()
        )
        mask_rows.append(
            corpus.candidate_mask[idx].detach().cpu().numpy()
        )

    q_values = np.concatenate(q_rows, axis=0)
    targets = np.concatenate(target_rows, axis=0)
    mask = np.concatenate(mask_rows, axis=0)

    correct, total, accuracy = pairwise_ordering_accuracy_numpy(
        q_values,
        targets,
        mask,
    )

    return {
        "states": int(q_values.shape[0]),
        "pairwise_correct": int(correct),
        "pairwise_total": int(total),
        "pairwise_accuracy": float(accuracy),
        "q_span_mean": float(
            np.mean(
                np.nanmax(
                    np.where(mask, q_values, np.nan),
                    axis=1,
                )
                - np.nanmin(
                    np.where(mask, q_values, np.nan),
                    axis=1,
                )
            )
        ),
    }
