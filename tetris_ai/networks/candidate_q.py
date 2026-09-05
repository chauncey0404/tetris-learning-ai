from __future__ import annotations

import torch
import torch.nn as nn


class CandidateQNetwork(nn.Module):
    """Reusable candidate-scoring Q-network trunk.

    Game packages own their state/candidate contracts; this class only owns the
    neural architecture. Module names intentionally match V8.x so existing
    state_dict checkpoints retain bit-for-bit key compatibility.
    """

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
        teacher_score_scale: float = 1000.0,
        teacher_rank_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.state_size = int(state_size)
        self.candidate_size = int(candidate_size)
        self.teacher_score_scale = float(teacher_score_scale)
        self.teacher_rank_scale = float(teacher_rank_scale)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_size, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_size, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(67, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(),
        )
        self.q_head = nn.Linear(32, 1)
        nn.init.zeros_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)

    def forward(self, *, state, candidates, rewards, teacher_scores, teacher_ranks):
        if state.ndim != 2 or state.shape[-1] != self.state_size:
            raise ValueError(f"state must be (B,{self.state_size}); got {tuple(state.shape)}")
        if candidates.ndim != 3 or candidates.shape[-1] != self.candidate_size:
            raise ValueError(
                f"candidates must be (B,K,{self.candidate_size}); got {tuple(candidates.shape)}"
            )
        batch_size, candidate_count, _ = candidates.shape
        for name, value in (("rewards", rewards), ("teacher_scores", teacher_scores), ("teacher_ranks", teacher_ranks)):
            if value.shape != (batch_size, candidate_count):
                raise ValueError(f"{name} must be {(batch_size, candidate_count)}; got {tuple(value.shape)}")

        state_latent = self.state_encoder(state).unsqueeze(1).expand(-1, candidate_count, -1)
        flat_candidates = candidates.reshape(batch_size * candidate_count, self.candidate_size)
        candidate_latent = self.candidate_encoder(flat_candidates).reshape(batch_size, candidate_count, -1)
        joint_input = torch.cat(
            (
                state_latent,
                candidate_latent,
                rewards.unsqueeze(-1),
                (teacher_scores / self.teacher_score_scale).unsqueeze(-1),
                (teacher_ranks / self.teacher_rank_scale).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.q_head(self.joint(joint_input)).squeeze(-1)
