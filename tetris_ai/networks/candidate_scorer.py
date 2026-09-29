from __future__ import annotations

import torch
import torch.nn as nn


class CandidateScoringNetwork(nn.Module):
    """Generic state-conditioned candidate scorer.

    Game packages own the state/candidate feature contracts.  This shared
    module only maps dense state/candidate vectors to one score per candidate.

    Unlike the V8 ``CandidateQNetwork``, this scorer has no reward/Teacher
    inputs.  It is suitable for leakage-safe imitation/ranking pipelines.
    """

    def __init__(
        self,
        *,
        state_size: int,
        candidate_size: int,
        latent_size: int = 256,
        hidden_size: int = 512,
        joint_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_size = int(state_size)
        self.candidate_size = int(candidate_size)
        self.latent_size = int(latent_size)

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_size),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_size),
            nn.GELU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(self.latent_size * 2, joint_hidden_size),
            nn.GELU(),
            nn.Linear(joint_hidden_size, 128),
            nn.GELU(),
        )
        self.score_head = nn.Linear(128, 1)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != self.state_size:
            raise ValueError(
                f"state must be (B,{self.state_size}); got {tuple(state.shape)}"
            )
        return self.state_encoder(state)

    def score_flat_from_state_latent(
        self,
        state_latent: torch.Tensor,
        candidates: torch.Tensor,
        candidate_owner: torch.Tensor,
    ) -> torch.Tensor:
        """Score only real candidates, with no [B,K,C] padding work.

        ``candidate_owner[n]`` selects the state latent for flat candidate n.
        """
        if state_latent.ndim != 2 or state_latent.shape[-1] != self.latent_size:
            raise ValueError(
                f"state_latent must be (B,{self.latent_size}); "
                f"got {tuple(state_latent.shape)}"
            )
        if candidates.ndim != 2 or candidates.shape[-1] != self.candidate_size:
            raise ValueError(
                f"flat candidates must be (N,{self.candidate_size}); "
                f"got {tuple(candidates.shape)}"
            )
        if candidate_owner.ndim != 1 or candidate_owner.shape[0] != candidates.shape[0]:
            raise ValueError("candidate_owner must have shape [N]")

        candidate_latent = self.candidate_encoder(candidates)
        selected_state = state_latent.index_select(
            0,
            candidate_owner.to(dtype=torch.long),
        )
        joint = torch.cat((selected_state, candidate_latent), dim=-1)
        return self.score_head(self.joint(joint)).squeeze(-1)

    def score_from_state_latent(
        self,
        state_latent: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if state_latent.ndim != 2 or state_latent.shape[-1] != self.latent_size:
            raise ValueError(
                f"state_latent must be (B,{self.latent_size}); "
                f"got {tuple(state_latent.shape)}"
            )
        if candidates.ndim != 3 or candidates.shape[-1] != self.candidate_size:
            raise ValueError(
                f"candidates must be (B,K,{self.candidate_size}); "
                f"got {tuple(candidates.shape)}"
            )
        if candidates.shape[0] != state_latent.shape[0]:
            raise ValueError("state/candidate batch sizes differ")

        batch_size, candidate_count, _ = candidates.shape
        flat = candidates.reshape(batch_size * candidate_count, self.candidate_size)
        candidate_latent = self.candidate_encoder(flat).reshape(
            batch_size, candidate_count, self.latent_size
        )
        expanded_state = state_latent.unsqueeze(1).expand(-1, candidate_count, -1)
        joint = torch.cat((expanded_state, candidate_latent), dim=-1)
        return self.score_head(self.joint(joint)).squeeze(-1)

    def forward(self, *, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.score_from_state_latent(self.encode_state(state), candidates)
