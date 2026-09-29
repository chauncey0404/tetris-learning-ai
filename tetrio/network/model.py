from __future__ import annotations

import torch
import torch.nn as nn

from tetris_ai.networks import CandidateScoringNetwork
from tetrio.network.encoding import CANDIDATE_SIZE, STATE_SIZE


class TetrioExpertV0Network(nn.Module):
    """Staged Expert-v0 imitation network.

    Placement ranking is trained inside the expert-selected hold branch.  A
    separate state-only hold head learns the binary hold decision.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scorer = CandidateScoringNetwork(
            state_size=STATE_SIZE,
            candidate_size=CANDIDATE_SIZE,
            latent_size=256,
            hidden_size=512,
            joint_hidden_size=256,
        )
        self.hold_head = nn.Sequential(
            nn.Linear(self.scorer.latent_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        *,
        state: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_latent = self.scorer.encode_state(state)
        scores = self.scorer.score_from_state_latent(state_latent, candidates)
        hold_logit = self.hold_head(state_latent).squeeze(-1)
        return scores, hold_logit

    def forward_flat(
        self,
        *,
        state: torch.Tensor,
        candidates: torch.Tensor,
        candidate_owner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact training path: score only real flat candidates."""
        state_latent = self.scorer.encode_state(state)
        scores = self.scorer.score_flat_from_state_latent(
            state_latent,
            candidates,
            candidate_owner,
        )
        hold_logit = self.hold_head(state_latent).squeeze(-1)
        return scores, hold_logit
