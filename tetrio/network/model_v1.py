from __future__ import annotations

import torch
import torch.nn as nn

from tetris_ai.networks import CandidateScoringNetwork
from tetrio.network.encoding import CANDIDATE_SIZE, STATE_SIZE


class TetrioExpertV1Network(nn.Module):
    """Unified Expert-v1 candidate scorer.

    V0 first predicted HOLD/NO-HOLD with a separate binary head and then ranked
    placements only inside that chosen branch.

    V1 deliberately removes that staging.  Both no-hold and hold candidates
    are presented to the same CandidateScoringNetwork and compete in one
    listwise ranking.  ``candidate_use_hold`` is already part of the candidate
    feature contract, while state contains Active/Hold/Next5.
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

    def forward(
        self,
        *,
        state: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return self.scorer(state=state, candidates=candidates)

    def forward_flat(
        self,
        *,
        state: torch.Tensor,
        candidates: torch.Tensor,
        candidate_owner: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.scorer.encode_state(state)
        return self.scorer.score_flat_from_state_latent(
            latent,
            candidates,
            candidate_owner,
        )
