from __future__ import annotations

import torch
import torch.nn as nn

from tetrio.future.features import (
    FEATURE_SIZE,
    normalize_future_features_torch,
)
from tetrio.network.encoding import CANDIDATE_SIZE, STATE_SIZE
from tetris_ai.networks import CandidateScoringNetwork


class FutureResidualReranker(nn.Module):
    """Small bounded residual on top of the frozen Expert-v1 base score."""

    EXTRA_SIZE = 3  # base delta from best, rank percentile, use_hold

    def __init__(
        self,
        *,
        hidden_size: int = 96,
        max_adjustment: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_adjustment = float(max_adjustment)
        self.net = nn.Sequential(
            nn.Linear(FEATURE_SIZE + self.EXTRA_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        # Epoch-0 is exactly Expert-v1: residual starts at zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        *,
        base_scores: torch.Tensor,
        raw_features: torch.Tensor,
        candidate_use_hold: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if base_scores.ndim != 2:
            raise ValueError("base_scores must be [B,K]")
        if raw_features.ndim != 3:
            raise ValueError("raw_features must be [B,K,F]")

        norm = normalize_future_features_torch(raw_features)
        masked = base_scores.masked_fill(
            ~mask,
            torch.finfo(base_scores.dtype).min,
        )
        best = masked.max(dim=1, keepdim=True).values
        base_delta = (base_scores - best).clamp(-10.0, 0.0) / 5.0

        # Rank percentile in [0,1], 0 = best base candidate.
        order = torch.argsort(masked, dim=1, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        values = torch.arange(
            base_scores.shape[1],
            device=base_scores.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand_as(ranks)
        ranks.scatter_(1, order, values)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(2).to(torch.float32) - 1.0
        rank_pct = (ranks / denom).clamp(0.0, 1.0)

        hold = candidate_use_hold.to(base_scores.dtype)
        x = torch.cat(
            (
                norm,
                base_delta.unsqueeze(-1),
                rank_pct.unsqueeze(-1).to(base_scores.dtype),
                hold.unsqueeze(-1),
            ),
            dim=-1,
        )
        residual = (
            self.max_adjustment
            * torch.tanh(self.net(x).squeeze(-1))
        )
        return residual.masked_fill(~mask, 0.0)


class TetrioExpertV11Network(nn.Module):
    """Expert-v1.1 = frozen V1 scorer + future-aware residual reranker."""

    def __init__(
        self,
        *,
        reranker_hidden_size: int = 96,
        max_adjustment: float = 2.0,
    ) -> None:
        super().__init__()
        self.scorer = CandidateScoringNetwork(
            state_size=STATE_SIZE,
            candidate_size=CANDIDATE_SIZE,
            latent_size=256,
            hidden_size=512,
            joint_hidden_size=256,
        )
        self.reranker = FutureResidualReranker(
            hidden_size=reranker_hidden_size,
            max_adjustment=max_adjustment,
        )

    def freeze_scorer(self) -> None:
        self.scorer.eval()
        for p in self.scorer.parameters():
            p.requires_grad_(False)

    def final_scores(
        self,
        *,
        base_scores: torch.Tensor,
        raw_features: torch.Tensor,
        candidate_use_hold: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.reranker(
            base_scores=base_scores,
            raw_features=raw_features,
            candidate_use_hold=candidate_use_hold,
            mask=mask,
        )
        final = (base_scores + residual).masked_fill(
            ~mask,
            torch.finfo(base_scores.dtype).min,
        )
        return final, residual
