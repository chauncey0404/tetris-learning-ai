from __future__ import annotations

import torch
import torch.nn as nn

from tetrio.future.features import FEATURE_SIZE, normalize_future_features_torch
from tetrio.network.model_v1_1 import TetrioExpertV11Network
from tetrio.stateful.features import (
    STATEFUL_FEATURE_SIZE,
    normalize_stateful_features_torch,
)


class StatefulResidualReranker(nn.Module):
    """Candidate-specific residual that can only act through battle state.

    There is intentionally no candidate-only shortcut. The state branch uses
    bias-free linear layers, the interaction is multiplicative, and the output
    head has no bias. Therefore a ZERO STATE ablation produces exactly zero
    residual even after training.
    """

    EXTRA_SIZE = 3  # score delta, rank percentile, use_hold

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        max_adjustment: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_adjustment = float(max_adjustment)
        candidate_size = FEATURE_SIZE + self.EXTRA_SIZE

        self.state_encoder = nn.Sequential(
            nn.Linear(STATEFUL_FEATURE_SIZE, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Tanh(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_size, 1, bias=False)
        # E00 is exactly the frozen V1.1 policy.
        nn.init.zeros_(self.score_head.weight)

    def forward(
        self,
        *,
        v11_scores: torch.Tensor,
        raw_future_features: torch.Tensor,
        candidate_use_hold: torch.Tensor,
        battle_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if v11_scores.ndim != 2:
            raise ValueError("v11_scores must be [B,K]")
        if raw_future_features.ndim != 3:
            raise ValueError("raw_future_features must be [B,K,F]")
        if battle_state.ndim != 2:
            raise ValueError("battle_state must be [B,S]")
        if battle_state.shape[0] != v11_scores.shape[0]:
            raise ValueError("battle-state batch size differs from scores")

        norm_future = normalize_future_features_torch(raw_future_features)
        norm_state = normalize_stateful_features_torch(battle_state)

        masked = v11_scores.masked_fill(
            ~mask,
            torch.finfo(v11_scores.dtype).min,
        )
        best = masked.max(dim=1, keepdim=True).values
        score_delta = (v11_scores - best).clamp(-10.0, 0.0) / 5.0

        order = torch.argsort(masked, dim=1, descending=True)
        ranks = torch.empty_like(order, dtype=torch.float32)
        rank_values = torch.arange(
            v11_scores.shape[1],
            device=v11_scores.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand_as(ranks)
        ranks.scatter_(1, order, rank_values)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(2).float() - 1.0
        rank_pct = (ranks / denom).clamp(0.0, 1.0)

        candidate_input = torch.cat(
            (
                norm_future,
                score_delta.unsqueeze(-1),
                rank_pct.unsqueeze(-1).to(v11_scores.dtype),
                candidate_use_hold.to(v11_scores.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        candidate_latent = self.candidate_encoder(candidate_input)
        state_latent = self.state_encoder(norm_state).unsqueeze(1)
        interaction = candidate_latent * state_latent
        residual = self.max_adjustment * torch.tanh(
            self.score_head(interaction).squeeze(-1)
        )
        return residual.masked_fill(~mask, 0.0)


class TetrioExpertV12Network(nn.Module):
    """V1.2A = frozen V1.1 500K base + causal state interaction residual."""

    def __init__(
        self,
        *,
        v11_reranker_hidden_size: int = 96,
        v11_max_adjustment: float = 2.0,
        state_hidden_size: int = 64,
        state_max_adjustment: float = 1.0,
    ) -> None:
        super().__init__()
        self.base = TetrioExpertV11Network(
            reranker_hidden_size=v11_reranker_hidden_size,
            max_adjustment=v11_max_adjustment,
        )
        self.stateful = StatefulResidualReranker(
            hidden_size=state_hidden_size,
            max_adjustment=state_max_adjustment,
        )

    def freeze_base(self) -> None:
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)

    def final_scores_from_cached_base(
        self,
        *,
        base_scores: torch.Tensor,
        raw_future_features: torch.Tensor,
        candidate_use_hold: torch.Tensor,
        battle_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            v11_scores, _ = self.base.final_scores(
                base_scores=base_scores,
                raw_features=raw_future_features,
                candidate_use_hold=candidate_use_hold,
                mask=mask,
            )
        state_residual = self.stateful(
            v11_scores=v11_scores,
            raw_future_features=raw_future_features,
            candidate_use_hold=candidate_use_hold,
            battle_state=battle_state,
            mask=mask,
        )
        final = (v11_scores + state_residual).masked_fill(
            ~mask,
            torch.finfo(v11_scores.dtype).min,
        )
        return final, v11_scores, state_residual
