from __future__ import annotations

import torch

from tetrio.future.features import feature_index


H = feature_index("holes_after")
HEIGHT = feature_index("max_height_after")
NEXT_H = feature_index("next_min_holes")
NEXT_HEIGHT = feature_index("next_min_height")
LINES = feature_index("lines_now")
T_AFTER = feature_index("t_proxy_after")
T_DESTROYED = feature_index("t_opportunity_destroyed")
T_DEFERRED = feature_index("t_cashout_deferred")
DEAD = feature_index("next_dead_end")


def safe_dominance_mask(
    raw_features: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return [B,K,K] where [b,i,j] means candidate i safely dominates j.

    This is intentionally conservative.  A candidate must be no worse on:
    holes, height, one-step recoverability, immediate lines, tactical T value,
    and dead-end status; at least one criterion must be strictly better.
    """
    if raw_features.ndim != 3:
        raise ValueError("raw_features must be [B,K,F]")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be [B,K]")

    f = raw_features
    a = f.unsqueeze(2)  # [B,K,1,F]
    b = f.unsqueeze(1)  # [B,1,K,F]

    tactical_a = (
        0.25 * a[..., T_AFTER]
        - 2.0 * a[..., T_DESTROYED]
        - 1.0 * a[..., T_DEFERRED]
    )
    tactical_b = (
        0.25 * b[..., T_AFTER]
        - 2.0 * b[..., T_DESTROYED]
        - 1.0 * b[..., T_DEFERRED]
    )

    no_worse = (
        (a[..., H] <= b[..., H])
        & (a[..., HEIGHT] <= b[..., HEIGHT])
        & (a[..., NEXT_H] <= b[..., NEXT_H])
        & (a[..., NEXT_HEIGHT] <= b[..., NEXT_HEIGHT])
        & (a[..., LINES] >= b[..., LINES])
        & (a[..., DEAD] <= b[..., DEAD])
        & (tactical_a >= tactical_b)
    )

    strict = (
        (a[..., H] < b[..., H])
        | (a[..., HEIGHT] < b[..., HEIGHT])
        | (a[..., NEXT_H] < b[..., NEXT_H])
        | (a[..., NEXT_HEIGHT] < b[..., NEXT_HEIGHT])
        | (a[..., LINES] > b[..., LINES])
        | (a[..., DEAD] < b[..., DEAD])
        | (tactical_a > tactical_b)
    )

    valid_pair = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    eye = torch.eye(
        valid_mask.shape[1],
        device=valid_mask.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    return no_worse & strict & valid_pair & ~eye


def dominance_margin_loss(
    scores: torch.Tensor,
    raw_features: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    margin: float = 0.20,
) -> tuple[torch.Tensor, int]:
    dom = safe_dominance_mask(raw_features, valid_mask)
    if not dom.any():
        return scores.sum() * 0.0, 0

    winner = scores.unsqueeze(2)
    loser = scores.unsqueeze(1)
    loss = torch.relu(loser - winner + float(margin))
    return loss[dom].mean(), int(dom.sum().item())
