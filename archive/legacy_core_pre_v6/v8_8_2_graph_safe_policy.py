from __future__ import annotations

import torch


def normalized_margin_actions_top4_graphsafe(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    gate: float,
    eps: float = 1.0e-6,
):
    """
    Graph-safe, bit-exact equivalent of
    v8_7_scale_invariant_policy.normalized_margin_actions_torch
    for the production K=4 successor layout.

    It preserves the original edge-case behavior when no alternative candidate
    is valid: best_alt_index falls back to teacher index 0, therefore raw_gap=0.

    Unlike the original helper, this function does not construct CUDA tensors
    from Python +/-inf values with torch.tensor(...) inside the captured region.
    """
    if q_values.ndim != 2:
        raise ValueError("q_values must have shape [B, K]")
    if mask.shape != q_values.shape:
        raise ValueError("mask must match q_values shape")
    if q_values.shape[1] != 4:
        raise ValueError(
            "V8.8.2 graph-safe policy is intentionally specialized to K=4"
        )
    if gate < 0.0:
        raise ValueError("gate must be >= 0")

    mask = mask.to(dtype=torch.bool)
    teacher_valid = mask[:, 0]

    alt_mask = mask[:, 1:]
    has_alt = alt_mask.any(dim=1)

    alt_q = q_values[:, 1:].masked_fill(
        ~alt_mask,
        float("-inf"),
    )
    best_rel = alt_q.argmax(dim=1)
    best_alt_index = torch.where(
        has_alt,
        best_rel + 1,
        torch.zeros_like(best_rel),
    )

    teacher_q = q_values[:, 0]
    best_alt_q = torch.gather(
        q_values,
        dim=1,
        index=best_alt_index.unsqueeze(1),
    ).squeeze(1)
    raw_gap = best_alt_q - teacher_q

    q_max = q_values.masked_fill(
        ~mask,
        float("-inf"),
    ).max(dim=1).values
    q_min = q_values.masked_fill(
        ~mask,
        float("inf"),
    ).min(dim=1).values
    q_span = q_max - q_min

    valid = (
        teacher_valid
        & has_alt
        & torch.isfinite(raw_gap)
        & torch.isfinite(q_span)
        & (raw_gap > float(eps))
        & (q_span > float(eps))
    )

    safe_span = torch.where(
        valid,
        q_span,
        torch.ones_like(q_span),
    )

    margin = torch.where(
        valid,
        raw_gap / safe_span,
        torch.zeros_like(raw_gap),
    ).clamp_(0.0, 1.0)

    allow_alt = valid & (margin >= float(gate))

    action_index = torch.where(
        allow_alt,
        best_alt_index,
        torch.zeros_like(best_alt_index),
    )

    return action_index, margin, raw_gap, q_span
