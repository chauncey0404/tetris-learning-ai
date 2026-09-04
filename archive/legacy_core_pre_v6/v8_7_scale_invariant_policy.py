from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


DEFAULT_EPS = 1e-6


def normalized_margin_choice(
    q_values,
    gate: float,
    eps: float = DEFAULT_EPS,
) -> Tuple[int, float]:
    """
    Scale/offset-invariant conservative choice.

    Candidate 0 is the Teacher action.
    We consider the best non-Teacher candidate and compute:

        raw_gap = best_alt_q - teacher_q
        q_span  = max(q) - min(q)
        margin  = raw_gap / q_span

    For any positive affine transform q' = a*q + b (a > 0),
    both raw_gap and q_span are multiplied by a, so margin is unchanged.

    Returns:
        (chosen_index, normalized_margin)

    The Teacher is kept when:
      - no alternative exists,
      - best alternative does not beat Teacher,
      - Q spread is numerically degenerate,
      - normalized margin is below gate.
    """
    q = np.asarray(q_values, dtype=np.float64).reshape(-1)

    if q.size <= 1:
        return 0, 0.0

    if gate < 0.0:
        raise ValueError("gate must be >= 0")

    if not np.all(np.isfinite(q)):
        raise ValueError("q_values must all be finite")

    teacher_q = float(q[0])
    best_alt_index = int(np.argmax(q[1:])) + 1
    best_alt_q = float(q[best_alt_index])

    raw_gap = best_alt_q - teacher_q
    q_span = float(np.max(q) - np.min(q))

    if raw_gap <= eps or q_span <= eps:
        return 0, 0.0

    margin = raw_gap / q_span

    # Numerical clipping only. With best_alt_q > teacher_q and best_alt being
    # the global maximum, the mathematical value is in (0, 1].
    margin = float(np.clip(margin, 0.0, 1.0))

    if margin >= float(gate):
        return best_alt_index, margin

    return 0, margin


def normalized_margin_actions_torch(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    gate: float,
    eps: float = DEFAULT_EPS,
):
    """
    Batched torch version for DDQN next-action selection.

    Args:
        q_values: [B, K]
        mask:      [B, K] bool, True for valid candidates
        gate:      frozen normalized confidence threshold

    Returns:
        action_index: [B] long
        margin:       [B] float
        raw_gap:      [B] float
        q_span:       [B] float
    """
    if q_values.ndim != 2:
        raise ValueError("q_values must have shape [B, K]")
    if mask.shape != q_values.shape:
        raise ValueError("mask must match q_values shape")
    if q_values.shape[1] < 1:
        raise ValueError("q_values must contain at least the Teacher candidate")
    if gate < 0.0:
        raise ValueError("gate must be >= 0")

    mask = mask.to(dtype=torch.bool)
    b, k = q_values.shape
    device = q_values.device

    teacher_valid = mask[:, 0]

    if k == 1:
        zeros_long = torch.zeros(b, dtype=torch.long, device=device)
        zeros = torch.zeros(b, dtype=q_values.dtype, device=device)
        return zeros_long, zeros, zeros, zeros

    alt_mask = mask.clone()
    alt_mask[:, 0] = False
    has_alt = alt_mask.any(dim=1)

    neg_inf = torch.tensor(
        float("-inf"),
        dtype=q_values.dtype,
        device=device,
    )
    pos_inf = torch.tensor(
        float("inf"),
        dtype=q_values.dtype,
        device=device,
    )

    alt_q = q_values.masked_fill(~alt_mask, neg_inf)
    best_alt_index = alt_q.argmax(dim=1)

    rows = torch.arange(b, device=device)
    teacher_q = q_values[:, 0]
    best_alt_q = q_values[rows, best_alt_index]
    raw_gap = best_alt_q - teacher_q

    q_max = q_values.masked_fill(~mask, neg_inf).max(dim=1).values
    q_min = q_values.masked_fill(~mask, pos_inf).min(dim=1).values
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

    teacher_index = torch.zeros(
        b,
        dtype=torch.long,
        device=device,
    )

    action_index = torch.where(
        allow_alt,
        best_alt_index,
        teacher_index,
    )

    return action_index, margin, raw_gap, q_span
