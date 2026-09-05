from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


PAIR_I = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
PAIR_J = torch.tensor([1, 2, 3, 2, 3, 3], dtype=torch.long)


@dataclass
class RankingBatchMetrics:
    loss: float
    pair_accuracy: float
    valid_pairs: int
    q_span: float


def pairwise_logistic_ranking_loss(
    q_values: torch.Tensor,
    pair_targets: torch.Tensor,
    *,
    candidate_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pairwise auxiliary loss for top-4 candidate Q ordering.

    Args
    ----
    q_values:
        [B,4] predicted Q values.
    pair_targets:
        [B,6] values in {-1,0,+1}, using canonical pairs:
        (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
        +1 means the first member of the pair had the better realized
        counterfactual outcome; -1 means the second was better; 0 means tie
        or unavailable and is ignored.
    candidate_mask:
        Optional [B,4] reachable-candidate mask.
    temperature:
        Positive scale for the logistic ordering loss.

    Returns
    -------
    loss:
        Mean logistic ranking loss over valid non-tied pairs.
    accuracy:
        Fraction of valid pairs whose predicted Q ordering matches the target.
    valid_pair_count:
        Scalar tensor with the number of supervised pairs.

    Notes
    -----
    Realized counterfactual VALUE is never inserted into the TD reward.
    This objective supervises only relative candidate ordering.
    """
    if q_values.ndim != 2 or q_values.shape[1] != 4:
        raise ValueError("q_values must have shape [B,4].")
    if pair_targets.ndim != 2 or pair_targets.shape[1] != 6:
        raise ValueError("pair_targets must have shape [B,6].")
    if pair_targets.shape[0] != q_values.shape[0]:
        raise ValueError("q_values and pair_targets batch sizes differ.")
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    device = q_values.device
    pi = PAIR_I.to(device=device)
    pj = PAIR_J.to(device=device)

    qi = q_values[:, pi]
    qj = q_values[:, pj]

    targets = pair_targets.to(
        device=device,
        dtype=q_values.dtype,
    )
    valid = targets != 0

    if candidate_mask is not None:
        mask = candidate_mask.to(device=device, dtype=torch.bool)
        if mask.shape != q_values.shape:
            raise ValueError("candidate_mask must have shape [B,4].")
        valid = valid & mask[:, pi] & mask[:, pj]

    signed_gap = targets * (qi - qj) / float(temperature)
    per_pair = F.softplus(-signed_gap)

    valid_float = valid.to(dtype=q_values.dtype)
    denom = valid_float.sum().clamp_min(1.0)
    loss = (per_pair * valid_float).sum() / denom

    with torch.no_grad():
        predicted = torch.sign(qi - qj)
        correct = (predicted == targets) & valid
        accuracy = (
            correct.to(dtype=q_values.dtype).sum() / denom
        )
        valid_count = valid.sum()

    return loss, accuracy, valid_count


def pairwise_ordering_accuracy_numpy(
    q_values: np.ndarray,
    pair_targets: np.ndarray,
    candidate_mask: Optional[np.ndarray] = None,
) -> tuple[int, int, float]:
    q = np.asarray(q_values, dtype=np.float64)
    targets = np.asarray(pair_targets, dtype=np.int8)

    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("q_values must be [N,4].")
    if targets.shape != (q.shape[0], 6):
        raise ValueError("pair_targets must be [N,6].")

    pi = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
    pj = np.asarray([1, 2, 3, 2, 3, 3], dtype=np.int64)

    valid = targets != 0
    if candidate_mask is not None:
        mask = np.asarray(candidate_mask, dtype=bool)
        if mask.shape != q.shape:
            raise ValueError("candidate_mask must be [N,4].")
        valid &= mask[:, pi] & mask[:, pj]

    predicted = np.sign(q[:, pi] - q[:, pj]).astype(np.int8)
    correct = int(np.count_nonzero((predicted == targets) & valid))
    total = int(np.count_nonzero(valid))
    accuracy = correct / total if total else float("nan")
    return correct, total, accuracy


