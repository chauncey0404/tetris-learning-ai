from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_candidate_cross_entropy(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Listwise imitation loss over a padded candidate set.

    ``targets[b]`` is the local candidate index of the expert placement.
    Padded/unreachable entries are masked before cross entropy.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape [B,K]")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("targets must have shape [B]")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must match scores shape")
    if not 0.0 <= float(label_smoothing) < 1.0:
        raise ValueError("label_smoothing must be in [0,1)")

    mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
    target = targets.to(device=scores.device, dtype=torch.long)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every row must contain at least one valid candidate")
    if torch.any(target < 0) or torch.any(target >= scores.shape[1]):
        raise ValueError("target candidate index out of range")
    if not torch.all(mask.gather(1, target.unsqueeze(1)).squeeze(1)):
        raise ValueError("target points at a masked candidate")

    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    log_probs = F.log_softmax(masked_scores, dim=1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)

    smoothing = float(label_smoothing)
    if smoothing == 0.0:
        return nll.mean()

    # Smooth only across *valid* candidates.  PyTorch's built-in label
    # smoothing would allocate probability mass to padded/masked classes.
    valid_log_prob_sum = log_probs.masked_fill(~mask, 0.0).sum(dim=1)
    valid_count = mask.sum(dim=1).to(dtype=scores.dtype)
    smooth_loss = -valid_log_prob_sum / valid_count
    return ((1.0 - smoothing) * nll + smoothing * smooth_loss).mean()


def candidate_ranking_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return top-1/top-3/MRR metrics for padded candidate scores."""

    mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
    target = targets.to(device=scores.device, dtype=torch.long)
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    order = torch.argsort(masked, dim=1, descending=True)
    matches = order == target.unsqueeze(1)
    ranks = matches.to(torch.int64).argmax(dim=1) + 1

    return {
        "top1": (ranks == 1).to(scores.dtype).mean(),
        "top3": (ranks <= 3).to(scores.dtype).mean(),
        "mrr": (1.0 / ranks.to(scores.dtype)).mean(),
    }
