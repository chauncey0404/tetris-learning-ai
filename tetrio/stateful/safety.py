from __future__ import annotations

from dataclasses import dataclass

import torch

from tetrio.future.dominance import safe_dominance_mask


@dataclass(frozen=True)
class ExpertBaselineRelation:
    baseline_pred: torch.Tensor
    recall: torch.Tensor
    agreement: torch.Tensor
    safe_expert: torch.Tensor
    unsafe_expert: torch.Tensor
    ambiguous: torch.Tensor


def classify_expert_vs_baseline(
    *,
    baseline_scores: torch.Tensor,
    raw_features: torch.Tensor,
    inference_mask: torch.Tensor,
    expert_local: torch.Tensor,
    expert_in_shortlist: torch.Tensor,
) -> ExpertBaselineRelation:
    """Classify the expert target relative to frozen V1.1 Top-1.

    Categories are defined only on rows where the expert target is in the
    inference shortlist:

    agreement:
        expert == frozen V1.1 Top-1

    safe_expert:
        expert != baseline and expert safely dominates baseline under the
        existing conservative V1.1 structural-dominance contract

    unsafe_expert:
        baseline safely dominates the expert target

    ambiguous:
        disagreement, but neither candidate safely dominates the other

    The four categories form an exact partition of recall rows.
    """
    if baseline_scores.ndim != 2:
        raise ValueError("baseline_scores must be [B,K]")
    if raw_features.ndim != 3:
        raise ValueError("raw_features must be [B,K,F]")
    if inference_mask.shape != baseline_scores.shape:
        raise ValueError("inference_mask shape mismatch")

    masked = baseline_scores.masked_fill(
        ~inference_mask,
        torch.finfo(baseline_scores.dtype).min,
    )
    baseline_pred = masked.argmax(dim=1)
    recall = expert_in_shortlist.bool()

    rows = torch.arange(
        baseline_scores.shape[0],
        device=baseline_scores.device,
    )
    dom = safe_dominance_mask(raw_features, inference_mask)

    disagreement = recall & (expert_local != baseline_pred)
    expert_dom = dom[rows, expert_local, baseline_pred] & disagreement
    baseline_dom = dom[rows, baseline_pred, expert_local] & disagreement

    agreement = recall & (expert_local == baseline_pred)
    safe_expert = expert_dom & ~baseline_dom
    unsafe_expert = baseline_dom & ~expert_dom
    ambiguous = disagreement & ~safe_expert & ~unsafe_expert

    partition = agreement | safe_expert | unsafe_expert | ambiguous
    if not torch.equal(partition, recall):
        raise RuntimeError("expert/baseline relation does not partition recall rows")

    return ExpertBaselineRelation(
        baseline_pred=baseline_pred,
        recall=recall,
        agreement=agreement,
        safe_expert=safe_expert,
        unsafe_expert=unsafe_expert,
        ambiguous=ambiguous,
    )


def relation_row_weights(
    relation: ExpertBaselineRelation,
    *,
    agreement_weight: float,
    safe_weight: float,
    unsafe_weight: float,
    ambiguous_weight: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros_like(
        relation.baseline_pred,
        dtype=dtype,
    )
    out = torch.where(
        relation.agreement,
        torch.full_like(out, float(agreement_weight)),
        out,
    )
    out = torch.where(
        relation.safe_expert,
        torch.full_like(out, float(safe_weight)),
        out,
    )
    out = torch.where(
        relation.unsafe_expert,
        torch.full_like(out, float(unsafe_weight)),
        out,
    )
    out = torch.where(
        relation.ambiguous,
        torch.full_like(out, float(ambiguous_weight)),
        out,
    )
    return out


def unsafe_expert_preservation_loss(
    *,
    final_scores: torch.Tensor,
    expert_local: torch.Tensor,
    relation: ExpertBaselineRelation,
    margin: float,
) -> tuple[torch.Tensor, int]:
    """Keep the frozen baseline above an expert target it safely dominates."""
    mask = relation.unsafe_expert
    count = int(mask.sum().item())
    if count == 0:
        return final_scores.sum() * 0.0, 0

    rows = torch.arange(
        final_scores.shape[0],
        device=final_scores.device,
    )
    baseline_score = final_scores[rows, relation.baseline_pred]
    expert_score = final_scores[rows, expert_local]
    per = torch.relu(
        expert_score - baseline_score + float(margin)
    )
    return per[mask].mean(), count


def relation_counts(relation: ExpertBaselineRelation) -> dict[str, int]:
    recall = int(relation.recall.sum().item())
    agreement = int(relation.agreement.sum().item())
    safe = int(relation.safe_expert.sum().item())
    unsafe = int(relation.unsafe_expert.sum().item())
    ambiguous = int(relation.ambiguous.sum().item())
    return {
        "rows": int(relation.recall.shape[0]),
        "recall_rows": recall,
        "missing_expert_rows": int(relation.recall.shape[0]) - recall,
        "agreement_rows": agreement,
        "safe_expert_rows": safe,
        "unsafe_expert_rows": unsafe,
        "ambiguous_rows": ambiguous,
        "state_learnable_rows": safe + ambiguous,
    }


def change_diagnostics(
    *,
    baseline_pred: torch.Tensor,
    true_pred: torch.Tensor,
    expert_local: torch.Tensor,
    expert_in_shortlist: torch.Tensor,
    raw_features: torch.Tensor,
    inference_mask: torch.Tensor,
) -> dict[str, int]:
    """Classify state-induced Top-1 changes against frozen V1.1 Top-1."""
    if baseline_pred.shape != true_pred.shape:
        raise ValueError("prediction shape mismatch")

    changed = true_pred != baseline_pred
    dom = safe_dominance_mask(raw_features, inference_mask)
    rows = torch.arange(
        baseline_pred.shape[0],
        device=baseline_pred.device,
    )

    new_dom_old = dom[rows, true_pred, baseline_pred]
    old_dom_new = dom[rows, baseline_pred, true_pred]

    safe_change = changed & new_dom_old & ~old_dom_new
    unsafe_change = changed & old_dom_new & ~new_dom_old
    ambiguous_change = changed & ~safe_change & ~unsafe_change

    recall = expert_in_shortlist.bool()
    changed_to_expert = changed & recall & (true_pred == expert_local)
    changed_from_expert = (
        changed
        & recall
        & (baseline_pred == expert_local)
        & (true_pred != expert_local)
    )
    unsafe_changed_to_expert = changed_to_expert & unsafe_change

    return {
        "rows": int(changed.shape[0]),
        "changed_rows": int(changed.sum().item()),
        "safe_change_rows": int(safe_change.sum().item()),
        "unsafe_change_rows": int(unsafe_change.sum().item()),
        "ambiguous_change_rows": int(ambiguous_change.sum().item()),
        "changed_to_expert_rows": int(changed_to_expert.sum().item()),
        "changed_from_expert_rows": int(changed_from_expert.sum().item()),
        "unsafe_changed_to_expert_rows": int(
            unsafe_changed_to_expert.sum().item()
        ),
    }


def add_count_dict(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = int(total.get(key, 0)) + int(value)


def rates_from_counts(counts: dict[str, int]) -> dict[str, float]:
    rows = max(1, int(counts.get("rows", 0)))
    changed = max(1, int(counts.get("changed_rows", 0)))
    recall = max(1, int(counts.get("recall_rows", 0)))

    out: dict[str, float] = {}
    for key, value in counts.items():
        if key.endswith("_rows"):
            out[key.replace("_rows", "_rate")] = float(value) / rows

    if "changed_rows" in counts:
        for key in (
            "safe_change_rows",
            "unsafe_change_rows",
            "ambiguous_change_rows",
            "changed_to_expert_rows",
            "changed_from_expert_rows",
        ):
            if key in counts:
                out[key.replace("_rows", "_among_changed")] = (
                    float(counts[key]) / changed
                )

    for key in (
        "agreement_rows",
        "safe_expert_rows",
        "unsafe_expert_rows",
        "ambiguous_rows",
        "state_learnable_rows",
    ):
        if key in counts:
            out[key.replace("_rows", "_among_recall")] = (
                float(counts[key]) / recall
            )
    return out
