from __future__ import annotations

import torch

from tetris_ai.learning.ranking import (
    pairwise_logistic_ranking_loss,
)


def main() -> None:
    print("=" * 88)
    print("V8.8.7 PAIRWISE RANKING AUXILIARY UNIT TEST")
    print("=" * 88)

    q = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    # canonical pair (0,1) target +1 => Q0 should become > Q1.
    targets = torch.tensor(
        [[1, 0, 0, 0, 0, 0]],
        dtype=torch.int8,
    )
    mask = torch.ones((1, 4), dtype=torch.bool)

    before, acc0, count0 = pairwise_logistic_ranking_loss(
        q,
        targets,
        candidate_mask=mask,
        temperature=0.10,
    )
    before.backward()

    assert q.grad is not None
    assert float(q.grad[0, 0]) < 0.0
    assert float(q.grad[0, 1]) > 0.0
    assert int(count0) == 1
    print("PASS gradient direction")

    with torch.no_grad():
        q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        good_loss, good_acc, good_count = (
            pairwise_logistic_ranking_loss(
                q2,
                targets,
                candidate_mask=mask,
                temperature=0.10,
            )
        )
        q3 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        bad_loss, bad_acc, bad_count = (
            pairwise_logistic_ranking_loss(
                q3,
                targets,
                candidate_mask=mask,
                temperature=0.10,
            )
        )

    assert float(good_loss) < float(bad_loss)
    assert float(good_acc) == 1.0
    assert float(bad_acc) == 0.0
    assert int(good_count) == int(bad_count) == 1
    print("PASS ordered pair loss/accuracy")

    targets_zero = torch.zeros((2, 6), dtype=torch.int8)
    q_zero = torch.zeros((2, 4), dtype=torch.float32, requires_grad=True)
    zero_loss, zero_acc, zero_count = pairwise_logistic_ranking_loss(
        q_zero,
        targets_zero,
        candidate_mask=torch.ones((2, 4), dtype=torch.bool),
    )
    assert float(zero_loss.detach()) == 0.0
    assert int(zero_count) == 0
    print("PASS tied/unsupervised pair mask")

    q_mask = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    # Pair (0,1) would be wrong, but candidate #2 is unreachable.
    unreachable = torch.tensor(
        [[True, False, True, True]],
        dtype=torch.bool,
    )
    _, _, masked_count = pairwise_logistic_ranking_loss(
        q_mask,
        targets,
        candidate_mask=unreachable,
    )
    assert int(masked_count) == 0
    print("PASS unreachable candidate mask")

    print()
    print("V8.8.7 PAIRWISE RANKING AUXILIARY UNIT TEST: PASS")


if __name__ == "__main__":
    main()
