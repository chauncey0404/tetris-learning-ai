from types import SimpleNamespace

import numpy as np
import torch

from ai.observable_q_network import CANDIDATE_SIZE, STATE_SIZE


TOP_K = 4
BOARD_CELLS = 200
ROTATIONS = 4
BOARD_WIDTH = 10

# 200 board + 4 rotation one-hot + 10 x one-hot + 1 use_hold
EXPECTED_CANDIDATE_SIZE = BOARD_CELLS + ROTATIONS + BOARD_WIDTH + 1
if EXPECTED_CANDIDATE_SIZE != CANDIDATE_SIZE:
    raise RuntimeError(
        f"Candidate size mismatch: {EXPECTED_CANDIDATE_SIZE} != {CANDIDATE_SIZE}"
    )


def observable_candidate_features(successor):
    """
    Build a legal pre-action candidate representation.

    The ONLY information taken from the preview successor state is the first
    200 board cells. Queue/current/hold/can_hold fields at indices 200:243 are
    intentionally ignored because previewing them may reveal future pieces.
    """
    preview_state = np.asarray(
        successor.next_state_features,
        dtype=np.float32,
    )
    if preview_state.shape != (STATE_SIZE,):
        raise ValueError(
            f"next_state_features must have shape ({STATE_SIZE},); "
            f"got {preview_state.shape}"
        )

    action = successor.action
    rotation = int(action.rotation)
    x = int(action.x)
    use_hold = bool(action.use_hold)

    if not 0 <= rotation < ROTATIONS:
        raise ValueError(f"rotation out of range: {rotation}")
    if not 0 <= x < BOARD_WIDTH:
        raise ValueError(f"x out of range for 10-wide board: {x}")

    features = np.zeros(CANDIDATE_SIZE, dtype=np.float32)

    # Safe deterministic board-after-action only.
    features[:BOARD_CELLS] = preview_state[:BOARD_CELLS]

    offset = BOARD_CELLS
    features[offset + rotation] = 1.0

    offset += ROTATIONS
    features[offset + x] = 1.0

    offset += BOARD_WIDTH
    features[offset] = 1.0 if use_hold else 0.0

    return features


def candidate_arrays(successors, top_k=TOP_K):
    features = np.zeros((top_k, CANDIDATE_SIZE), dtype=np.float32)
    rewards = np.zeros(top_k, dtype=np.float32)
    teacher_scores = np.zeros(top_k, dtype=np.float32)
    teacher_ranks = np.zeros(top_k, dtype=np.float32)
    mask = np.zeros(top_k, dtype=np.bool_)

    for index, successor in enumerate(successors[:top_k]):
        features[index] = observable_candidate_features(successor)
        rewards[index] = float(successor.normalized_reward)
        teacher_scores[index] = float(successor.teacher_score)
        teacher_ranks[index] = float(successor.reachable_rank)
        mask[index] = True

    return features, rewards, teacher_scores, teacher_ranks, mask


def compact_candidate_arrays(successors):
    if not successors:
        return (
            np.zeros((0, CANDIDATE_SIZE), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    features = np.stack(
        [observable_candidate_features(item) for item in successors]
    ).astype(np.float32)
    rewards = np.asarray(
        [item.normalized_reward for item in successors],
        dtype=np.float32,
    )
    teacher_scores = np.asarray(
        [item.teacher_score for item in successors],
        dtype=np.float32,
    )
    teacher_ranks = np.asarray(
        [item.reachable_rank for item in successors],
        dtype=np.float32,
    )
    return features, rewards, teacher_scores, teacher_ranks


def conservative_choice(q_values, gate):
    q_values = np.asarray(q_values, dtype=np.float32)
    if q_values.size <= 1:
        return 0, 0.0

    teacher_q = float(q_values[0])
    best_alt_index = int(np.argmax(q_values[1:])) + 1
    gap = float(q_values[best_alt_index]) - teacher_q

    if gap >= float(gate):
        return best_alt_index, gap
    return 0, gap


@torch.no_grad()
def q_values_for_successors(
    *,
    model,
    state_features,
    successors,
    device,
):
    candidates, rewards, scores, ranks = compact_candidate_arrays(successors)

    state_tensor = torch.from_numpy(
        np.asarray(state_features, dtype=np.float32)
    ).to(device).unsqueeze(0)

    candidate_tensor = torch.from_numpy(candidates).to(device).unsqueeze(0)
    reward_tensor = torch.from_numpy(rewards).to(device).unsqueeze(0)
    score_tensor = torch.from_numpy(scores).to(device).unsqueeze(0)
    rank_tensor = torch.from_numpy(ranks).to(device).unsqueeze(0)

    q = model(
        state=state_tensor,
        candidates=candidate_tensor,
        rewards=reward_tensor,
        teacher_scores=score_tensor,
        teacher_ranks=rank_tensor,
    )[0]

    return q.detach().cpu().numpy()


def poison_preview_nonboard(successor, rng):
    """
    Test helper: return a successor-like object whose preview state indices
    200:243 are replaced with random garbage while all legal candidate inputs
    stay the same.
    """
    poisoned = np.asarray(successor.next_state_features, dtype=np.float32).copy()
    poisoned[BOARD_CELLS:] = rng.normal(
        loc=123.0,
        scale=77.0,
        size=STATE_SIZE - BOARD_CELLS,
    ).astype(np.float32)

    return SimpleNamespace(
        next_state_features=poisoned,
        action=successor.action,
        normalized_reward=successor.normalized_reward,
        teacher_score=successor.teacher_score,
        reachable_rank=successor.reachable_rank,
        lines_cleared=getattr(successor, "lines_cleared", 0),
        terminated=getattr(successor, "terminated", False),
        truncated=getattr(successor, "truncated", False),
    )
