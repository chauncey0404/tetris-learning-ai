from types import SimpleNamespace
import numpy as np
from tetris_ai.model.q_network import STATE_SIZE
BOARD_CELLS = 200

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
