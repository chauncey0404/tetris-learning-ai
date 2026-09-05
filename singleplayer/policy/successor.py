from dataclasses import dataclass
import copy

import numpy as np

from singleplayer.game.executor import (
    execute_placement,
)

from singleplayer.network.state_encoder import (
    encode_state,
)


# ============================================================
# Reward V8 V1
# ============================================================

LINE_REWARD = {
    0: 0.0,
    1: 900.0,
    2: 2000.0,
    3: 3300.0,
    4: 6000.0,
}

REWARD_SCALE = 6000.0

GAME_OVER_PENALTY = 12000.0


# ============================================================
# Safe raw TetrisState clone
#
# IMPORTANT:
#
# tetris-gymnasium set_state() attaches state objects.
# Never reuse one mutable snapshot for multiple branches.
# ============================================================

def clone_raw_state(state):

    randomizer = copy.copy(
        state.randomizer
    )

    queue = state.queue.copy(
        randomizer
    )

    return type(state)(
        board=state.board.copy(),

        active_tetromino=copy.copy(
            state.active_tetromino
        ),

        x=state.x,
        y=state.y,

        queue=queue,

        holder=copy.copy(
            state.holder
        ),

        randomizer=randomizer,

        has_swapped=state.has_swapped,
        game_over=state.game_over,
        score=state.score,
    )


# ============================================================
# Successor candidate
# ============================================================

@dataclass
class SuccessorCandidate:

    teacher_rank: int
    reachable_rank: int

    teacher_score: float

    action: object

    lines_cleared: int

    raw_reward: float
    normalized_reward: float

    terminated: bool
    truncated: bool

    next_state_features: np.ndarray


# ============================================================
# Reward
# ============================================================

def calculate_reward(
    lines_cleared,
    terminated,
    truncated,
):

    reward = float(
        LINE_REWARD.get(
            int(lines_cleared),
            0.0,
        )
    )

    if (
        terminated
        or truncated
    ):

        reward -= (
            GAME_OVER_PENALTY
        )

    return (
        reward,
        reward / REWARD_SCALE,
    )


# ============================================================
# Preview top-K ACTUALLY reachable successors
#
# We intentionally do NOT use the old residual rollout code.
#
# Every candidate is:
#
#   restore root
#   execute candidate
#   verify simulator
#   record real next observable state
#
# Then restore root again.
# ============================================================

def preview_top_k_successors(
    *,
    adapter,
    teacher,
    state,
    top_k=4,
):

    ranked = teacher.rank(
        state
    )

    root_snapshot = (
        adapter.raw.get_state()
    )

    successors = []

    try:

        for teacher_index, decision in enumerate(
            ranked
        ):

            if len(successors) >= top_k:
                break

            # ================================================
            # Fresh root copy for every branch
            # ================================================

            adapter.raw.set_state(
                clone_raw_state(
                    root_snapshot
                )
            )

            candidate = (
                decision.candidate
            )

            try:

                result = execute_placement(
                    adapter,
                    candidate.action,
                )

            except Exception:

                # Geometrically valid but physically
                # unreachable candidate.
                continue

            next_state = (
                result[
                    "state"
                ]
            )

            # ================================================
            # Reachability / simulator identity check
            # ================================================

            if not np.array_equal(
                candidate.after_board,
                next_state.board,
            ):

                continue

            actual_lines = int(
                result[
                    "info"
                ].get(
                    "lines_cleared",
                    0,
                )
            )

            if (
                actual_lines
                !=
                int(
                    candidate.lines_cleared
                )
            ):

                continue

            terminated = bool(
                result[
                    "terminated"
                ]
            )

            truncated = bool(
                result[
                    "truncated"
                ]
            )

            (
                raw_reward,
                normalized_reward,
            ) = calculate_reward(
                lines_cleared=(
                    actual_lines
                ),

                terminated=(
                    terminated
                ),

                truncated=(
                    truncated
                ),
            )

            next_features = (
                encode_state(
                    next_state
                )
                .astype(
                    np.float32
                )
                .copy()
            )

            successors.append(
                SuccessorCandidate(
                    teacher_rank=(
                        teacher_index + 1
                    ),

                    reachable_rank=(
                        len(successors) + 1
                    ),

                    teacher_score=float(
                        decision.score
                    ),

                    action=(
                        candidate.action
                    ),

                    lines_cleared=(
                        actual_lines
                    ),

                    raw_reward=(
                        raw_reward
                    ),

                    normalized_reward=(
                        normalized_reward
                    ),

                    terminated=(
                        terminated
                    ),

                    truncated=(
                        truncated
                    ),

                    next_state_features=(
                        next_features
                    ),
                )
            )

    finally:

        # ====================================================
        # CRITICAL
        #
        # Return Gym to exactly the root branch.
        # ====================================================

        adapter.raw.set_state(
            clone_raw_state(
                root_snapshot
            )
        )

    return successors


# ============================================================
# Decode observable state feature vector for debugging
#
# Encoder layout:
#
# board   : 200
# current : 7
# hold    : 7
# next4   : 28
# can_hold: 1
# total   : 243
# ============================================================

def _decode_one_hot(values):

    values = np.asarray(
        values
    )

    if (
        values.size == 0
        or
        float(
            values.max()
        )
        <= 0.0
    ):

        return -1

    return int(
        np.argmax(
            values
        )
    )


def decode_observable_features(
    features,
):

    features = np.asarray(
        features
    )

    if features.shape != (243,):

        raise ValueError(
            f"Expected (243,), got "
            f"{features.shape}"
        )

    current = _decode_one_hot(
        features[
            200:207
        ]
    )

    hold = _decode_one_hot(
        features[
            207:214
        ]
    )

    next_block = (
        features[
            214:242
        ]
        .reshape(
            4,
            7,
        )
    )

    next4 = [
        _decode_one_hot(
            row
        )
        for row in next_block
    ]

    can_hold = bool(
        features[
            242
        ]
        >=
        0.5
    )

    return {
        "current":
            current,

        "hold":
            hold,

        "next4":
            next4,

        "can_hold":
            can_hold,
    }