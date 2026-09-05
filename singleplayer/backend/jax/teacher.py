
from __future__ import annotations

"""
V8.8 vectorized JAX implementation of HeuristicTeacherV2.

This module preserves the production V8.4/V8.7 observable contract:
- current state: 243 floats
- candidate: board-after-action 200 + rotation4 + x10 + hold1 = 215
- only the top 4 actually reachable candidates are exposed to Q
- reachable rank is 1..4 and remains the scalar rank input used by the
  existing ObservableSafeQNetwork

The scalar Python HeuristicTeacherV2 remains the reference implementation.
Run test_v8_8_jax_teacher_parity.py before production training.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from singleplayer.backend.jax.vector_env import (
    HEIGHT,
    WIDTH,
    PADDING,
    N_CANDIDATES,
    V88State,
    all_candidates_batch,
    all_candidates_one,
    candidate_table,
    encode_observable_243,
    encode_observable_243_batch,
    observable_candidate_215,
    playable_board,
)


TOP_K = 4
REWARD_SCALE = 6000.0
GAME_OVER_PENALTY = 12000.0

# Piece ids: I=0, O=1, T=2, S=3, Z=4, J=5, L=6
I_PIECE = 0

# Canonical unique rotations from tetris_placement.get_rotations().
# I/S/Z -> 0,1 ; O -> 0 ; T/J/L -> 0,1,2,3
UNIQUE_ROTATION_MASK = jnp.asarray(
    [
        [1, 1, 0, 0],  # I
        [1, 0, 0, 0],  # O
        [1, 1, 1, 1],  # T
        [1, 1, 0, 0],  # S
        [1, 1, 0, 0],  # Z
        [1, 1, 1, 1],  # J
        [1, 1, 1, 1],  # L
    ],
    dtype=jnp.bool_,
)

LINE_CLEAR_REWARD = jnp.asarray(
    [0.0, 900.0, 2000.0, 3300.0, 6000.0],
    dtype=jnp.float32,
)


# Bit-exact float32 values of the REFERENCE Python calculation:
#
#   np.float32((float(line_reward) - penalty) / 6000.0)
#
# Do NOT recompute these with float32 division inside XLA. XLA may implement
# the division as a float32 reciprocal/multiply and land one ULP away from
# Python-double-then-float32, e.g. 900/6000:
#
#   Python reference -> 0.15000000596046448
#   direct JAX f32   -> 0.14999999105930328
#
# These tables preserve the production replay reward bits exactly.
NORMALIZED_REWARD_NO_GO = jnp.asarray(
    [
        0.0,
        0.15000000596046448,
        0.3333333432674408,
        0.550000011920929,
        1.0,
    ],
    dtype=jnp.float32,
)

NORMALIZED_REWARD_GAME_OVER = jnp.asarray(
    [
        -2.0,
        -1.850000023841858,
        -1.6666666269302368,
        -1.4500000476837158,
        -1.0,
    ],
    dtype=jnp.float32,
)

C_ROTATION, C_X, C_USE_HOLD = candidate_table()
C_ROTATION = C_ROTATION.astype(jnp.int32)
C_X = C_X.astype(jnp.int32)
C_USE_HOLD = C_USE_HOLD.astype(jnp.bool_)


class TopKBundle(NamedTuple):
    # Current observable state.
    state_features: jax.Array          # [B,243] or [243]

    # Teacher-ranked, reachable top-K candidate interface consumed by Q.
    candidate_features: jax.Array      # [B,K,215] or [K,215]
    rewards: jax.Array                 # [B,K] or [K]
    teacher_scores: jax.Array          # [B,K] or [K]
    teacher_ranks: jax.Array           # [B,K] or [K], reachable ranks 1..K
    mask: jax.Array                    # [B,K] or [K]

    # Mapping back to the fixed 80-slot candidate table.
    candidate_indices: jax.Array       # [B,K] or [K]

    # Full next states for the selected K candidates, kept on the JAX side.
    candidate_states: V88State         # leaves [B,K,...] or [K,...]

    # Diagnostics.
    lines: jax.Array                   # [B,K] or [K]
    game_over: jax.Array               # [B,K] or [K]


def _played_piece_ids_one(state: V88State) -> jax.Array:
    """Piece actually placed for each of the fixed 80 candidate slots."""
    hold_piece = jnp.where(
        state.held_piece >= 0,
        state.held_piece,
        state.queue[0],
    )
    return jnp.where(C_USE_HOLD, hold_piece, state.active).astype(jnp.int32)


def _unique_rotation_candidates_one(state: V88State) -> jax.Array:
    played = _played_piece_ids_one(state)
    return UNIQUE_ROTATION_MASK[played, C_ROTATION]


def _board_features(boards: jax.Array):
    """
    Vectorized equivalent of teacher.extract_board_features_v2().

    boards: [...,20,10], bool/int occupied.
    """
    occupied = boards > 0

    any_occ = jnp.any(occupied, axis=-2)  # [...,10]
    first_occ = jnp.argmax(occupied, axis=-2)  # returns 0 when all False
    heights = jnp.where(
        any_occ,
        HEIGHT - first_occ,
        0,
    ).astype(jnp.int32)

    # A hole is an empty cell at/under the first occupied cell in its column.
    seen_occupied = jnp.cumsum(
        occupied.astype(jnp.int32),
        axis=-2,
    ) > 0
    holes = jnp.sum(
        (~occupied) & seen_occupied,
        axis=(-2, -1),
    ).astype(jnp.int32)

    aggregate_height = jnp.sum(heights, axis=-1).astype(jnp.int32)
    max_height = jnp.max(heights, axis=-1).astype(jnp.int32)
    bumpiness = jnp.sum(
        jnp.abs(heights[..., :-1] - heights[..., 1:]),
        axis=-1,
    ).astype(jnp.int32)

    left_neighbor = jnp.concatenate(
        [heights[..., :1], heights[..., :-1]],
        axis=-1,
    )
    right_neighbor = jnp.concatenate(
        [heights[..., 1:], heights[..., -1:]],
        axis=-1,
    )

    interior_wall = jnp.minimum(left_neighbor, right_neighbor)

    # Edges compare against their only neighbor.
    wall_height = interior_wall
    wall_height = wall_height.at[..., 0].set(heights[..., 1])
    wall_height = wall_height.at[..., -1].set(heights[..., -2])

    wells = jnp.maximum(0, wall_height - heights).astype(jnp.int32)

    total_well_depth = jnp.sum(wells, axis=-1).astype(jnp.int32)
    max_well_depth = jnp.max(wells, axis=-1).astype(jnp.int32)
    deep_wells = jnp.sum(wells >= 4, axis=-1).astype(jnp.int32)
    left_well_depth = wells[..., 0]
    right_well_depth = wells[..., -1]

    center_max = jnp.max(heights[..., 1:-1], axis=-1)
    edge_support = jnp.maximum(heights[..., 0], heights[..., -1])
    center_tower_excess = jnp.maximum(
        0,
        center_max - edge_support,
    ).astype(jnp.int32)

    return {
        "heights": heights,
        "holes": holes,
        "aggregate_height": aggregate_height,
        "max_height": max_height,
        "bumpiness": bumpiness,
        "wells": wells,
        "total_well_depth": total_well_depth,
        "max_well_depth": max_well_depth,
        "deep_wells": deep_wells,
        "left_well_depth": left_well_depth,
        "right_well_depth": right_well_depth,
        "center_tower_excess": center_tower_excess,
    }


def _visible_i_supply_one(state: V88State) -> jax.Array:
    """
    Vectorized equivalent of known_future_pieces()+visible I supply for 80 actions.
    """
    queue_i_all = jnp.sum(state.queue == I_PIECE).astype(jnp.int32)
    queue_i_tail = jnp.sum(state.queue[1:] == I_PIECE).astype(jnp.int32)

    no_hold_supply = (
        queue_i_all
        + (state.held_piece == I_PIECE).astype(jnp.int32)
    )

    hold_nonempty_supply = (
        queue_i_all
        + (state.active == I_PIECE).astype(jnp.int32)
    )

    hold_empty_supply = (
        queue_i_tail
        + (state.active == I_PIECE).astype(jnp.int32)
    )

    hold_supply = jnp.where(
        state.held_piece >= 0,
        hold_nonempty_supply,
        hold_empty_supply,
    )

    return jnp.where(
        C_USE_HOLD,
        hold_supply,
        no_hold_supply,
    ).astype(jnp.int32)


def _teacher_scores_one(
    state: V88State,
    candidate_states: V88State,
    lines: jax.Array,
    reachable: jax.Array,
):
    """
    Score all 80 slots with HeuristicTeacherV2 semantics.

    We sort only physically reachable candidates. This is equivalent to the
    old pipeline's "sort all geometric candidates then skip unreachable ones"
    for the resulting reachable order. The network uses reachable_rank, not
    the discarded geometric teacher_rank.
    """
    boards = (
        candidate_states.board[:, :HEIGHT, PADDING:PADDING + WIDTH] > 0
    ).astype(jnp.int8)

    f = _board_features(boards)

    unique_rotation = _unique_rotation_candidates_one(state)

    # A collision at the initial hard-drop position is Gym's pre-lock top-out
    # branch: game_over=True with no board commit. Python Teacher removes
    # canonical top_out candidates before reachability checking.
    current_board = (playable_board(state) > 0).astype(jnp.int8)
    board_changed = jnp.any(
        boards != current_board[None, ...],
        axis=(-2, -1),
    )
    prelock_topout = candidate_states.game_over & (~board_changed)

    usable = (
        reachable
        & unique_rotation
        & (~prelock_topout)
    )

    wells = f["wells"]
    i_debt = jnp.sum(
        jnp.where(wells >= 4, wells // 4, 0),
        axis=-1,
    ).astype(jnp.int32)

    visible_i_supply = _visible_i_supply_one(state)
    uncovered_i_debt = jnp.maximum(
        0,
        i_debt - visible_i_supply,
    ).astype(jnp.int32)

    left = f["left_well_depth"]
    right = f["right_well_depth"]

    exactly_one_edge_well = (left > 0) ^ (right > 0)
    edge_depth = jnp.maximum(left, right)

    # Other deep wells exclude whichever edge is the active clean edge.
    positions = jnp.arange(WIDTH)
    active_edge_pos = jnp.where(left > 0, 0, WIDTH - 1)
    other_deep = jnp.any(
        (wells >= 4)
        & (positions[None, :] != active_edge_pos[:, None]),
        axis=-1,
    )

    clean_edge = (
        exactly_one_edge_well
        & (f["holes"] == 0)
        & (~other_deep)
        & (edge_depth <= 7)
    )

    lines_i = jnp.clip(lines.astype(jnp.int32), 0, 4)
    score = LINE_CLEAR_REWARD[lines_i]

    score = score - f["holes"].astype(jnp.float32) * 250.0
    score = score - f["aggregate_height"].astype(jnp.float32) * 4.0
    score = score - f["bumpiness"].astype(jnp.float32) * 3.0
    score = score - f["max_height"].astype(jnp.float32) * 8.0

    danger = jnp.maximum(0, f["max_height"] - 10)
    score = score - (
        (danger * danger).astype(jnp.float32) * 30.0
    )

    emergency = jnp.maximum(0, f["max_height"] - 15)
    score = score - (
        (f["max_height"] >= 16).astype(jnp.float32)
        * emergency.astype(jnp.float32)
        * 500.0
    )

    center_excess = jnp.maximum(0, f["center_tower_excess"] - 3)
    score = score - (
        (f["center_tower_excess"] > 3).astype(jnp.float32)
        * (center_excess * center_excess).astype(jnp.float32)
        * 160.0
    )

    score = score - (
        (f["deep_wells"] >= 2).astype(jnp.float32)
        * (f["deep_wells"] - 1).astype(jnp.float32)
        * 700.0
    )

    score = score - (
        ((left >= 4) & (right >= 4)).astype(jnp.float32)
        * 1200.0
    )

    excess_depth = jnp.maximum(0, f["max_well_depth"] - 7)
    score = score - (
        (f["max_well_depth"] > 7).astype(jnp.float32)
        * (excess_depth * excess_depth).astype(jnp.float32)
        * 90.0
    )

    has_uncovered = uncovered_i_debt > 0

    normal_i_penalty = (
        uncovered_i_debt.astype(jnp.float32) * 500.0
    )
    clean_i_penalty = (
        uncovered_i_debt.astype(jnp.float32) * 120.0
    )

    extra_i = jnp.maximum(0, uncovered_i_debt - 1)
    nonlinear_i = (
        (uncovered_i_debt >= 2).astype(jnp.float32)
        * (extra_i * extra_i).astype(jnp.float32)
        * 350.0
    )

    score = score - jnp.where(
        has_uncovered,
        jnp.where(
            clean_edge,
            clean_i_penalty,
            normal_i_penalty + nonlinear_i,
        ),
        0.0,
    )

    height_risk = jnp.maximum(0, f["max_height"] - 12)
    score = score - (
        ((i_debt > 0) & (f["max_height"] > 12)).astype(jnp.float32)
        * i_debt.astype(jnp.float32)
        * height_risk.astype(jnp.float32)
        * 70.0
    )

    # Tetris-well potential V2.1.
    well_bonus = jnp.where(
        edge_depth == 1,
        120.0,
        jnp.where(
            edge_depth == 2,
            300.0,
            jnp.where(
                edge_depth == 3,
                600.0,
                jnp.where(
                    (edge_depth >= 4) & (edge_depth <= 7),
                    1100.0,
                    0.0,
                ),
            ),
        ),
    )
    supply_bonus = jnp.where(
        visible_i_supply >= 1,
        450.0,
        0.0,
    )
    score = score + clean_edge.astype(jnp.float32) * (
        well_bonus + supply_bonus
    )

    # Invalid candidates are never ranked. Keep a large finite sentinel to
    # avoid NaNs/infs in sorting.
    score_for_sort = jnp.where(
        usable,
        score,
        jnp.float32(-1.0e20),
    )

    return score, score_for_sort, usable


def _gather_state_k(candidate_states: V88State, indices: jax.Array) -> V88State:
    return jax.tree_util.tree_map(
        lambda x: x[indices],
        candidate_states,
    )


def _topk_one(state: V88State) -> TopKBundle:
    candidate_states, lines, reachable = all_candidates_one(state)

    score, score_for_sort, usable = _teacher_scores_one(
        state,
        candidate_states,
        lines,
        reachable,
    )

    indices = jnp.arange(N_CANDIDATES, dtype=jnp.int32)

    # Exact Python Teacher tie-break order (descending):
    #   score,
    #   lines_cleared,
    #   -use_hold,
    #   -rotation,
    #   -x
    #
    # lax.sort is ascending, therefore keys are:
    #   -score, -lines, use_hold, rotation, x
    sorted_values = jax.lax.sort(
        (
            -score_for_sort,
            -lines.astype(jnp.int32),
            C_USE_HOLD.astype(jnp.int32),
            C_ROTATION.astype(jnp.int32),
            C_X.astype(jnp.int32),
            indices,
        ),
        dimension=0,
        is_stable=True,
        num_keys=5,
    )
    sorted_indices = sorted_values[-1]
    top_indices = sorted_indices[:TOP_K]

    top_mask = usable[top_indices]
    top_states = _gather_state_k(candidate_states, top_indices)

    top_lines = lines[top_indices].astype(jnp.int32)
    top_scores = score[top_indices].astype(jnp.float32)
    top_game_over = top_states.game_over.astype(jnp.bool_)

    # Candidate representation is observable-safe by construction:
    # board after action + action metadata only.
    top_features = jax.vmap(observable_candidate_215)(
        top_states,
        C_ROTATION[top_indices],
        C_X[top_indices],
        C_USE_HOLD[top_indices],
    )

    reward_index = jnp.clip(top_lines, 0, 4).astype(jnp.int32)
    rewards = jnp.where(
        top_game_over,
        NORMALIZED_REWARD_GAME_OVER[reward_index],
        NORMALIZED_REWARD_NO_GO[reward_index],
    ).astype(jnp.float32)

    reachable_ranks = jnp.arange(
        1,
        TOP_K + 1,
        dtype=jnp.float32,
    )

    # Zero out padded/invalid tail slots. Candidate 0 should always be valid
    # for a nonterminal state that has any legal move, but masking keeps the
    # interface robust near terminal/no-successor states.
    top_features = jnp.where(
        top_mask[:, None],
        top_features,
        jnp.zeros_like(top_features),
    )
    rewards = jnp.where(top_mask, rewards, 0.0)
    top_scores = jnp.where(top_mask, top_scores, 0.0)
    ranks = jnp.where(top_mask, reachable_ranks, 0.0)
    top_lines = jnp.where(top_mask, top_lines, 0)
    top_game_over = jnp.where(top_mask, top_game_over, False)

    return TopKBundle(
        state_features=encode_observable_243(state),
        candidate_features=top_features,
        rewards=rewards,
        teacher_scores=top_scores,
        teacher_ranks=ranks,
        mask=top_mask,
        candidate_indices=top_indices,
        candidate_states=top_states,
        lines=top_lines,
        game_over=top_game_over,
    )


topk_one = jax.jit(_topk_one)
topk_batch = jax.jit(jax.vmap(_topk_one))


def select_candidate_state(
    candidate_states: V88State,
    chosen_index: jax.Array,
) -> V88State:
    """
    Select one of K candidate states for every environment.

    candidate_states leaves: [B,K,...]
    chosen_index: [B]
    """
    chosen_index = chosen_index.astype(jnp.int32)

    def gather_leaf(x):
        rows = jnp.arange(x.shape[0], dtype=jnp.int32)
        return x[rows, chosen_index]

    return jax.tree_util.tree_map(gather_leaf, candidate_states)


select_candidate_state_jit = jax.jit(select_candidate_state)


def replace_done_or_segment_states(
    states: V88State,
    reset_states: V88State,
    reset_mask: jax.Array,
) -> V88State:
    """Select reset state per environment without changing static batch shape."""
    reset_mask = reset_mask.astype(jnp.bool_)

    def choose(new, old):
        shape = (reset_mask.shape[0],) + (1,) * (old.ndim - 1)
        mask = reset_mask.reshape(shape)
        return jnp.where(mask, new, old)

    return jax.tree_util.tree_map(choose, reset_states, states)


replace_done_or_segment_states_jit = jax.jit(replace_done_or_segment_states)
