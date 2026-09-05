from __future__ import annotations

"""
V8.8 JAX vectorized Tetris rollout backend.

Design goals
------------
- Match the project's validated Gym execution semantics:
  * 20x10 canonical board, padding=4.
  * no SRS / no wall kicks.
  * PlacementAction(rotation, x, use_hold)
      rotation: canonical target orientation 0..3
      x       : canonical left-most occupied board column
      use_hold: execute Hold before rotation/translation.
  * Hold preserves piece orientation.
  * Hold may be used at most once before a lock.
  * visible next queue contains 4 pieces.
  * 7-bag randomizer.
- JAX/JIT/VMAP all placement simulations.
- Fixed 80 candidate slots per state:
      2 hold modes x 4 rotations x 10 x positions.
  Invalid/unreachable placements are represented by a boolean mask rather
  than Python-side variable-length lists.

This module intentionally does NOT replace HeuristicTeacherV2 yet. It
accelerates the expensive board/action simulation layer first. The parity
test must pass before wiring this backend into the production trainer.
"""

from typing import NamedTuple, Tuple
import copy as pycopy

import jax
import jax.numpy as jnp
import numpy as np

from tetris_ai.core.tetrominoes import PIECE_NAMES as CORE_PIECE_NAMES, padded_rotation_tensor


WIDTH = 10
HEIGHT = 20
PADDING = 4
PADDED_WIDTH = WIDTH + 2 * PADDING
PADDED_HEIGHT = HEIGHT + PADDING
QUEUE_SIZE = 4
BAG_SIZE = 7
N_ROTATIONS = 4
N_CANDIDATES = 2 * N_ROTATIONS * WIDTH

# Piece order and canonical rotations come from the one shared geometry source.
PIECE_NAMES = CORE_PIECE_NAMES
PIECE_MATRIX_SIZE = jnp.asarray([4, 2, 3, 3, 3, 3, 3], dtype=jnp.int16)
PIECE_MATRICES = jnp.asarray(padded_rotation_tensor(dtype=np.int8), dtype=jnp.int8)


class V88State(NamedTuple):
    rng_key: jax.Array
    board: jax.Array              # [24, 18], 0 empty, 1 bedrock, 2..8 pieces
    active: jax.Array             # scalar 0..6
    rotation: jax.Array           # scalar 0..3
    x: jax.Array                  # raw padded-board x
    y: jax.Array                  # raw padded-board y
    queue: jax.Array              # [4], piece ids 0..6
    bag: jax.Array                # [7], current 7-bag
    bag_index: jax.Array          # scalar index into bag
    next_bag: jax.Array           # [7], already-shuffled future bag
    held_piece: jax.Array         # scalar -1 empty else 0..6
    held_rotation: jax.Array      # scalar 0..3
    can_hold: jax.Array           # bool
    game_over: jax.Array          # bool
    score: jax.Array              # float32; not used by current learner


def create_board() -> jax.Array:
    playable = jnp.zeros((HEIGHT, WIDTH), dtype=jnp.int8)
    return jnp.pad(
        playable,
        ((0, PADDING), (PADDING, PADDING)),
        mode="constant",
        constant_values=1,
    )


def _spawn_x(piece: jax.Array) -> jax.Array:
    # Exact classic Tetris-Gymnasium reset_tetromino_position formula:
    # width_padded // 2 - active.matrix.shape[0] // 2
    return (
        PADDED_WIDTH // 2
        - PIECE_MATRIX_SIZE[piece] // 2
    ).astype(jnp.int16)


def _matrix(piece: jax.Array, rotation: jax.Array) -> jax.Array:
    return PIECE_MATRICES[piece, rotation]


def _left_occupied_offset(piece: jax.Array, rotation: jax.Array) -> jax.Array:
    matrix = _matrix(piece, rotation)
    occupied_cols = jnp.any(matrix > 0, axis=0)
    idx = jnp.where(occupied_cols, jnp.arange(4), 4)
    return jnp.min(idx).astype(jnp.int16)


def canonical_x(state: V88State) -> jax.Array:
    return (
        state.x
        - PADDING
        + _left_occupied_offset(state.active, state.rotation)
    ).astype(jnp.int16)


def _target_raw_x(piece: jax.Array, rotation: jax.Array, canonical_target_x: jax.Array) -> jax.Array:
    return (
        PADDING
        + canonical_target_x
        - _left_occupied_offset(piece, rotation)
    ).astype(jnp.int16)


def collision(board: jax.Array, piece: jax.Array, rotation: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    matrix = _matrix(piece, rotation)
    section = jax.lax.dynamic_slice(
        board,
        (y.astype(jnp.int32), x.astype(jnp.int32)),
        (4, 4),
    )
    return jnp.any((section > 0) & (matrix > 0))


def _project(board: jax.Array, piece: jax.Array, rotation: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    matrix = _matrix(piece, rotation) * (piece.astype(jnp.int8) + jnp.int8(2))
    update = jax.lax.dynamic_update_slice(
        jnp.zeros_like(board),
        matrix,
        (y.astype(jnp.int32), x.astype(jnp.int32)),
    )
    return board + update


def _clear_rows(board: jax.Array) -> Tuple[jax.Array, jax.Array]:
    playable = board[:HEIGHT, PADDING:PADDING + WIDTH]
    full = jnp.all(playable > 0, axis=1)
    n = jnp.sum(full).astype(jnp.int32)

    # Keep non-full rows in their original order; place synthetic zero rows
    # at the top. This avoids dynamic-shape indexing under JIT.
    indices = jnp.where(full, HEIGHT, jnp.arange(HEIGHT))
    indices = jnp.sort(indices)
    compacted = jnp.take(
        playable,
        indices,
        axis=0,
        mode="fill",
        fill_value=0,
    )
    compacted = jnp.roll(compacted, shift=n, axis=0)

    padded = jnp.pad(
        compacted.astype(jnp.int8),
        ((0, PADDING), (PADDING, PADDING)),
        mode="constant",
        constant_values=1,
    )
    return padded, n


def _shuffle_bag(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    key, sub = jax.random.split(key)
    bag = jax.random.permutation(sub, jnp.arange(BAG_SIZE, dtype=jnp.int8))
    return key, bag


def _draw_piece(
    key: jax.Array,
    bag: jax.Array,
    bag_index: jax.Array,
    next_bag: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Draw one piece while carrying one already-shuffled future bag.

    Why V8.8 keeps next_bag:
    - Gym's BagRandomizer reshuffles IMMEDIATELY after the final piece of a bag
      is drawn.
    - One PlacementAction(use_hold=True) with an empty holder can consume TWO
      queue pieces in a single simulated transition (one for Hold, one after
      lock).
    - A one-step parity check may therefore cross a 7-bag boundary.
    - JAX PRNG and NumPy PCG64 intentionally use different RNG algorithms, so
      generating the new bag inside JAX would make only the newly appended
      queue tail differ even though board semantics are correct.

    Carrying next_bag lets the Python Gym bridge prefetch the exact future Gym
    bag without mutating the real environment. Production JAX rollouts still
    remain fully JIT-able: when next_bag is promoted, JAX generates another
    future bag in the background.
    """
    piece = bag[bag_index]
    next_index = bag_index + jnp.int16(1)

    def promote_future(_):
        new_key, generated_future = _shuffle_bag(key)
        return (
            new_key,
            next_bag,
            jnp.int16(0),
            generated_future,
        )

    def keep_current(_):
        return (
            key,
            bag,
            next_index.astype(jnp.int16),
            next_bag,
        )

    new_key, new_bag, new_index, new_next_bag = jax.lax.cond(
        next_index >= BAG_SIZE,
        promote_future,
        keep_current,
        operand=None,
    )

    return (
        piece.astype(jnp.int8),
        new_key,
        new_bag.astype(jnp.int8),
        new_index,
        new_next_bag.astype(jnp.int8),
    )


def _pop_visible_queue(state: V88State) -> Tuple[jax.Array, V88State]:
    next_active = state.queue[0]
    appended, key, bag, bag_index, next_bag = _draw_piece(
        state.rng_key,
        state.bag,
        state.bag_index,
        state.next_bag,
    )
    queue = jnp.concatenate([state.queue[1:], appended[None]], axis=0)
    return next_active, state._replace(
        rng_key=key,
        queue=queue.astype(jnp.int8),
        bag=bag.astype(jnp.int8),
        bag_index=bag_index.astype(jnp.int16),
        next_bag=next_bag.astype(jnp.int8),
    )


def reset_one(key: jax.Array) -> V88State:
    key, bag = _shuffle_bag(key)
    key, next_bag = _shuffle_bag(key)
    bag_index = jnp.int16(0)

    def draw(carry, _):
        key_i, bag_i, index_i, next_bag_i = carry
        piece, key_i, bag_i, index_i, next_bag_i = _draw_piece(
            key_i,
            bag_i,
            index_i,
            next_bag_i,
        )
        return (key_i, bag_i, index_i, next_bag_i), piece

    (key, bag, bag_index, next_bag), sequence = jax.lax.scan(
        draw,
        (key, bag, bag_index, next_bag),
        xs=None,
        length=5,
    )

    active = sequence[0]
    queue = sequence[1:5]

    return V88State(
        rng_key=key,
        board=create_board(),
        active=active.astype(jnp.int8),
        rotation=jnp.int8(0),
        x=_spawn_x(active),
        y=jnp.int16(0),
        queue=queue.astype(jnp.int8),
        bag=bag.astype(jnp.int8),
        bag_index=bag_index.astype(jnp.int16),
        next_bag=next_bag.astype(jnp.int8),
        held_piece=jnp.int8(-1),
        held_rotation=jnp.int8(0),
        can_hold=jnp.bool_(True),
        game_over=jnp.bool_(False),
        score=jnp.float32(0.0),
    )


reset_one_jit = jax.jit(reset_one)
reset_batch = jax.jit(jax.vmap(reset_one))


def _hold(state: V88State) -> Tuple[V88State, jax.Array]:
    """Execute classic Hold. Returns (state, hold_was_allowed)."""
    allowed = state.can_hold & (~state.game_over)

    def do_hold(s: V88State) -> V88State:
        old_active = s.active
        old_rotation = s.rotation

        def empty_holder(ss: V88State) -> V88State:
            new_active, ss = _pop_visible_queue(ss)
            return ss._replace(
                active=new_active.astype(jnp.int8),
                rotation=jnp.int8(0),
                x=_spawn_x(new_active),
                y=jnp.int16(0),
                held_piece=old_active.astype(jnp.int8),
                held_rotation=old_rotation.astype(jnp.int8),
                can_hold=jnp.bool_(False),
            )

        def occupied_holder(ss: V88State) -> V88State:
            new_active = ss.held_piece
            new_rotation = ss.held_rotation
            return ss._replace(
                active=new_active.astype(jnp.int8),
                rotation=new_rotation.astype(jnp.int8),
                x=_spawn_x(new_active),
                y=jnp.int16(0),
                held_piece=old_active.astype(jnp.int8),
                held_rotation=old_rotation.astype(jnp.int8),
                can_hold=jnp.bool_(False),
            )

        return jax.lax.cond(
            s.held_piece < 0,
            empty_holder,
            occupied_holder,
            s,
        )

    new_state = jax.lax.cond(allowed, do_hold, lambda s: s, state)
    return new_state, allowed


def _rotate_to(state: V88State, target_rotation: jax.Array) -> Tuple[V88State, jax.Array]:
    target_rotation = target_rotation.astype(jnp.int8) % 4
    requested = ((target_rotation - state.rotation) % 4).astype(jnp.int8)

    def body(i, s):
        should_rotate = i < requested
        next_rotation = ((s.rotation + 1) % 4).astype(jnp.int8)
        blocked = collision(s.board, s.active, next_rotation, s.x, s.y)
        apply = should_rotate & (~blocked)
        return s._replace(
            rotation=jnp.where(apply, next_rotation, s.rotation).astype(jnp.int8)
        )

    state = jax.lax.fori_loop(0, 3, body, state)
    reached = state.rotation == target_rotation
    return state, reached


def _move_to_canonical_x(state: V88State, target_x: jax.Array) -> Tuple[V88State, jax.Array]:
    raw_target = _target_raw_x(state.active, state.rotation, target_x)

    # 18 is safely above every possible horizontal travel distance on this board.
    def body(_, s):
        delta = raw_target - s.x
        direction = jnp.sign(delta).astype(jnp.int16)
        should_move = delta != 0
        proposed_x = (s.x + direction).astype(jnp.int16)
        blocked = collision(
            s.board,
            s.active,
            s.rotation,
            proposed_x,
            s.y,
        )
        apply = should_move & (~blocked)
        return s._replace(
            x=jnp.where(apply, proposed_x, s.x).astype(jnp.int16)
        )

    state = jax.lax.fori_loop(0, 18, body, state)
    reached = (
        canonical_x(state) == target_x.astype(jnp.int16)
    )
    return state, reached


def _hard_drop_y(state: V88State) -> jax.Array:
    def cond(y):
        return ~collision(
            state.board,
            state.active,
            state.rotation,
            state.x,
            (y + 1).astype(jnp.int16),
        )

    def body(y):
        return (y + 1).astype(jnp.int16)

    return jax.lax.while_loop(cond, body, state.y)


def _lock_and_spawn(state: V88State) -> Tuple[V88State, jax.Array]:
    """Match classic commit_active_tetromino semantics."""
    colliding_now = collision(
        state.board,
        state.active,
        state.rotation,
        state.x,
        state.y,
    )

    def gameover_branch(s):
        # Classic collision-before-drop path does not reset the hold flag.
        return s._replace(game_over=jnp.bool_(True)), jnp.int32(0)

    def normal_branch(s):
        final_y = _hard_drop_y(s)
        board = _project(
            s.board,
            s.active,
            s.rotation,
            s.x,
            final_y,
        )
        board, lines = _clear_rows(board)

        next_active, s2 = _pop_visible_queue(s)
        spawn_x = _spawn_x(next_active)
        spawn_y = jnp.int16(0)
        spawn_rotation = jnp.int8(0)

        game_over = collision(
            board,
            next_active,
            spawn_rotation,
            spawn_x,
            spawn_y,
        )

        # score is diagnostic only in this project; keep a deterministic
        # classic-style cumulative value for convenience.
        step_score = (lines.astype(jnp.float32) ** 2) * WIDTH + 1.0
        step_score = jnp.where(game_over, 0.0, step_score)

        s2 = s2._replace(
            board=board.astype(jnp.int8),
            active=next_active.astype(jnp.int8),
            rotation=spawn_rotation,
            x=spawn_x,
            y=spawn_y,
            can_hold=jnp.bool_(True),
            game_over=game_over,
            score=s.score + step_score,
        )
        return s2, lines.astype(jnp.int32)

    return jax.lax.cond(
        colliding_now,
        gameover_branch,
        normal_branch,
        state,
    )


def simulate_placement_one(
    state: V88State,
    rotation: jax.Array,
    x: jax.Array,
    use_hold: jax.Array,
) -> Tuple[V88State, jax.Array, jax.Array]:
    """
    Simulate one PlacementAction.

    Returns
    -------
    next_state
    lines_cleared
    valid_reachable

    Invalid/unreachable actions leave the state unchanged.
    """
    requested_hold = use_hold.astype(jnp.bool_)
    x = x.astype(jnp.int16)
    rotation = rotation.astype(jnp.int8)

    hold_allowed = (~requested_hold) | state.can_hold
    state_after_hold, actual_hold_allowed = jax.lax.cond(
        requested_hold,
        lambda s: _hold(s),
        lambda s: (s, jnp.bool_(True)),
        state,
    )

    valid = hold_allowed & actual_hold_allowed & (~state.game_over)

    state_after_rotation, rotation_ok = _rotate_to(
        state_after_hold,
        rotation,
    )
    valid = valid & rotation_ok

    state_after_move, x_ok = _move_to_canonical_x(
        state_after_rotation,
        x,
    )
    valid = valid & x_ok

    def commit(s):
        return _lock_and_spawn(s)

    def reject(s):
        return s, jnp.int32(0)

    next_state, lines = jax.lax.cond(
        valid,
        commit,
        reject,
        state_after_move,
    )

    # Invalid candidate must be a true no-op from the original state.
    next_state = jax.tree_util.tree_map(
        lambda new, old: jnp.where(valid, new, old),
        next_state,
        state,
    )
    lines = jnp.where(valid, lines, 0).astype(jnp.int32)

    return next_state, lines, valid


simulate_placement_one_jit = jax.jit(simulate_placement_one)

# Fixed candidate table: no-hold first, then hold; within each:
# rotation 0..3, x 0..9.
_C_USE_HOLD = jnp.repeat(jnp.asarray([0, 1], dtype=jnp.int8), 4 * WIDTH)
_C_ROTATION = jnp.tile(
    jnp.repeat(jnp.arange(4, dtype=jnp.int8), WIDTH),
    2,
)
_C_X = jnp.tile(jnp.arange(WIDTH, dtype=jnp.int16), 2 * 4)


def candidate_table() -> Tuple[jax.Array, jax.Array, jax.Array]:
    return _C_ROTATION, _C_X, _C_USE_HOLD


def _all_candidates_one(state: V88State):
    return jax.vmap(
        simulate_placement_one,
        in_axes=(None, 0, 0, 0),
    )(
        state,
        _C_ROTATION,
        _C_X,
        _C_USE_HOLD,
    )


all_candidates_one = jax.jit(_all_candidates_one)
all_candidates_batch = jax.jit(jax.vmap(_all_candidates_one))


def playable_board(state: V88State) -> jax.Array:
    return state.board[:HEIGHT, PADDING:PADDING + WIDTH]


def encode_observable_243(state: V88State) -> jax.Array:
    """
    Encode the project's V8 observable state:
        board200 + current7 + hold7 + next4*7 + can_hold1 = 243.

    The parity harness compares this directly against ai.state_encoder.encode_state
    before this encoder is allowed into production.
    """
    board = (playable_board(state) > 0).astype(jnp.float32).reshape(-1)
    current = jax.nn.one_hot(state.active, 7, dtype=jnp.float32)

    hold_valid = state.held_piece >= 0
    hold_index = jnp.maximum(state.held_piece, 0)
    hold = jax.nn.one_hot(hold_index, 7, dtype=jnp.float32)
    hold = hold * hold_valid.astype(jnp.float32)

    queue = jax.nn.one_hot(state.queue, 7, dtype=jnp.float32).reshape(-1)
    can_hold = jnp.asarray([state.can_hold], dtype=jnp.float32)

    return jnp.concatenate([board, current, hold, queue, can_hold], axis=0)


encode_observable_243_jit = jax.jit(encode_observable_243)
encode_observable_243_batch = jax.jit(jax.vmap(encode_observable_243))


def observable_candidate_215(
    next_state: V88State,
    rotation: jax.Array,
    x: jax.Array,
    use_hold: jax.Array,
) -> jax.Array:
    """
    V8.4/V8.7 candidate representation:
        after-action board200 + rotation4 + x10 + hold1 = 215.
    """
    board = (playable_board(next_state) > 0).astype(jnp.float32).reshape(-1)
    rot = jax.nn.one_hot(rotation.astype(jnp.int32), 4, dtype=jnp.float32)
    xpos = jax.nn.one_hot(x.astype(jnp.int32), 10, dtype=jnp.float32)
    hold = jnp.asarray([use_hold], dtype=jnp.float32)
    return jnp.concatenate([board, rot, xpos, hold], axis=0)


def candidates_215_one(candidate_states: V88State) -> jax.Array:
    return jax.vmap(observable_candidate_215)(
        candidate_states,
        _C_ROTATION,
        _C_X,
        _C_USE_HOLD,
    )


candidates_215_one_jit = jax.jit(candidates_215_one)
candidates_215_batch = jax.jit(jax.vmap(candidates_215_one))


# ---------------------------------------------------------------------------
# Python bridge helpers for parity with the existing Gym backend.
# These run only in tests / migration, not in the hot JAX rollout loop.
# ---------------------------------------------------------------------------

def _rotation_from_classic_matrix(raw, piece_index: int, matrix) -> int:
    base = np.asarray(raw.TETROMINOES[piece_index].matrix)
    target = np.asarray(matrix) > 0
    for r in range(4):
        # Convert Gym's actual matrix orientation back into the project's
        # canonical geometric-clockwise rotation number.
        candidate = np.rot90(base, k=-r) > 0
        if candidate.shape == target.shape and np.array_equal(candidate, target):
            return r
    raise RuntimeError(
        f"Could not identify rotation for {PIECE_NAMES[piece_index]} "
        f"matrix shape={target.shape}"
    )


def _peek_next_gym_bag(randomizer) -> np.ndarray:
    """
    Return the bag that Gym will use after the CURRENT bag is exhausted,
    without mutating the live environment.

    BagRandomizer.__copy__ preserves both the current bag/index and the PCG64
    bit-generator state. Consuming exactly the remaining pieces triggers Gym's
    own shuffle_bag(), giving us an exact lookahead bag for parity.
    """
    clone = pycopy.copy(randomizer)
    remaining = BAG_SIZE - int(clone.index)

    # If index is 0, the current bag has 7 pieces remaining; consuming all 7
    # triggers the next shuffle. If index is 6, a single draw triggers it.
    for _ in range(remaining):
        clone.get_next_tetromino()

    return np.asarray(clone.bag, dtype=np.int8).copy()


def snapshot_from_gym_raw(raw, *, key_seed: int = 0) -> V88State:
    """
    Convert the current classic Tetris-Gymnasium raw state into V88State.

    This preserves the currently visible queue and current BagRandomizer
    bag/index, which is sufficient for exact one-placement parity even though
    NumPy-PCG64 and JAX PRNG use different shuffle algorithms.
    """
    base_pixel_count = len(raw.base_pixels)

    active_piece = int(raw.active_tetromino.id) - base_pixel_count
    active_rotation = _rotation_from_classic_matrix(
        raw,
        active_piece,
        raw.active_tetromino.matrix,
    )

    held = raw.holder.get_tetrominoes()
    if held:
        held_piece = int(held[0].id) - base_pixel_count
        held_rotation = _rotation_from_classic_matrix(
            raw,
            held_piece,
            held[0].matrix,
        )
    else:
        held_piece = -1
        held_rotation = 0

    queue = np.asarray(raw.queue.get_queue(), dtype=np.int8)
    if queue.shape != (QUEUE_SIZE,):
        raise RuntimeError(
            f"Expected classic visible queue shape (4,), got {queue.shape}"
        )

    randomizer = raw.randomizer
    if not hasattr(randomizer, "bag") or not hasattr(randomizer, "index"):
        raise RuntimeError(
            "V8.8 parity bridge currently requires classic BagRandomizer."
        )

    bag = np.asarray(randomizer.bag, dtype=np.int8)
    if bag.shape != (BAG_SIZE,):
        raise RuntimeError(
            f"Expected 7-bag shape (7,), got {bag.shape}"
        )

    next_gym_bag = _peek_next_gym_bag(randomizer)
    if next_gym_bag.shape != (BAG_SIZE,):
        raise RuntimeError(
            f"Expected prefetched next 7-bag shape (7,), got {next_gym_bag.shape}"
        )

    return V88State(
        rng_key=jax.random.PRNGKey(int(key_seed)),
        board=jnp.asarray(raw.board, dtype=jnp.int8),
        active=jnp.int8(active_piece),
        rotation=jnp.int8(active_rotation),
        x=jnp.int16(raw.x),
        y=jnp.int16(raw.y),
        queue=jnp.asarray(queue, dtype=jnp.int8),
        bag=jnp.asarray(bag, dtype=jnp.int8),
        bag_index=jnp.int16(int(randomizer.index)),
        next_bag=jnp.asarray(next_gym_bag, dtype=jnp.int8),
        held_piece=jnp.int8(held_piece),
        held_rotation=jnp.int8(held_rotation),
        can_hold=jnp.bool_(not bool(raw.has_swapped)),
        game_over=jnp.bool_(bool(raw.game_over)),
        # tetris-gymnasium 0.3.1 defines raw.score as a METHOD
        # (score(rows_cleared)), while get_state()/set_state() may also leave
        # a numeric value in this attribute. V8.8 does not use cumulative
        # score for policy/state encoding, so never call a callable here.
        score=jnp.float32(
            float(raw.score)
            if not callable(getattr(raw, "score", None))
            else 0.0
        ),
    )


def state_to_numpy_dict(state: V88State) -> dict:
    return {
        "board": np.asarray(state.board),
        "active": int(np.asarray(state.active)),
        "rotation": int(np.asarray(state.rotation)),
        "x": int(np.asarray(state.x)),
        "y": int(np.asarray(state.y)),
        "queue": np.asarray(state.queue),
        "bag": np.asarray(state.bag),
        "bag_index": int(np.asarray(state.bag_index)),
        "next_bag": np.asarray(state.next_bag),
        "held_piece": int(np.asarray(state.held_piece)),
        "held_rotation": int(np.asarray(state.held_rotation)),
        "can_hold": bool(np.asarray(state.can_hold)),
        "game_over": bool(np.asarray(state.game_over)),
        "score": float(np.asarray(state.score)),
    }
