from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetrio.future.search_cache import fast_landings, occupancy_key
from tetrio.future.features import FEATURE_NAMES
from tetrio.future.state_transition import (
    KnownFutureState,
    advance_after_lock,
    available_next_branch_pieces,
    t_distance_flags,
    t_is_near,
)
from tetrio.future.structural_features import (
    BoardStructure,
    board_structure,
    board_structure_batch,
)
from tetrio.future.tactical_cavity import (
    TOpportunity,
    TSpinTarget,
    classify_target_t_spin,
    scan_t_opportunities,
)
from tetrio.network.encoding import unpack_boards
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.types import PieceState


@dataclass(frozen=True)
class FutureFeatureConfig:
    fast_max_states: int = 10_000
    reference_max_states: int = 50_000
    tactical_preview_depth: int = 2
    # Bulk cache default. Exact path-sensitive T-spin reconstruction is kept
    # for sampled diagnostics, not every shortlisted candidate.
    exact_immediate_t: bool = False
    # Exact process-local memoization only. Disabling it must preserve features.
    use_search_cache: bool = True
    # Per-row exact memoization; execution-only, policy must remain identical.
    use_feature_memo: bool = True


@dataclass(frozen=True)
class FutureCandidateInput:
    board_after: np.ndarray
    piece: str
    rotation: int
    x: int
    y: int
    use_hold: bool
    lines: int

    @property
    def landing_state(self) -> PieceState:
        return PieceState(
            piece=self.piece,
            x=int(self.x),
            y=int(self.y),
            rotation=int(self.rotation) % 4,
        )


@dataclass(frozen=True)
class NextEnvelope:
    candidate_count: int
    min_holes: int
    min_height: int
    min_bumpiness: int
    max_lines: int
    no_new_hole_fraction: float
    dead_end: bool


def _next_envelope(
    board_after: np.ndarray,
    future_state: KnownFutureState,
    *,
    holes_after: int,
    config: FutureFeatureConfig,
) -> NextEnvelope:
    # Reachability itself remains unchanged. The safe optimization is to batch
    # the purely structural integer metrics after all legal next boards have
    # been produced, instead of repeatedly scanning 40x10 boards in Python.
    next_boards: list[np.ndarray] = []
    next_lines: list[int] = []

    for _, piece in available_next_branch_pieces(future_state):
        for landing in fast_landings(
            board_after,
            piece,
            max_states=int(config.fast_max_states),
            use_cache=bool(config.use_search_cache),
        ):
            locked = lock_piece(board_after, landing, TETRIO_MOVEMENT)
            after2, lines = clear_lines(locked, TETRIO_MOVEMENT)
            next_boards.append(np.asarray(after2, dtype=np.uint8))
            next_lines.append(int(lines))

    count = len(next_boards)
    if count == 0:
        return NextEnvelope(
            candidate_count=0,
            min_holes=holes_after + 8,
            min_height=40,
            min_bumpiness=80,
            max_lines=0,
            no_new_hole_fraction=0.0,
            dead_end=True,
        )

    metrics = board_structure_batch(np.stack(next_boards, axis=0))
    holes = metrics.holes
    return NextEnvelope(
        candidate_count=count,
        min_holes=int(holes.min()),
        min_height=int(metrics.max_height.min()),
        min_bumpiness=int(metrics.bumpiness.min()),
        max_lines=int(max(next_lines)),
        no_new_hole_fraction=float(np.count_nonzero(holes <= holes_after) / count),
        dead_end=False,
    )


def _t_before_for_row(
    board_before: np.ndarray,
    *,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
    config: FutureFeatureConfig,
) -> TOpportunity:
    near = t_is_near(
        active=active,
        hold=hold,
        preview=preview,
        depth=config.tactical_preview_depth,
    )
    if not near:
        return TOpportunity()

    return scan_t_opportunities(
        board_before,
        fast_max_states=config.fast_max_states,
        reference_max_states=config.reference_max_states,
        exact_if_proxy=False,
        use_fast_cache=bool(config.use_search_cache),
    )


def build_candidate_future_features(
    *,
    board_before: np.ndarray,
    structure_before: BoardStructure,
    t_before: TOpportunity,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
    candidate: FutureCandidateInput,
    config: FutureFeatureConfig,
    structure_memo: dict[bytes, BoardStructure] | None = None,
    envelope_memo: dict[tuple, NextEnvelope] | None = None,
    t_memo: dict[tuple, TOpportunity] | None = None,
) -> np.ndarray:
    after = np.asarray(candidate.board_after).reshape(40, 10).astype(np.uint8)
    after_key = occupancy_key(after)
    if structure_memo is None:
        s = board_structure(after)
    else:
        s = structure_memo.get(after_key)
        if s is None:
            s = board_structure(after)
            structure_memo[after_key] = s

    future_state = advance_after_lock(
        active=active,
        hold=hold,
        preview=preview,
        use_hold=bool(candidate.use_hold),
        placed_piece=candidate.piece,
    )
    branch_pieces = tuple(
        piece for _, piece in available_next_branch_pieces(future_state)
    )
    env_key = (after_key, branch_pieces, int(s.holes), int(config.fast_max_states))
    if envelope_memo is None:
        env = _next_envelope(
            after, future_state, holes_after=s.holes, config=config,
        )
    else:
        env = envelope_memo.get(env_key)
        if env is None:
            env = _next_envelope(
                after, future_state, holes_after=s.holes, config=config,
            )
            envelope_memo[env_key] = env

    t_active, t_hold, t_p1, t_p2 = t_distance_flags(future_state)
    t_near_after = bool(t_active or t_hold or t_p1 or t_p2)
    if t_near_after:
        t_key = (
            after_key,
            int(config.fast_max_states),
            int(config.reference_max_states),
            bool(config.use_search_cache),
        )
        if t_memo is None:
            t_after = scan_t_opportunities(
                after,
                fast_max_states=config.fast_max_states,
                reference_max_states=config.reference_max_states,
                exact_if_proxy=False,
                use_fast_cache=bool(config.use_search_cache),
            )
        else:
            t_after = t_memo.get(t_key)
            if t_after is None:
                t_after = scan_t_opportunities(
                    after,
                    fast_max_states=config.fast_max_states,
                    reference_max_states=config.reference_max_states,
                    exact_if_proxy=False,
                    use_fast_cache=bool(config.use_search_cache),
                )
                t_memo[t_key] = t_after
    else:
        t_after = TOpportunity()

    # IMPORTANT: no per-candidate reference BFS in bulk features.
    # Exact-T columns are retained in the schema for diagnostics but the model
    # neutralizes them. Current tactical value is represented by the T-slot
    # proxy + opportunity-preserved/destroyed features.
    current_t = TSpinTarget()

    t_before_total = t_before.proxy_total
    t_after_total = t_after.proxy_total
    t_relevant = int(
        active == "T"
        or hold == "T"
        or "T" in tuple(preview)[: config.tactical_preview_depth]
        or t_near_after
    )

    t_destroyed = int(
        t_relevant
        and t_before_total > 0
        and t_after_total == 0
    )
    t_created = int(
        t_relevant
        and t_before_total == 0
        and t_after_total > 0
    )
    t_cashout_deferred = int(
        active == "T"
        and candidate.use_hold
        and t_before_total > 0
    )

    values = {
        "holes_after": float(s.holes),
        "hole_delta": float(s.holes - structure_before.holes),
        "max_height_after": float(s.max_height),
        "aggregate_height_after": float(s.aggregate_height),
        "bumpiness_after": float(s.bumpiness),
        "max_well_after": float(s.max_well),
        "lines_now": float(candidate.lines),
        "next_candidate_count": float(env.candidate_count),
        "next_min_holes": float(env.min_holes),
        "next_best_hole_delta": float(env.min_holes - s.holes),
        "next_min_height": float(env.min_height),
        "next_min_bumpiness": float(env.min_bumpiness),
        "next_max_lines": float(env.max_lines),
        "next_no_new_hole_fraction": float(env.no_new_hole_fraction),
        "t_active_next": float(t_active),
        "t_hold_next": float(t_hold),
        "t_preview1": float(t_p1),
        "t_preview2": float(t_p2),
        "t_proxy_before": float(t_before.proxy_total),
        "t_proxy_after": float(t_after.proxy_total),
        "t_exact_full_after": float(t_after.exact_full),
        "t_exact_mini_after": float(t_after.exact_mini),
        "t_opportunity_destroyed": float(t_destroyed),
        "t_opportunity_created": float(t_created),
        "current_tspin_full": float(current_t.full),
        "current_tspin_mini": float(current_t.mini),
        "current_tspin_lines": float(current_t.lines),
        "t_cashout_deferred": float(t_cashout_deferred),
        "next_dead_end": float(env.dead_end),
        "use_hold": float(candidate.use_hold),
    }
    return np.asarray(
        [values[name] for name in FEATURE_NAMES],
        dtype=np.float32,
    )


def build_row_future_features(
    *,
    board_before: np.ndarray,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
    candidates: tuple[FutureCandidateInput, ...],
    config: FutureFeatureConfig,
) -> np.ndarray:
    board_before = np.asarray(board_before).reshape(40, 10).astype(np.uint8)
    structure_before = board_structure(board_before)
    t_before = _t_before_for_row(
        board_before,
        active=active,
        hold=hold,
        preview=preview,
        config=config,
    )

    if not bool(config.use_feature_memo):
        return np.stack(
            [
                build_candidate_future_features(
                    board_before=board_before,
                    structure_before=structure_before,
                    t_before=t_before,
                    active=active,
                    hold=hold,
                    preview=preview,
                    candidate=c,
                    config=config,
                )
                for c in candidates
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    structure_memo: dict[bytes, BoardStructure] = {}
    envelope_memo: dict[tuple, NextEnvelope] = {}
    t_memo: dict[tuple, TOpportunity] = {}
    rows = []
    for c in candidates:
        rows.append(
            build_candidate_future_features(
                board_before=board_before,
                structure_before=structure_before,
                t_before=t_before,
                active=active,
                hold=hold,
                preview=preview,
                candidate=c,
                config=config,
                structure_memo=structure_memo,
                envelope_memo=envelope_memo,
                t_memo=t_memo,
            )
        )
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def packed_board_to_array(packed: np.ndarray) -> np.ndarray:
    return unpack_boards(np.asarray(packed, dtype=np.uint8)[None, :])[0].reshape(40, 10)
