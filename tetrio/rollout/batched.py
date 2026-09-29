from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
import os
import time
from typing import Iterable

import numpy as np
import torch

from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetrio.future.features import feature_index
from tetrio.future.lookahead import (
    FutureCandidateInput,
    FutureFeatureConfig,
    build_row_future_features,
)
from tetrio.network.encoding import (
    PREVIEW_DEPTH,
    dense_candidate_batch,
    dense_state_batch,
    pack_board,
    piece_id,
)
from tetrio.reachability import enumerate_tetrio_reachable_placements
from tetrio.ruleset import TETRIO_MOVEMENT
from tetrio.tools.build_expert_v1_1_future_cache import select_inference_shortlist
from tetrio.tools.watch_expert_v0 import (
    BranchPlan,
    RolloutStats,
    SevenBagQueue,
    aggregate_results,
    board_metrics,
)
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.types import PieceState


TDEST = feature_index("t_opportunity_destroyed")
TDEFER = feature_index("t_cashout_deferred")


@dataclass(frozen=True)
class BatchedRolloutConfig:
    max_pieces: int = 5000
    backend: str = "fast"
    fast_max_states: int = 10_000
    reference_max_states: int = 50_000
    reference_audit_every: int = 250
    workers: int = max(1, min(16, (os.cpu_count() or 2) - 2))
    state_batch: int = 20
    progress_every: int = 512
    top_overall: int = 8
    top_per_branch: int = 4
    collect_trace: bool = False
    adaptive_future_scheduling: bool = True
    future_search_cache: bool = True
    future_feature_memo: bool = True
    future_max_chunks_per_row: int = 4


@dataclass(frozen=True)
class CandidateRecord:
    state: PieceState
    board_after: np.ndarray
    lines: int
    use_hold: bool
    branch_mode: str


@dataclass(frozen=True)
class EnumeratedState:
    seed: int
    candidates: tuple[CandidateRecord, ...]
    branch_plans: tuple[BranchPlan, ...]
    fast_reference_audits: int
    fast_reference_fallbacks: int
    any_audit: bool
    error: str | None = None


@dataclass
class RolloutState:
    seed: int
    max_pieces: int
    stream: SevenBagQueue = field(init=False)
    board: np.ndarray = field(init=False)
    active: str = field(init=False)
    hold: str | None = field(init=False, default=None)
    stats: RolloutStats = field(init=False)
    risk_events: list[dict] = field(init=False, default_factory=list)
    trace: list[dict] = field(init=False, default_factory=list)
    terminal_reason: str = ""
    game_over: bool = False
    t_destroyed_moves: int = 0
    t_deferred_moves: int = 0

    def __post_init__(self) -> None:
        self.stream = SevenBagQueue(int(self.seed))
        self.board = TETRIO_MOVEMENT.empty_board()
        self.active = self.stream.pop()
        self.hold = None
        self.stats = RolloutStats()
        self._ensure_preview()

    def _ensure_preview(self) -> None:
        self.stream.ensure(PREVIEW_DEPTH + 2)

    def preview(self) -> tuple[str, ...]:
        self._ensure_preview()
        return self.stream.peek(PREVIEW_DEPTH)

    def plan_branch(self, use_hold: bool) -> BranchPlan:
        self._ensure_preview()
        future = self.stream.peek(PREVIEW_DEPTH + 2)

        if not use_hold:
            return BranchPlan(
                selected_piece=self.active,
                use_hold=False,
                hold_after=self.hold,
                next_active=future[0],
                consume_count=1,
                mode="no_hold",
            )

        if self.hold is None:
            return BranchPlan(
                selected_piece=future[0],
                use_hold=True,
                hold_after=self.active,
                next_active=future[1],
                consume_count=2,
                mode="hold_empty",
            )

        return BranchPlan(
            selected_piece=self.hold,
            use_hold=True,
            hold_after=self.active,
            next_active=future[0],
            consume_count=1,
            mode="hold_swap",
        )

    def finish_if_at_limit(self) -> bool:
        if self.max_pieces > 0 and self.stats.pieces >= self.max_pieces:
            self.game_over = True
            self.terminal_reason = "LIMIT"
            return True
        return False

    def mark_no_candidates(self) -> None:
        self.game_over = True
        self.terminal_reason = "NO_REACHABLE_PLACEMENTS_BOTH_BRANCHES"

    def commit(
        self,
        *,
        candidates: tuple[CandidateRecord, ...],
        scores: np.ndarray,
        chosen_index: int,
        chosen_branch: BranchPlan,
        holes_before: int,
        min_candidate_holes: int,
        safer_index: int | None,
        any_audit: bool,
        collect_trace: bool,
        t_destroyed: int = 0,
        t_deferred: int = 0,
    ) -> None:
        chosen = candidates[int(chosen_index)]
        chosen_holes = board_metrics(chosen.board_after)[1]
        chosen_hole_delta = int(chosen_holes - holes_before)
        avoidable = (
            chosen_hole_delta > 0
            and int(min_candidate_holes) <= int(holes_before)
        )

        self.board = np.asarray(chosen.board_after, dtype=np.uint8).copy()

        if chosen.use_hold:
            self.stats.holds += 1
        self.stats.pieces += 1
        self.stats.candidate_sum += len(candidates)
        if any_audit:
            # Actual per-branch audit/fallback counters are added by the runner.
            pass

        if chosen_hole_delta > 0:
            self.stats.hole_creation_moves += 1
        if avoidable:
            self.stats.avoidable_hole_moves += 1
            safer = None if safer_index is None else candidates[int(safer_index)]
            self.risk_events.append(
                {
                    "piece_index": self.stats.pieces,
                    "active": self.active,
                    "hold_before": self.hold,
                    "preview": list(self.preview()),
                    "use_hold": bool(chosen.use_hold),
                    "branch_mode": chosen_branch.mode,
                    "chosen": {
                        "piece": chosen.state.piece,
                        "rotation": int(chosen.state.rotation) % 4,
                        "x": int(chosen.state.x),
                        "y": int(chosen.state.y),
                        "score": float(scores[chosen_index]),
                        "lines": int(chosen.lines),
                        "holes_after": int(chosen_holes),
                    },
                    "holes_before": int(holes_before),
                    "hole_delta": int(chosen_hole_delta),
                    "min_candidate_holes": int(min_candidate_holes),
                    "safer_alternative": None if safer is None else {
                        "piece": safer.state.piece,
                        "rotation": int(safer.state.rotation) % 4,
                        "x": int(safer.state.x),
                        "y": int(safer.state.y),
                        "score": float(scores[safer_index]),
                        "lines": int(safer.lines),
                        "holes_after": int(min_candidate_holes),
                        "score_gap_vs_chosen": float(
                            scores[chosen_index] - scores[safer_index]
                        ),
                    },
                }
            )

        if chosen.lines in self.stats.line_counts:
            self.stats.line_counts[chosen.lines] += 1

        consumed = [
            self.stream.pop()
            for _ in range(chosen_branch.consume_count)
        ]
        if chosen_branch.mode == "hold_empty":
            if consumed[0] != chosen_branch.selected_piece:
                raise RuntimeError("7-bag hold-empty selected-piece drift")
            if consumed[1] != chosen_branch.next_active:
                raise RuntimeError("7-bag hold-empty next-active drift")
        else:
            if consumed[0] != chosen_branch.next_active:
                raise RuntimeError("7-bag next-active drift")

        if collect_trace:
            self.trace.append(
                {
                    "piece_index": int(self.stats.pieces),
                    "active": str(self.active),
                    "hold_before": self.hold,
                    "preview": list(self.preview()),
                    "use_hold": bool(chosen.use_hold),
                    "branch_mode": chosen_branch.mode,
                    "piece": chosen.state.piece,
                    "rotation": int(chosen.state.rotation) % 4,
                    "x": int(chosen.state.x),
                    "y": int(chosen.state.y),
                    "lines": int(chosen.lines),
                }
            )

        self.hold = chosen_branch.hold_after
        self.active = chosen_branch.next_active

        height, holes = board_metrics(self.board)
        self.stats.current_height = int(height)
        self.stats.max_height = max(self.stats.max_height, int(height))
        self.stats.current_holes = int(holes)
        self.stats.max_holes = max(self.stats.max_holes, int(holes))
        self.stats.height_sum += int(height)

        self.t_destroyed_moves += int(bool(t_destroyed))
        self.t_deferred_moves += int(bool(t_deferred))

        self._ensure_preview()
        self.finish_if_at_limit()

    def result(self) -> dict:
        pieces = self.stats.pieces
        result = {
            "seed": int(self.seed),
            "pieces": int(pieces),
            "lines": int(self.stats.lines),
            "singles": int(self.stats.line_counts[1]),
            "doubles": int(self.stats.line_counts[2]),
            "triples": int(self.stats.line_counts[3]),
            "tetrises": int(self.stats.tetrises),
            "holds": int(self.stats.holds),
            "hold_rate": float(self.stats.hold_rate),
            "avg_height": float(self.stats.avg_height),
            "max_height": int(self.stats.max_height),
            "holes": int(self.stats.current_holes),
            "max_holes": int(self.stats.max_holes),
            "avg_candidates": float(self.stats.avg_candidates),
            "fast_reference_audits": int(self.stats.fast_reference_audits),
            "fast_reference_fallbacks": int(self.stats.fast_reference_fallbacks),
            "hole_creation_moves": int(self.stats.hole_creation_moves),
            "avoidable_hole_moves": int(self.stats.avoidable_hole_moves),
            "hole_creation_rate": (
                0.0
                if pieces == 0
                else self.stats.hole_creation_moves / pieces
            ),
            "avoidable_hole_rate": (
                0.0
                if pieces == 0
                else self.stats.avoidable_hole_moves / pieces
            ),
            "risk_events": list(self.risk_events),
            "game_over": bool(
                self.game_over and self.terminal_reason != "LIMIT"
            ),
            "terminal_reason": str(self.terminal_reason),
            "t_destroyed_moves": int(self.t_destroyed_moves),
            "t_deferred_moves": int(self.t_deferred_moves),
            "t_destroyed_rate": (
                0.0 if pieces == 0 else self.t_destroyed_moves / pieces
            ),
            "t_deferred_rate": (
                0.0 if pieces == 0 else self.t_deferred_moves / pieces
            ),
        }
        if self.trace:
            result["trace"] = list(self.trace)
        return result


def _reference_landings(
    board: np.ndarray,
    piece: str,
    max_states: int,
) -> list[PieceState]:
    best = {}
    for placement in enumerate_tetrio_reachable_placements(
        board,
        piece,
        max_states=int(max_states),
    ):
        key = placement.landing_state.geometry_key()
        old = best.get(key)
        if old is None or len(placement.path) < len(old.path):
            best[key] = placement
    return [
        p.landing_state
        for p in sorted(
            best.values(),
            key=lambda p: (
                p.landing_state.rotation % 4,
                p.landing_state.x,
                p.landing_state.y,
                len(p.path),
            ),
        )
    ]


def _geometry_set(states: Iterable[PieceState]) -> set[tuple]:
    return {s.geometry_key() for s in states}


def _enumerate_branch(
    *,
    board: np.ndarray,
    piece: str,
    backend: str,
    fast_max_states: int,
    reference_max_states: int,
    audit: bool,
    seed: int,
    piece_index: int,
) -> tuple[list[PieceState], int, int, bool]:
    if backend == "reference":
        return (
            _reference_landings(
                board,
                piece,
                reference_max_states,
            ),
            0,
            0,
            False,
        )

    fast = enumerate_tetrio_reachable_geometries_fast(
        board,
        piece,
        max_states=int(fast_max_states),
    )
    if not fast:
        reference = _reference_landings(
            board,
            piece,
            reference_max_states,
        )
        if reference:
            return reference, 0, 1, True
        return [], 0, 0, True

    if audit:
        reference = _reference_landings(
            board,
            piece,
            reference_max_states,
        )
        fast_keys = _geometry_set(fast)
        ref_keys = _geometry_set(reference)
        if fast_keys != ref_keys:
            only_fast = sorted(fast_keys - ref_keys)[:8]
            only_ref = sorted(ref_keys - fast_keys)[:8]
            raise RuntimeError(
                "FAST/REFERENCE BATCHED ROLLOUT PARITY FAILURE "
                f"seed={seed} piece_index={piece_index} piece={piece} "
                f"fast={len(fast_keys)} ref={len(ref_keys)} "
                f"only_fast={only_fast} only_ref={only_ref}"
            )
        return fast, 1, 0, True

    return fast, 0, 0, False


def _enumerate_state(task) -> EnumeratedState:
    (
        seed,
        piece_index,
        board,
        active,
        hold,
        preview,
        backend,
        fast_max_states,
        reference_max_states,
        reference_audit_every,
    ) = task

    try:
        board = np.asarray(board, dtype=np.uint8).reshape(40, 10)
        preview = tuple(preview)
        plans: list[BranchPlan] = []
        candidates: list[CandidateRecord] = []
        audits = 0
        fallbacks = 0
        any_audit = False
        audit_now = (
            backend == "fast"
            and int(reference_audit_every) > 0
            and int(piece_index) % int(reference_audit_every) == 0
        )

        for use_hold in (False, True):
            future = preview
            if not use_hold:
                plan = BranchPlan(
                    selected_piece=active,
                    use_hold=False,
                    hold_after=hold,
                    next_active=future[0],
                    consume_count=1,
                    mode="no_hold",
                )
            elif hold is None:
                plan = BranchPlan(
                    selected_piece=future[0],
                    use_hold=True,
                    hold_after=active,
                    next_active=future[1],
                    consume_count=2,
                    mode="hold_empty",
                )
            else:
                plan = BranchPlan(
                    selected_piece=hold,
                    use_hold=True,
                    hold_after=active,
                    next_active=future[0],
                    consume_count=1,
                    mode="hold_swap",
                )

            landings, a, f, audited = _enumerate_branch(
                board=board,
                piece=plan.selected_piece,
                backend=backend,
                fast_max_states=fast_max_states,
                reference_max_states=reference_max_states,
                audit=audit_now,
                seed=seed,
                piece_index=piece_index,
            )
            audits += a
            fallbacks += f
            any_audit = any_audit or audited

            for landing in landings:
                locked = lock_piece(board, landing, TETRIO_MOVEMENT)
                after, cleared = clear_lines(locked, TETRIO_MOVEMENT)
                candidates.append(
                    CandidateRecord(
                        state=landing,
                        board_after=np.asarray(after, dtype=np.uint8),
                        lines=int(cleared),
                        use_hold=bool(use_hold),
                        branch_mode=plan.mode,
                    )
                )
                plans.append(plan)

        return EnumeratedState(
            seed=int(seed),
            candidates=tuple(candidates),
            branch_plans=tuple(plans),
            fast_reference_audits=int(audits),
            fast_reference_fallbacks=int(fallbacks),
            any_audit=bool(any_audit),
            error=None,
        )
    except Exception as exc:
        return EnumeratedState(
            seed=int(seed),
            candidates=(),
            branch_plans=(),
            fast_reference_audits=0,
            fast_reference_fallbacks=0,
            any_audit=False,
            error=f"{type(exc).__name__}:{exc}",
        )


def _future_state_task(task) -> np.ndarray:
    (
        board_before,
        active,
        hold,
        preview,
        candidate_payload,
        fast_max_states,
        reference_max_states,
    ) = task

    candidates = tuple(
        FutureCandidateInput(
            board_after=np.asarray(item[0], dtype=np.uint8),
            piece=str(item[1]),
            rotation=int(item[2]),
            x=int(item[3]),
            y=int(item[4]),
            use_hold=bool(item[5]),
            lines=int(item[6]),
        )
        for item in candidate_payload
    )
    return build_row_future_features(
        board_before=np.asarray(board_before, dtype=np.uint8),
        active=str(active),
        hold=None if hold is None else str(hold),
        preview=tuple(preview),
        candidates=candidates,
        config=FutureFeatureConfig(
            fast_max_states=int(fast_max_states),
            reference_max_states=int(reference_max_states),
            tactical_preview_depth=2,
            exact_immediate_t=False,
        ),
    )



def _future_chunk_task(task):
    (
        row_index,
        start,
        board_before,
        active,
        hold,
        preview,
        candidate_payload,
        fast_max_states,
        reference_max_states,
        use_search_cache,
        use_feature_memo,
    ) = task
    candidates = tuple(
        FutureCandidateInput(
            board_after=np.asarray(item[0], dtype=np.uint8),
            piece=str(item[1]),
            rotation=int(item[2]),
            x=int(item[3]),
            y=int(item[4]),
            use_hold=bool(item[5]),
            lines=int(item[6]),
        )
        for item in candidate_payload
    )
    features = build_row_future_features(
        board_before=np.asarray(board_before, dtype=np.uint8),
        active=str(active),
        hold=None if hold is None else str(hold),
        preview=tuple(preview),
        candidates=candidates,
        config=FutureFeatureConfig(
            fast_max_states=int(fast_max_states),
            reference_max_states=int(reference_max_states),
            tactical_preview_depth=2,
            exact_immediate_t=False,
            use_search_cache=bool(use_search_cache),
            use_feature_memo=bool(use_feature_memo),
        ),
    )
    return int(row_index), int(start), features


def _parallel_future_features(
    *,
    states: list[RolloutState],
    enumerated: list[EnumeratedState],
    shortlists: list[list[int]],
    executor: ProcessPoolExecutor,
    config: BatchedRolloutConfig,
) -> list[np.ndarray]:
    rows = len(states)
    if rows == 0:
        return []

    chunks_per_row = 1
    if config.adaptive_future_scheduling and rows < int(config.workers):
        chunks_per_row = min(
            max(1, int(config.future_max_chunks_per_row)),
            max(1, int(np.ceil(int(config.workers) / max(rows, 1)))),
        )

    tasks = []
    expected = []
    for row, (state, enum, short) in enumerate(zip(states, enumerated, shortlists)):
        k = len(short)
        expected.append(k)
        if k == 0:
            continue
        chunks = min(chunks_per_row, k)
        bounds = np.linspace(0, k, num=chunks + 1, dtype=np.int64)
        for c in range(chunks):
            start = int(bounds[c])
            stop = int(bounds[c + 1])
            if stop <= start:
                continue
            payload = tuple(
                (
                    enum.candidates[i].board_after,
                    enum.candidates[i].state.piece,
                    enum.candidates[i].state.rotation,
                    enum.candidates[i].state.x,
                    enum.candidates[i].state.y,
                    enum.candidates[i].use_hold,
                    enum.candidates[i].lines,
                )
                for i in short[start:stop]
            )
            tasks.append((
                row,
                start,
                state.board,
                state.active,
                state.hold,
                state.preview(),
                payload,
                config.fast_max_states,
                config.reference_max_states,
                config.future_search_cache,
                config.future_feature_memo,
            ))

    parts = list(executor.map(_future_chunk_task, tasks, chunksize=1))
    buffers: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(rows)]
    for row, start, feat in parts:
        buffers[row].append((start, feat))

    out = []
    for row in range(rows):
        ordered = sorted(buffers[row], key=lambda x: x[0])
        if not ordered:
            feat = np.empty((0, 0), dtype=np.float32)
        else:
            feat = np.concatenate([x[1] for x in ordered], axis=0)
        if int(feat.shape[0]) != int(expected[row]):
            raise RuntimeError(
                "adaptive future scheduling reassembly mismatch: "
                f"row={row} expected={expected[row]} got={feat.shape[0]}"
            )
        out.append(feat)
    return out

def _to_device(
    array: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(array))
    return t.to(
        device=device,
        non_blocking=(device.type == "cuda"),
    )


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _state_tensor_exact(
    state: RolloutState,
    device: torch.device,
) -> torch.Tensor:
    """Build the exact B=1 dense state tensor used by the legacy runner."""
    dense = dense_state_batch(
        np.stack([pack_board(state.board)]),
        np.asarray([piece_id(state.active)], dtype=np.uint8),
        np.asarray([piece_id(state.hold)], dtype=np.uint8),
        np.asarray(
            [[piece_id(p) for p in state.preview()]],
            dtype=np.uint8,
        ),
    )
    return torch.from_numpy(dense).to(
        device=device,
        non_blocking=(device.type == "cuda"),
    )


def _candidate_tensor_exact(
    candidates: tuple[CandidateRecord, ...],
    indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Build the exact (1,K,C) candidate tensor used by one legacy branch."""
    selected = [candidates[i] for i in indices]
    dense = dense_candidate_batch(
        np.stack([pack_board(c.board_after) for c in selected]),
        np.asarray([piece_id(c.state.piece) for c in selected], dtype=np.uint8),
        np.asarray([c.state.rotation for c in selected], dtype=np.uint8),
        np.asarray([c.state.x for c in selected], dtype=np.int8),
        np.asarray([c.state.y for c in selected], dtype=np.int8),
        np.asarray([int(c.use_hold) for c in selected], dtype=np.uint8),
        np.asarray([c.lines for c in selected], dtype=np.uint8),
    )
    return torch.from_numpy(dense).to(
        device=device,
        non_blocking=(device.type == "cuda"),
    ).unsqueeze(0)


def _score_state_exact(
    scorer,
    *,
    state: RolloutState,
    enum: EnumeratedState,
    device: torch.device,
) -> np.ndarray:
    """Strict legacy-equivalent Expert-v1 scoring.

    BF16 results can move slightly when GEMM shapes change. Near ties can then
    flip HOLD/NO-HOLD and change the entire closed-loop trajectory. Therefore
    the strict A/B runner preserves the legacy CUDA call shapes:
      * state encoder B=1
      * NO-HOLD candidates in one call
      * HOLD candidates in one call
    """
    state_tensor = _state_tensor_exact(state, device)
    amp = _amp_dtype(device)

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp,
        enabled=(device.type == "cuda"),
    ):
        state_latent = scorer.encode_state(state_tensor)

    out = np.empty(len(enum.candidates), dtype=np.float32)

    for use_hold in (False, True):
        indices = [
            i
            for i, c in enumerate(enum.candidates)
            if bool(c.use_hold) == use_hold
        ]
        if not indices:
            continue

        candidate_tensor = _candidate_tensor_exact(
            enum.candidates,
            indices,
            device,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp,
            enabled=(device.type == "cuda"),
        ):
            scores = scorer.score_from_state_latent(
                state_latent,
                candidate_tensor,
            )[0]

        out[np.asarray(indices, dtype=np.int64)] = (
            scores.float().cpu().numpy()
        )

    return out


def _score_many_states_exact(
    scorer,
    *,
    states: list[RolloutState],
    enumerated: list[EnumeratedState],
    device: torch.device,
) -> list[np.ndarray]:
    """CPU search stays parallel; neural scoring preserves legacy semantics."""
    return [
        _score_state_exact(
            scorer,
            state=state,
            enum=enum,
            device=device,
        )
        for state, enum in zip(states, enumerated)
    ]



def _safer_index_v1(
    *,
    candidates: tuple[CandidateRecord, ...],
    scores: np.ndarray,
    chosen_index: int,
) -> tuple[int, int, int | None]:
    _, holes_before = board_metrics(candidates[0].board_after)
    # caller replaces holes_before with the actual pre-move board. This helper
    # only computes candidate hole extrema and the safer index.
    candidate_holes = [
        board_metrics(c.board_after)[1]
        for c in candidates
    ]
    min_holes = min(candidate_holes)
    chosen_holes = int(candidate_holes[chosen_index])
    safer_index = None
    if min_holes < chosen_holes:
        safer_pool = [
            i for i, holes in enumerate(candidate_holes)
            if holes == min_holes
        ]
        safer_index = max(
            safer_pool,
            key=lambda i: float(scores[i]),
        )
    return chosen_holes, int(min_holes), safer_index


def _v11_rerank(
    model,
    *,
    states: list[RolloutState],
    enumerated: list[EnumeratedState],
    base_scores: list[np.ndarray],
    executor: ProcessPoolExecutor,
    config: BatchedRolloutConfig,
    device: torch.device,
) -> tuple[
    list[np.ndarray],
    list[int],
    list[int | None],
    list[int],
    list[int],
]:
    shortlists: list[list[int]] = []
    for state, enum, scores in zip(states, enumerated, base_scores):
        holds = np.asarray(
            [int(c.use_hold) for c in enum.candidates],
            dtype=np.uint8,
        )
        short = select_inference_shortlist(
            scores,
            holds,
            top_overall=config.top_overall,
            top_per_branch=config.top_per_branch,
        )
        shortlists.append(short)

    features_list = _parallel_future_features(
        states=states,
        enumerated=enumerated,
        shortlists=shortlists,
        executor=executor,
        config=config,
    )

    full_scores: list[np.ndarray] = []
    chosen_indices: list[int] = []
    safer_indices: list[int | None] = []
    t_destroyed: list[int] = []
    t_deferred: list[int] = []
    amp = _amp_dtype(device)

    # Legacy ExpertV11Rollout reranks one state at a time with [1,K].
    for state, enum, scores, short, feat in zip(
        states,
        enumerated,
        base_scores,
        shortlists,
        features_list,
    ):
        base_t = torch.from_numpy(
            np.asarray(scores[short], dtype=np.float32)[None, :]
        ).to(device=device)
        feat_t = torch.from_numpy(
            np.asarray(feat, dtype=np.float32)[None, :, :]
        ).to(device=device)
        hold_t = torch.from_numpy(
            np.asarray(
                [int(enum.candidates[i].use_hold) for i in short],
                dtype=np.uint8,
            )[None, :]
        ).to(device=device).bool()
        mask_t = torch.ones(
            (1, len(short)),
            device=device,
            dtype=torch.bool,
        )

        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=amp,
            enabled=(device.type == "cuda"),
        ):
            final_t, _ = model.final_scores(
                base_scores=base_t,
                raw_features=feat_t,
                candidate_use_hold=hold_t,
                mask=mask_t,
            )

        local_scores = final_t[0].float().cpu().numpy()
        winner_local = int(np.argmax(local_scores))
        winner_global = int(short[winner_local])

        global_scores = np.full(
            len(enum.candidates),
            -1e9,
            dtype=np.float32,
        )
        for local_i, global_i in enumerate(short):
            global_scores[global_i] = float(local_scores[local_i])

        full_scores.append(global_scores)
        chosen_indices.append(winner_global)

        candidate_holes = [
            board_metrics(c.board_after)[1]
            for c in enum.candidates
        ]
        chosen_holes = candidate_holes[winner_global]
        min_holes = min(candidate_holes)

        safer = None
        if min_holes < chosen_holes:
            safer_pool = [
                i
                for i, h in enumerate(candidate_holes)
                if h == min_holes
            ]
            safer = max(
                safer_pool,
                key=lambda i: float(global_scores[i]),
            )

        safer_indices.append(safer)
        t_destroyed.append(
            int(feat[winner_local, TDEST] > 0.5)
        )
        t_deferred.append(
            int(feat[winner_local, TDEFER] > 0.5)
        )

    return (
        full_scores,
        chosen_indices,
        safer_indices,
        t_destroyed,
        t_deferred,
    )


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _run_batched(
    model,
    *,
    seeds: list[int],
    device: torch.device,
    config: BatchedRolloutConfig,
    policy: str,
    label: str,
) -> dict:
    if config.backend not in ("fast", "reference"):
        raise ValueError("backend must be 'fast' or 'reference'")
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.state_batch < 1:
        raise ValueError("state_batch must be >= 1")
    if not seeds:
        raise ValueError("seeds cannot be empty")

    pending = iter([int(s) for s in seeds])
    active: list[RolloutState] = []
    completed: dict[int, dict] = {}

    def refill() -> None:
        while len(active) < min(config.state_batch, len(seeds)):
            try:
                seed = next(pending)
            except StopIteration:
                break
            active.append(
                RolloutState(
                    seed=seed,
                    max_pieces=config.max_pieces,
                )
            )

    refill()
    processed = 0
    total_candidates = 0
    next_progress = max(1, int(config.progress_every))
    started = time.perf_counter()

    print("-" * 112)
    print(f"{label} — STRICT-PARITY PARALLEL HEADLESS")
    print("-" * 112)
    print(f"Seeds         : {seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print(f"Workers       : {config.workers}")
    print(f"State batch   : {min(config.state_batch, len(seeds))}")
    print("Neural mode   : legacy-shape strict parity")
    print(f"Max pieces    : {config.max_pieces}")
    print(f"Backend       : {config.backend}")
    print(f"Reference audit every: {config.reference_audit_every}")
    if policy == "v11":
        print(
            f"Future shortlist: top{config.top_overall} overall "
            f"+ top{config.top_per_branch}/branch"
        )
        print(
            "Future scheduling: "
            f"{'adaptive exact' if config.adaptive_future_scheduling else 'one-state-per-task'}"
        )
        print(
            "Future execution : "
            f"search_cache={config.future_search_cache} "
            f"feature_memo={config.future_feature_memo} "
            f"max_chunks={config.future_max_chunks_per_row}"
        )
    print()

    executor = ProcessPoolExecutor(max_workers=config.workers)
    try:
        while active:
            # States that hit LIMIT on the previous commit should be retired.
            survivors = []
            for s in active:
                if s.game_over:
                    completed[s.seed] = s.result()
                else:
                    survivors.append(s)
            active = survivors
            refill()
            if not active:
                break

            tasks = [
                (
                    s.seed,
                    s.stats.pieces,
                    s.board,
                    s.active,
                    s.hold,
                    s.stream.peek(PREVIEW_DEPTH + 2),
                    config.backend,
                    config.fast_max_states,
                    config.reference_max_states,
                    config.reference_audit_every,
                )
                for s in active
            ]
            enumerated_all = list(
                executor.map(
                    _enumerate_state,
                    tasks,
                    chunksize=1,
                )
            )

            score_states: list[RolloutState] = []
            score_enums: list[EnumeratedState] = []

            for s, enum in zip(active, enumerated_all):
                if enum.error:
                    raise RuntimeError(
                        f"batched rollout worker failed seed={s.seed}: "
                        f"{enum.error}"
                    )
                s.stats.fast_reference_audits += enum.fast_reference_audits
                s.stats.fast_reference_fallbacks += enum.fast_reference_fallbacks
                if not enum.candidates:
                    s.mark_no_candidates()
                    completed[s.seed] = s.result()
                else:
                    score_states.append(s)
                    score_enums.append(enum)

            if not score_states:
                active = [s for s in active if not s.game_over]
                refill()
                continue

            base_scores = _score_many_states_exact(
                model.scorer,
                states=score_states,
                enumerated=score_enums,
                device=device,
            )

            if policy == "v1":
                final_scores = base_scores
                chosen_indices = [
                    int(np.argmax(scores))
                    for scores in final_scores
                ]
                safer_indices: list[int | None] = []
                t_destroyed = [0] * len(score_states)
                t_deferred = [0] * len(score_states)

                for enum, scores, chosen in zip(
                    score_enums,
                    final_scores,
                    chosen_indices,
                ):
                    candidate_holes = [
                        board_metrics(c.board_after)[1]
                        for c in enum.candidates
                    ]
                    min_holes = min(candidate_holes)
                    chosen_holes = candidate_holes[chosen]
                    safer = None
                    if min_holes < chosen_holes:
                        pool = [
                            i for i, h in enumerate(candidate_holes)
                            if h == min_holes
                        ]
                        safer = max(
                            pool,
                            key=lambda i: float(scores[i]),
                        )
                    safer_indices.append(safer)
            elif policy == "v11":
                (
                    final_scores,
                    chosen_indices,
                    safer_indices,
                    t_destroyed,
                    t_deferred,
                ) = _v11_rerank(
                    model,
                    states=score_states,
                    enumerated=score_enums,
                    base_scores=base_scores,
                    executor=executor,
                    config=config,
                    device=device,
                )
            else:
                raise ValueError(f"unknown policy: {policy}")

            for (
                s,
                enum,
                scores,
                chosen_index,
                safer_index,
                tdest,
                tdef,
            ) in zip(
                score_states,
                score_enums,
                final_scores,
                chosen_indices,
                safer_indices,
                t_destroyed,
                t_deferred,
            ):
                holes_before = board_metrics(s.board)[1]
                candidate_holes = [
                    board_metrics(c.board_after)[1]
                    for c in enum.candidates
                ]
                min_holes = min(candidate_holes)
                chosen_branch = enum.branch_plans[int(chosen_index)]

                s.commit(
                    candidates=enum.candidates,
                    scores=scores,
                    chosen_index=int(chosen_index),
                    chosen_branch=chosen_branch,
                    holes_before=int(holes_before),
                    min_candidate_holes=int(min_holes),
                    safer_index=safer_index,
                    any_audit=enum.any_audit,
                    collect_trace=config.collect_trace,
                    t_destroyed=int(tdest),
                    t_deferred=int(tdef),
                )
                processed += 1
                total_candidates += len(enum.candidates)

            if processed >= next_progress:
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 1e-9)
                finished = len(
                    {
                        *completed.keys(),
                        *(s.seed for s in active if s.game_over),
                    }
                )
                if config.max_pieces > 0:
                    max_total = len(seeds) * config.max_pieces
                    remaining_upper = max(0, max_total - processed)
                    eta = remaining_upper / max(rate, 1e-9)
                else:
                    eta = None
                print(
                    f"  states={processed:,} rate={rate:.1f}/s "
                    f"active={len(active)} finished={finished}/{len(seeds)} "
                    f"meanK={total_candidates/max(processed,1):.1f} "
                    f"upper-ETA={_fmt_eta(eta)}",
                    flush=True,
                )
                while next_progress <= processed:
                    next_progress += max(1, int(config.progress_every))

        for s in active:
            if s.seed not in completed:
                completed[s.seed] = s.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    elapsed = time.perf_counter() - started
    results = [completed[int(seed)] for seed in seeds]
    aggregate = aggregate_results(results)
    aggregate["mean_t_destroyed_rate"] = float(
        np.mean([r["t_destroyed_rate"] for r in results])
    )
    aggregate["mean_t_deferred_rate"] = float(
        np.mean([r["t_deferred_rate"] for r in results])
    )

    return {
        "results": results,
        "aggregate": aggregate,
        "runtime": {
            "seconds": float(elapsed),
            "processed_states": int(processed),
            "states_per_second": float(
                processed / max(elapsed, 1e-9)
            ),
            "mean_candidates": float(
                total_candidates / max(processed, 1)
            ),
            "workers": int(config.workers),
            "state_batch": int(config.state_batch),
        },
    }


def run_batched_v1(
    model,
    *,
    seeds: list[int],
    device: torch.device,
    config: BatchedRolloutConfig,
) -> dict:
    return _run_batched(
        model,
        seeds=seeds,
        device=device,
        config=config,
        policy="v1",
        label="EXPERT V1",
    )


def run_batched_v11(
    model,
    *,
    seeds: list[int],
    device: torch.device,
    config: BatchedRolloutConfig,
) -> dict:
    return _run_batched(
        model,
        seeds=seeds,
        device=device,
        config=config,
        policy="v11",
        label="EXPERT V1.1",
    )
