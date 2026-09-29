from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch

from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetrio.future.dominance import safe_dominance_mask
from tetrio.future.features import FEATURE_SIZE, feature_index
from tetrio.future.lookahead import (
    FutureCandidateInput,
    FutureFeatureConfig,
    build_row_future_features,
)
from tetrio.network.checkpoint import load_expert_v1
from tetrio.network.encoding import (
    PREVIEW_DEPTH,
    pack_board,
    piece_id,
    torch_dense_candidate_batch,
    torch_dense_state_batch,
)
from tetrio.reachability import enumerate_tetrio_reachable_placements
from tetrio.ruleset import TETRIO_MOVEMENT
from tetrio.tools.build_expert_v1_1_future_cache import select_inference_shortlist
from tetrio.tools.watch_expert_v0 import SevenBagQueue
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.types import PieceState


HDELTA = feature_index("hole_delta")
TDEST = feature_index("t_opportunity_destroyed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parallel/batched Expert-v1 recovery collector. "
            "CPU ProcessPool generates reachable/future candidates while one "
            "batched CUDA inference scores many simultaneously active seeds."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_joint_100k.pt"),
    )
    p.add_argument("--seeds", default="9031-9050")
    p.add_argument("--max-pieces", type=int, default=1000)
    p.add_argument(
        "--max-events",
        type=int,
        default=750,
        help="Stop after this many conservative recovery pairs.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 2) - 2)),
    )
    p.add_argument(
        "--state-batch",
        type=int,
        default=20,
        help=(
            "Number of independent seed trajectories advanced together. "
            "Candidate scoring from the whole group is merged into one GPU batch."
        ),
    )
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument(
        "--progress-every",
        type=int,
        default=256,
        help="Print progress after roughly this many processed states.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_recovery_states.jsonl"),
    )
    return p.parse_args()


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("No seeds parsed from --seeds")
    return out


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


@dataclass(frozen=True)
class BranchPlan:
    selected_piece: str
    use_hold: bool
    hold_after: str | None
    next_active: str
    consume_count: int
    mode: str


@dataclass
class RecoveryState:
    seed: int
    max_pieces: int
    stream: SevenBagQueue = field(init=False)
    board: np.ndarray = field(init=False)
    active: str = field(init=False)
    hold: str | None = field(init=False, default=None)
    pieces: int = 0
    terminal_reason: str = ""
    events: int = 0

    def __post_init__(self) -> None:
        self.stream = SevenBagQueue(int(self.seed))
        self.board = TETRIO_MOVEMENT.empty_board()
        self.active = self.stream.pop()
        self.hold = None
        self._ensure_preview()

    @property
    def game_over(self) -> bool:
        return bool(self.terminal_reason)

    def _ensure_preview(self) -> None:
        self.stream.ensure(PREVIEW_DEPTH + 2)

    def preview(self) -> tuple[str, ...]:
        self._ensure_preview()
        return self.stream.peek(PREVIEW_DEPTH)

    def branch_plan(self, use_hold: bool) -> BranchPlan:
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

    def commit(self, candidate: "CandidateRecord") -> None:
        plan = self.branch_plan(bool(candidate.use_hold))

        if candidate.piece != plan.selected_piece:
            raise RuntimeError(
                "collector branch/piece drift: "
                f"seed={self.seed} expected={plan.selected_piece} "
                f"got={candidate.piece}"
            )

        self.board = np.asarray(candidate.board_after, dtype=np.uint8).copy()

        consumed = [self.stream.pop() for _ in range(plan.consume_count)]
        if plan.mode == "hold_empty":
            if consumed[0] != plan.selected_piece:
                raise RuntimeError("hold-empty selected piece drift")
            if consumed[1] != plan.next_active:
                raise RuntimeError("hold-empty next-active drift")
        else:
            if consumed[0] != plan.next_active:
                raise RuntimeError("next-active drift")

        self.hold = plan.hold_after
        self.active = plan.next_active
        self.pieces += 1
        self._ensure_preview()

        if self.max_pieces > 0 and self.pieces >= self.max_pieces:
            self.terminal_reason = "LIMIT"


@dataclass(frozen=True)
class CandidateRecord:
    board_after: np.ndarray
    piece: str
    rotation: int
    x: int
    y: int
    use_hold: bool
    lines: int
    branch_mode: str


@dataclass(frozen=True)
class EnumeratedState:
    seed: int
    candidates: tuple[CandidateRecord, ...]
    used_reference_fallback: bool
    error: str | None = None


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


def _enumerate_one_state(task) -> EnumeratedState:
    (
        seed,
        board,
        active,
        hold,
        preview,
        fast_max_states,
        reference_max_states,
    ) = task

    try:
        board = np.asarray(board, dtype=np.uint8).reshape(40, 10)
        preview = tuple(preview)
        used_fallback = False
        candidates: list[CandidateRecord] = []

        def plan(use_hold: bool) -> tuple[str, str]:
            if not use_hold:
                return active, "no_hold"
            if hold is None:
                if not preview:
                    raise RuntimeError("hold-empty state missing preview[0]")
                return preview[0], "hold_empty"
            return hold, "hold_swap"

        for use_hold in (False, True):
            selected_piece, mode = plan(use_hold)
            landings = enumerate_tetrio_reachable_geometries_fast(
                board,
                selected_piece,
                max_states=int(fast_max_states),
            )

            if not landings:
                ref = _reference_landings(
                    board,
                    selected_piece,
                    int(reference_max_states),
                )
                if ref:
                    landings = ref
                    used_fallback = True

            for landing in landings:
                locked = lock_piece(board, landing, TETRIO_MOVEMENT)
                after, lines = clear_lines(locked, TETRIO_MOVEMENT)
                candidates.append(
                    CandidateRecord(
                        board_after=np.asarray(after, dtype=np.uint8),
                        piece=str(landing.piece),
                        rotation=int(landing.rotation) % 4,
                        x=int(landing.x),
                        y=int(landing.y),
                        use_hold=bool(use_hold),
                        lines=int(lines),
                        branch_mode=mode,
                    )
                )

        return EnumeratedState(
            seed=int(seed),
            candidates=tuple(candidates),
            used_reference_fallback=used_fallback,
            error=None,
        )

    except Exception as exc:
        return EnumeratedState(
            seed=int(seed),
            candidates=(),
            used_reference_fallback=False,
            error=f"{type(exc).__name__}:{exc}",
        )


def _future_one_state(task) -> np.ndarray:
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
            # Speed-V2 contract: exact path-sensitive T classification is a
            # sampled diagnostic, not a per-candidate bulk operation.
            exact_immediate_t=False,
        ),
    )


def _to_device(
    array: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        try:
            t = t.pin_memory()
        except RuntimeError:
            pass
    return t.to(
        device=device,
        non_blocking=(device.type == "cuda"),
    )


def _score_state_batch(
    model,
    *,
    states: list[RecoveryState],
    enumerated: list[EnumeratedState],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Score all candidates from many independent states in one CUDA batch.

    Returns:
      flat_scores [total_candidates]
      owner       [total_candidates] -> state row
    """
    state_board = np.stack(
        [pack_board(s.board) for s in states]
    ).astype(np.uint8)
    state_active = np.asarray(
        [piece_id(s.active) for s in states],
        dtype=np.uint8,
    )
    state_hold = np.asarray(
        [piece_id(s.hold) for s in states],
        dtype=np.uint8,
    )
    state_preview = np.asarray(
        [
            [piece_id(p) for p in s.preview()]
            for s in states
        ],
        dtype=np.uint8,
    )

    owners = []
    board_packed = []
    piece = []
    rotation = []
    xs = []
    ys = []
    use_hold = []
    lines = []

    for owner, enum in enumerate(enumerated):
        for c in enum.candidates:
            owners.append(owner)
            board_packed.append(pack_board(c.board_after))
            piece.append(piece_id(c.piece))
            rotation.append(c.rotation)
            xs.append(c.x)
            ys.append(c.y)
            use_hold.append(int(c.use_hold))
            lines.append(c.lines)

    if not owners:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    state_dense = torch_dense_state_batch(
        _to_device(state_board, device),
        _to_device(state_active, device),
        _to_device(state_hold, device),
        _to_device(state_preview, device),
    )
    candidate_dense = torch_dense_candidate_batch(
        _to_device(np.stack(board_packed).astype(np.uint8), device),
        _to_device(np.asarray(piece, dtype=np.uint8), device),
        _to_device(np.asarray(rotation, dtype=np.uint8), device),
        _to_device(np.asarray(xs, dtype=np.int8), device),
        _to_device(np.asarray(ys, dtype=np.int8), device),
        _to_device(np.asarray(use_hold, dtype=np.uint8), device),
        _to_device(np.asarray(lines, dtype=np.uint8), device),
    )
    owner_t = _to_device(
        np.asarray(owners, dtype=np.int32),
        device,
    ).long()

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        flat_scores = model.forward_flat(
            state=state_dense,
            candidates=candidate_dense,
            candidate_owner=owner_t,
        )

    return (
        flat_scores.float().cpu().numpy(),
        np.asarray(owners, dtype=np.int32),
    )


def _shortlist_for_state(
    candidates: tuple[CandidateRecord, ...],
    scores: np.ndarray,
    *,
    top_overall: int,
    top_per_branch: int,
) -> list[int]:
    holds = np.asarray(
        [int(c.use_hold) for c in candidates],
        dtype=np.uint8,
    )
    return select_inference_shortlist(
        np.asarray(scores, dtype=np.float32),
        holds,
        top_overall=int(top_overall),
        top_per_branch=int(top_per_branch),
    )


def _dominance_batch(
    features_list: list[np.ndarray],
) -> list[np.ndarray]:
    if not features_list:
        return []

    max_k = max(len(x) for x in features_list)
    padded = np.zeros(
        (len(features_list), max_k, FEATURE_SIZE),
        dtype=np.float32,
    )
    mask = np.zeros(
        (len(features_list), max_k),
        dtype=bool,
    )

    for i, feat in enumerate(features_list):
        k = len(feat)
        padded[i, :k] = feat
        mask[i, :k] = True

    with torch.inference_mode():
        dom = safe_dominance_mask(
            torch.from_numpy(padded),
            torch.from_numpy(mask),
        ).cpu().numpy()

    return [
        dom[i, : len(features_list[i]), : len(features_list[i])]
        for i in range(len(features_list))
    ]


def _event_eta(
    *,
    events: int,
    target_events: int,
    processed: int,
    elapsed: float,
) -> float | None:
    if events <= 0 or processed <= 0 or elapsed <= 0:
        return None
    events_per_sec = events / elapsed
    if events_per_sec <= 0:
        return None
    return max(0.0, target_events - events) / events_per_sec


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.state_batch < 1:
        raise SystemExit("--state-batch must be >= 1")
    if args.max_events < 1:
        raise SystemExit("--max-events must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")

    seeds = parse_seeds(args.seeds)
    device = torch.device(args.device)
    model, ckpt = load_expert_v1(
        args.checkpoint,
        device=device,
    )
    model.eval()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    pending_seeds = iter(seeds)
    active: list[RecoveryState] = []

    def refill() -> None:
        while len(active) < min(args.state_batch, len(seeds)):
            try:
                active.append(
                    RecoveryState(
                        seed=next(pending_seeds),
                        max_pieces=args.max_pieces,
                    )
                )
            except StopIteration:
                break

    refill()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_events = 0
    processed_states = 0
    total_candidates = 0
    total_shortlist = 0
    reference_fallbacks = 0
    completed_seeds = 0
    next_progress = int(args.progress_every)
    started = time.perf_counter()

    print("=" * 112)
    print("TETR.IO EXPERT V1 — RECOVERY COLLECTOR V2 (CPU PARALLEL + GPU BATCHED)")
    print("=" * 112)
    print(f"Checkpoint    : {args.checkpoint} epoch={ckpt.get('epoch')}")
    print(f"Seeds         : {seeds[0]}..{seeds[-1]} ({len(seeds)} total)")
    print(f"Max pieces    : {args.max_pieces}/seed")
    print(f"Target events : {args.max_events}")
    print(f"Workers       : {args.workers}")
    print(f"State batch   : {min(args.state_batch, len(seeds))}")
    print(
        f"Shortlist     : top{args.top_overall} overall "
        f"+ top{args.top_per_branch}/branch"
    )
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(device)}")
    print("Future mode   : FAST PROXY (no per-candidate Reference T BFS)")
    print()

    executor = ProcessPoolExecutor(max_workers=args.workers)

    try:
        with args.output.open("w", encoding="utf-8", buffering=1) as out:
            while active and total_events < args.max_events:
                # ----------------------------------------------------------
                # Phase A: CPU-parallel candidate generation for many seeds.
                # ----------------------------------------------------------
                enum_tasks = [
                    (
                        s.seed,
                        s.board,
                        s.active,
                        s.hold,
                        s.preview(),
                        args.fast_max_states,
                        args.reference_max_states,
                    )
                    for s in active
                ]
                enumerated = list(
                    executor.map(
                        _enumerate_one_state,
                        enum_tasks,
                        chunksize=1,
                    )
                )

                # Remove hard-failed/dead states before GPU scoring.
                score_states: list[RecoveryState] = []
                score_enums: list[EnumeratedState] = []

                for s, enum in zip(active, enumerated):
                    if enum.error:
                        s.terminal_reason = f"ERROR:{enum.error}"
                    elif not enum.candidates:
                        s.terminal_reason = "NO_REACHABLE_PLACEMENTS"
                    else:
                        score_states.append(s)
                        score_enums.append(enum)
                        reference_fallbacks += int(
                            enum.used_reference_fallback
                        )

                # ----------------------------------------------------------
                # Phase B: one merged CUDA batch for all candidate scores.
                # ----------------------------------------------------------
                if score_states:
                    flat_scores, owner = _score_state_batch(
                        model,
                        states=score_states,
                        enumerated=score_enums,
                        device=device,
                    )
                else:
                    flat_scores = np.empty((0,), dtype=np.float32)
                    owner = np.empty((0,), dtype=np.int32)

                score_slices: list[np.ndarray] = []
                shortlists: list[list[int]] = []
                future_tasks = []

                for row, (s, enum) in enumerate(
                    zip(score_states, score_enums)
                ):
                    idx = np.flatnonzero(owner == row)
                    scores = flat_scores[idx]
                    if len(scores) != len(enum.candidates):
                        raise RuntimeError(
                            "GPU candidate score alignment mismatch"
                        )
                    score_slices.append(scores)

                    short = _shortlist_for_state(
                        enum.candidates,
                        scores,
                        top_overall=args.top_overall,
                        top_per_branch=args.top_per_branch,
                    )
                    shortlists.append(short)

                    payload = tuple(
                        (
                            enum.candidates[i].board_after,
                            enum.candidates[i].piece,
                            enum.candidates[i].rotation,
                            enum.candidates[i].x,
                            enum.candidates[i].y,
                            enum.candidates[i].use_hold,
                            enum.candidates[i].lines,
                        )
                        for i in short
                    )
                    future_tasks.append(
                        (
                            s.board,
                            s.active,
                            s.hold,
                            s.preview(),
                            payload,
                            args.fast_max_states,
                            args.reference_max_states,
                        )
                    )

                # ----------------------------------------------------------
                # Phase C: CPU-parallel future analysis.
                # ----------------------------------------------------------
                if future_tasks:
                    features_list = list(
                        executor.map(
                            _future_one_state,
                            future_tasks,
                            chunksize=1,
                        )
                    )
                    dominance = _dominance_batch(features_list)
                else:
                    features_list = []
                    dominance = []

                # ----------------------------------------------------------
                # Phase D: create conservative pairs, then commit BASE V1 Top1.
                # Recovery collection must follow the original V1 trajectory,
                # not the recovery candidate, otherwise labels alter collection.
                # ----------------------------------------------------------
                for row, (
                    s,
                    enum,
                    scores,
                    short,
                    features,
                    dom,
                ) in enumerate(
                    zip(
                        score_states,
                        score_enums,
                        score_slices,
                        shortlists,
                        features_list,
                        dominance,
                    )
                ):
                    base_top1_global = int(np.argmax(scores))

                    # The base winner is always in top-overall shortlist.
                    try:
                        top_local = short.index(base_top1_global)
                    except ValueError as exc:
                        raise RuntimeError(
                            "base Top1 missing from recovery shortlist"
                        ) from exc

                    base_bad = bool(
                        features[top_local, HDELTA] > 0
                        or features[top_local, TDEST] > 0
                    )
                    dominators = np.flatnonzero(dom[:, top_local])

                    if (
                        base_bad
                        and dominators.size
                        and total_events < args.max_events
                    ):
                        best_dom_local = int(
                            max(
                                dominators,
                                key=lambda j: float(
                                    scores[short[int(j)]]
                                ),
                            )
                        )
                        chosen_i = base_top1_global
                        winner_i = int(short[best_dom_local])
                        chosen = enum.candidates[chosen_i]
                        winner = enum.candidates[winner_i]

                        event = {
                            "seed": int(s.seed),
                            "piece_index": int(s.pieces),
                            "active": s.active,
                            "hold": s.hold,
                            "preview": list(s.preview()),
                            "candidate_count": len(enum.candidates),
                            "shortlist_count": len(short),
                            "base_top1": {
                                "piece": chosen.piece,
                                "rotation": int(chosen.rotation) % 4,
                                "x": int(chosen.x),
                                "y": int(chosen.y),
                                "score": float(scores[chosen_i]),
                                "use_hold": bool(chosen.use_hold),
                                "future_features": (
                                    features[top_local].tolist()
                                ),
                            },
                            "dominant_recovery": {
                                "piece": winner.piece,
                                "rotation": int(winner.rotation) % 4,
                                "x": int(winner.x),
                                "y": int(winner.y),
                                "score": float(scores[winner_i]),
                                "use_hold": bool(winner.use_hold),
                                "future_features": (
                                    features[best_dom_local].tolist()
                                ),
                            },
                        }
                        out.write(
                            json.dumps(event, ensure_ascii=False) + "\n"
                        )
                        total_events += 1
                        s.events += 1

                    # Preserve the original V1 trajectory.
                    s.commit(enum.candidates[base_top1_global])

                    processed_states += 1
                    total_candidates += len(enum.candidates)
                    total_shortlist += len(short)

                # ----------------------------------------------------------
                # Finalize ended trajectories and replace with new seeds.
                # ----------------------------------------------------------
                survivors: list[RecoveryState] = []
                for s in active:
                    if s.game_over:
                        completed_seeds += 1
                        print(
                            f"  Seed {s.seed}: pieces={s.pieces:,} "
                            f"events={s.events} reason={s.terminal_reason}"
                        )
                    else:
                        survivors.append(s)
                active = survivors
                refill()

                # ----------------------------------------------------------
                # Progress / ETA.
                # ----------------------------------------------------------
                if (
                    processed_states >= next_progress
                    or total_events >= args.max_events
                ):
                    elapsed = time.perf_counter() - started
                    rate = processed_states / max(elapsed, 1e-9)
                    event_rate = (
                        total_events / max(processed_states, 1)
                    )
                    eta = _event_eta(
                        events=total_events,
                        target_events=args.max_events,
                        processed=processed_states,
                        elapsed=elapsed,
                    )
                    mean_k = (
                        total_candidates / max(processed_states, 1)
                    )
                    mean_short = (
                        total_shortlist / max(processed_states, 1)
                    )
                    print(
                        f"  states={processed_states:,} "
                        f"events={total_events}/{args.max_events} "
                        f"event_rate={event_rate:.2%} "
                        f"rate={rate:.1f} states/s "
                        f"meanK={mean_k:.1f} shortK={mean_short:.1f} "
                        f"active={len(active)} "
                        f"ETA={_fmt_eta(eta)}",
                        flush=True,
                    )
                    while next_progress <= processed_states:
                        next_progress += int(args.progress_every)

    except KeyboardInterrupt:
        print()
        print("Interrupted. JSONL events already written remain valid.")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    elapsed = time.perf_counter() - started
    rate = processed_states / max(elapsed, 1e-9)
    event_rate = total_events / max(processed_states, 1)

    print()
    print("=" * 112)
    print("RECOVERY COLLECTOR V2 RESULT")
    print("=" * 112)
    print(f"Checkpoint        : {args.checkpoint} epoch={ckpt.get('epoch')}")
    print(f"Processed states  : {processed_states:,}")
    print(f"Events            : {total_events:,}/{args.max_events:,}")
    print(f"Event rate        : {event_rate:.2%}")
    print(f"Throughput        : {rate:.2f} states/s")
    print(
        f"Mean candidates   : "
        f"{total_candidates/max(processed_states,1):.2f}"
    )
    print(
        f"Mean shortlist    : "
        f"{total_shortlist/max(processed_states,1):.2f}"
    )
    print(f"Reference fallback: {reference_fallbacks}")
    print(f"Completed seeds   : {completed_seeds}/{len(seeds)}")
    print(f"Elapsed           : {_fmt_eta(elapsed)}")
    print(f"Output            : {args.output}")
    if total_events == 0:
        print("Status            : NO EVENTS — do not train recovery phase yet")
    elif total_events < min(250, args.max_events):
        print("Status            : SMALL PILOT — usable for diagnostics only")
    else:
        print("Status            : RECOVERY PAIRS READY FOR V1.1 PILOT")
    print(
        "Pairs remain conservative relative preferences; "
        "they are not oracle action labels."
    )


if __name__ == "__main__":
    main()
