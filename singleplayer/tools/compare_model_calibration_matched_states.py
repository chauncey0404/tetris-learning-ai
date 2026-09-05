from __future__ import annotations

"""
Matched-state calibration comparison for V8.8.6 31.2M vs 41.2M.

Purpose
-------
The existing override scanner evaluates each model on its own trajectory, so
"same seed" is not necessarily "same state" after the policies diverge.

This tool instead:
1) builds a deterministic state corpus from BOTH already-consumed model
   trajectories;
2) evaluates BOTH checkpoints on the exact same state243 / top-k candidates;
3) runs each candidate branch from that exact state;
4) uses ONE FIXED continuation policy for every branch (31.2M by default);
5) compares ranking, gate choice, confidence, Q margin and realized outcomes.

The selected corpus is diagnostic / development-only. It is deliberately
stratified by confidence and is NOT an estimate of the natural override
distribution.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from singleplayer.tools.watch_models import (
        DecisionInfo,
        HeuristicTeacherV2,
        ModelSession,
        board_metrics,
        choose_device,
        compact_candidate_arrays,
        load_policy,
        preview_top_k_successors,
    )
except ImportError as exc:
    raise RuntimeError(
        "compare_model_calibration_matched_states.py requires the finalized "
        "tools\\watch_models.py."
    ) from exc

from singleplayer.game.executor import execute_placement
from singleplayer.network.state_encoder import encode_state


DEFAULT_CHAMPION = (
    r"models\v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
)
DEFAULT_CHALLENGER = (
    r"models\v8_8_6_control_continued_td_41200k.pt"
)
DEFAULT_SEEDS = "4761-4775"

PROTECTED_FINAL_SEEDS = set(range(6, 21))
LINE_VALUE = {0: 0, 1: 900, 2: 2000, 3: 3300, 4: 6000}


@dataclass
class ModelDecision:
    chosen_rank: int
    confidence: float
    q_teacher: float
    q_chosen: float
    q_margin_vs_teacher: float
    q_best_rank: int
    q_values: str


@dataclass
class StateEvent:
    event_id: int
    source: str
    seed: int
    piece: int
    state_hash: str
    current_piece: str
    hold_piece: str
    next4: str
    before_height: int
    before_holes: int
    candidate_actions: str
    anchor_confidence: float
    confidence_bin: str

    champion_chosen_rank: int
    champion_confidence: float
    champion_q_teacher: float
    champion_q_chosen: float
    champion_q_margin_vs_teacher: float
    champion_q_best_rank: int
    champion_q_values: str

    challenger_chosen_rank: int
    challenger_confidence: float
    challenger_q_teacher: float
    challenger_q_chosen: float
    challenger_q_margin_vs_teacher: float
    challenger_q_best_rank: int
    challenger_q_values: str

    models_disagree: bool


@dataclass
class BranchResult:
    event_id: int
    source: str
    seed: int
    piece: int
    state_hash: str
    branch_rank: int
    action: str
    teacher_top1: bool

    teacher_score: float
    normalized_reward: float
    immediate_lines: int
    immediate_height: int
    immediate_holes: int

    horizon: int
    survived: bool
    pieces_after_branch: int
    cumulative_lines_from_branch: int
    cumulative_tetrises_from_branch: int
    cumulative_value_from_branch: int
    current_height: int
    current_holes: int
    max_height_from_branch: int
    max_holes_from_branch: int
    done_reason: str


@dataclass
class MatchedOutcome:
    event_id: int
    source: str
    seed: int
    piece: int
    state_hash: str
    confidence_bin: str
    models_disagree: bool
    best_ranks: str

    champion_rank: int
    challenger_rank: int

    champion_confidence: float
    challenger_confidence: float
    champion_q_margin: float
    challenger_q_margin: float

    champion_is_best: bool
    challenger_is_best: bool

    champion_vs_teacher: str
    challenger_vs_teacher: str
    challenger_vs_champion: str

    champion_value: int
    challenger_value: int
    teacher_value: int
    best_value: int

    champion_delta_vs_teacher: int
    challenger_delta_vs_teacher: int
    challenger_minus_champion_value: int

    champion_value_regret: int
    challenger_value_regret: int

    champion_pairwise_q_correct: int
    champion_pairwise_q_total: int
    challenger_pairwise_q_correct: int
    challenger_pairwise_q_total: int


def parse_seed_spec(spec: str) -> list[int]:
    out: list[int] = []
    for raw in str(spec).split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("No seeds parsed.")
    return list(dict.fromkeys(out))


def parse_float_list(text: str) -> list[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if len(values) < 2:
        raise ValueError("Need at least two confidence-bin boundaries.")
    return values


def action_text(action) -> str:
    return (
        f"{'H ' if bool(action.use_hold) else ''}"
        f"R{int(action.rotation)} X{int(action.x)}"
    )


def state_hash(state_features: np.ndarray) -> str:
    arr = np.ascontiguousarray(
        np.asarray(state_features, dtype=np.float32).reshape(-1)
    )
    return hashlib.sha256(arr.tobytes()).hexdigest()[:20]


def confidence_bin_label(value: float, edges: list[float]) -> str:
    v = float(value)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        if lo <= v < hi or (
            last and math.isclose(v, hi, abs_tol=1e-9)
        ):
            return (
                f"[{lo:.2f},{hi:.2f}"
                f"{']' if last else ')'}"
            )
    return (
        f"<{edges[0]:.2f}"
        if v < edges[0]
        else f">={edges[-1]:.2f}"
    )


def candidate_board_metrics(features: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(features, dtype=np.float32).reshape(-1)
    board = (arr[:200].reshape(20, 10) > 0.5).astype(np.uint8)
    return board_metrics(board)


def force_successor(
    session: ModelSession,
    successor,
    *,
    candidate_index: int,
    q_values: np.ndarray,
    confidence: float,
) -> None:
    """Commit one candidate using the same bookkeeping as ModelSession.step."""
    piece_before = str(session.state.current_piece)
    result = execute_placement(session.adapter, successor.action)

    session.state = result["state"]
    session.state_features = encode_state(session.state).astype(
        np.float32,
        copy=True,
    )

    lines = int(result["info"].get("lines_cleared", 0))
    if lines in session.stats.line_counts:
        session.stats.line_counts[lines] += 1

    session.stats.pieces += 1
    h, holes = board_metrics(session.state.board)
    session.stats.current_height = int(h)
    session.stats.max_height = max(session.stats.max_height, int(h))
    session.stats.holes = int(holes)
    session.stats.max_holes = max(session.stats.max_holes, int(holes))
    session.stats.height_sum += int(h)

    if candidate_index != 0:
        session.stats.interventions += 1

    session.last = DecisionInfo(
        piece=piece_before,
        action_text=action_text(successor.action),
        source=(
            "Teacher"
            if candidate_index == 0
            else f"FORCED #{candidate_index + 1}"
        ),
        chosen_index=int(candidate_index),
        confidence=float(confidence),
        q_values=[float(x) for x in q_values],
        teacher_score=float(successor.teacher_score),
        lines_cleared=lines,
    )
    session.visual_drop = None

    if bool(result["terminated"]):
        session.done = True
        session.game_over = True
        session.done_reason = "GAME OVER"
    elif bool(result["truncated"]):
        session.done = True
        session.game_over = False
        session.done_reason = "TRUNCATED"


def inspect_successors(session: ModelSession):
    successors = preview_top_k_successors(
        adapter=session.adapter,
        teacher=session.teacher,
        state=session.state,
        top_k=session.top_k,
    )
    if not successors:
        raise RuntimeError("No reachable successors.")
    candidate_features, rewards, _, _ = compact_candidate_arrays(successors)
    return successors, candidate_features, rewards


def decision_for_policy(
    session: ModelSession,
    policy,
    successors,
) -> tuple[ModelDecision, np.ndarray]:
    original = session.policy
    try:
        session.policy = policy
        q_values = session._q_values(successors)
        chosen_idx, confidence = session._choose_index(q_values)
    finally:
        session.policy = original

    chosen_idx = int(chosen_idx)
    q_values = np.asarray(q_values, dtype=np.float32)

    return (
        ModelDecision(
            chosen_rank=chosen_idx + 1,
            confidence=float(confidence),
            q_teacher=float(q_values[0]),
            q_chosen=float(q_values[chosen_idx]),
            q_margin_vs_teacher=float(
                q_values[chosen_idx] - q_values[0]
            ),
            q_best_rank=int(np.argmax(q_values)) + 1,
            q_values="|".join(f"{float(x):.9g}" for x in q_values),
        ),
        q_values,
    )


def event_from_state(
    *,
    event_id: int,
    source: str,
    session: ModelSession,
    champion,
    challenger,
    confidence_min: float,
    edges: list[float],
) -> Optional[StateEvent]:
    successors, candidate_features, _ = inspect_successors(session)

    champ, _ = decision_for_policy(session, champion, successors)
    chall, _ = decision_for_policy(session, challenger, successors)

    qualifying_confidences = []
    if (
        champ.chosen_rank != 1
        and champ.confidence >= confidence_min
    ):
        qualifying_confidences.append(champ.confidence)
    if (
        chall.chosen_rank != 1
        and chall.confidence >= confidence_min
    ):
        qualifying_confidences.append(chall.confidence)

    if not qualifying_confidences:
        return None

    anchor_conf = max(qualifying_confidences)
    state = session.state
    actions = "|".join(action_text(s.action) for s in successors)

    return StateEvent(
        event_id=event_id,
        source=source,
        seed=int(session.seed),
        piece=int(session.stats.pieces + 1),
        state_hash=state_hash(session.state_features),
        current_piece=str(state.current_piece),
        hold_piece=str(state.hold_piece or "-"),
        next4=" ".join(str(x) for x in state.next_pieces[:4]),
        before_height=int(session.stats.current_height),
        before_holes=int(session.stats.holes),
        candidate_actions=actions,
        anchor_confidence=float(anchor_conf),
        confidence_bin=confidence_bin_label(anchor_conf, edges),

        champion_chosen_rank=champ.chosen_rank,
        champion_confidence=champ.confidence,
        champion_q_teacher=champ.q_teacher,
        champion_q_chosen=champ.q_chosen,
        champion_q_margin_vs_teacher=champ.q_margin_vs_teacher,
        champion_q_best_rank=champ.q_best_rank,
        champion_q_values=champ.q_values,

        challenger_chosen_rank=chall.chosen_rank,
        challenger_confidence=chall.confidence,
        challenger_q_teacher=chall.q_teacher,
        challenger_q_chosen=chall.q_chosen,
        challenger_q_margin_vs_teacher=chall.q_margin_vs_teacher,
        challenger_q_best_rank=chall.q_best_rank,
        challenger_q_values=chall.q_values,

        models_disagree=(
            champ.chosen_rank != chall.chosen_rank
        ),
    )


def scan_source_trajectory(
    *,
    source_name: str,
    source_policy,
    champion,
    challenger,
    teacher,
    device,
    seeds: list[int],
    max_pieces: int,
    top_k: int,
    confidence_min: float,
    edges: list[float],
    start_event_id: int,
) -> tuple[list[StateEvent], int]:
    events: list[StateEvent] = []
    next_id = int(start_event_id)

    session = ModelSession(
        source_policy,
        seed=seeds[0],
        max_pieces=max_pieces,
        top_k=top_k,
        device=device,
        teacher=teacher,
    )

    try:
        for si, seed in enumerate(seeds, 1):
            session.max_pieces = max_pieces
            session.reset(seed)
            started = time.perf_counter()
            eligible = 0

            while (
                not session.done
                and session.stats.pieces < max_pieces
            ):
                successors, _, _ = inspect_successors(session)

                champ, champ_q = decision_for_policy(
                    session, champion, successors
                )
                chall, chall_q = decision_for_policy(
                    session, challenger, successors
                )

                if source_name == "champion":
                    source_decision = champ
                    source_q = champ_q
                else:
                    source_decision = chall
                    source_q = chall_q

                qualifying = []
                if (
                    champ.chosen_rank != 1
                    and champ.confidence >= confidence_min
                ):
                    qualifying.append(champ.confidence)
                if (
                    chall.chosen_rank != 1
                    and chall.confidence >= confidence_min
                ):
                    qualifying.append(chall.confidence)

                if qualifying:
                    next_id += 1
                    state = session.state
                    event = StateEvent(
                        event_id=next_id,
                        source=source_name,
                        seed=int(seed),
                        piece=int(session.stats.pieces + 1),
                        state_hash=state_hash(session.state_features),
                        current_piece=str(state.current_piece),
                        hold_piece=str(state.hold_piece or "-"),
                        next4=" ".join(
                            str(x) for x in state.next_pieces[:4]
                        ),
                        before_height=int(session.stats.current_height),
                        before_holes=int(session.stats.holes),
                        candidate_actions="|".join(
                            action_text(s.action) for s in successors
                        ),
                        anchor_confidence=float(max(qualifying)),
                        confidence_bin=confidence_bin_label(
                            max(qualifying), edges
                        ),

                        champion_chosen_rank=champ.chosen_rank,
                        champion_confidence=champ.confidence,
                        champion_q_teacher=champ.q_teacher,
                        champion_q_chosen=champ.q_chosen,
                        champion_q_margin_vs_teacher=(
                            champ.q_margin_vs_teacher
                        ),
                        champion_q_best_rank=champ.q_best_rank,
                        champion_q_values=champ.q_values,

                        challenger_chosen_rank=chall.chosen_rank,
                        challenger_confidence=chall.confidence,
                        challenger_q_teacher=chall.q_teacher,
                        challenger_q_chosen=chall.q_chosen,
                        challenger_q_margin_vs_teacher=(
                            chall.q_margin_vs_teacher
                        ),
                        challenger_q_best_rank=chall.q_best_rank,
                        challenger_q_values=chall.q_values,

                        models_disagree=(
                            champ.chosen_rank != chall.chosen_rank
                        ),
                    )
                    events.append(event)
                    eligible += 1

                source_idx = source_decision.chosen_rank - 1
                force_successor(
                    session,
                    successors[source_idx],
                    candidate_index=source_idx,
                    q_values=source_q,
                    confidence=source_decision.confidence,
                )

            print(
                f"[{source_name} {si:>2}/{len(seeds)}] seed {seed} "
                f"| P{session.stats.pieces} eligible={eligible} "
                f"| {time.perf_counter()-started:.1f}s"
            )
    finally:
        session.close()

    return events, next_id


def deduplicate_same_state(events: list[StateEvent]) -> list[StateEvent]:
    """
    Only collapse the same seed+piece+visible state reached by both sources.
    Do not deduplicate across different seeds/pieces because hidden future bag
    state is intentionally preserved by deterministic replay.
    """
    out: list[StateEvent] = []
    seen = set()
    for e in events:
        key = (e.seed, e.piece, e.state_hash)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def select_stratified_balanced(
    events: list[StateEvent],
    *,
    edges: list[float],
    events_per_bin: int,
    max_events_per_seed: int,
    sample_seed: int,
) -> list[StateEvent]:
    rng = random.Random(sample_seed)
    labels = [
        confidence_bin_label(
            (edges[i] + edges[i + 1]) / 2.0,
            edges,
        )
        for i in range(len(edges) - 1)
    ]

    per_seed: dict[int, int] = {}
    selected: list[StateEvent] = []
    selected_ids = set()

    for label in labels:
        pool = [e for e in events if e.confidence_bin == label]

        by_source = {
            "champion": [e for e in pool if e.source == "champion"],
            "challenger": [e for e in pool if e.source == "challenger"],
        }
        rng.shuffle(by_source["champion"])
        rng.shuffle(by_source["challenger"])

        picked = 0
        source_turn = ["champion", "challenger"]

        # First pass: alternate state-source trajectories where possible.
        while picked < events_per_bin:
            progress = False
            for source in source_turn:
                candidates = by_source[source]
                while candidates:
                    e = candidates.pop()
                    if e.event_id in selected_ids:
                        continue
                    if per_seed.get(e.seed, 0) >= max_events_per_seed:
                        continue

                    selected.append(e)
                    selected_ids.add(e.event_id)
                    per_seed[e.seed] = per_seed.get(e.seed, 0) + 1
                    picked += 1
                    progress = True
                    break

                if picked >= events_per_bin:
                    break

            if not progress:
                break

        # Second pass: fill any shortfall from either source.
        if picked < events_per_bin:
            remaining = [
                e for e in pool
                if e.event_id not in selected_ids
            ]
            rng.shuffle(remaining)
            for e in remaining:
                if picked >= events_per_bin:
                    break
                if per_seed.get(e.seed, 0) >= max_events_per_seed:
                    continue
                selected.append(e)
                selected_ids.add(e.event_id)
                per_seed[e.seed] = per_seed.get(e.seed, 0) + 1
                picked += 1

    return sorted(
        selected,
        key=lambda e: (e.seed, e.piece, e.source),
    )


def fast_forward(
    session: ModelSession,
    target_piece: int,
) -> None:
    target_completed = int(target_piece) - 1
    while (
        session.stats.pieces < target_completed
        and not session.done
    ):
        session.step()
        session.visual_drop = None

    if session.done:
        raise RuntimeError(
            f"Ended before seed={session.seed} piece={target_piece}: "
            f"{session.done_reason}"
        )


def replay_selected_state(
    *,
    event: StateEvent,
    source_policy,
    teacher,
    device,
    top_k: int,
    max_horizon: int,
) -> ModelSession:
    # Never use max_pieces=0 here; this remains compatible with both historical
    # and current ModelSession semantics.
    limit = max(
        10_000,
        int(event.piece + max_horizon + 100),
    )

    session = ModelSession(
        source_policy,
        seed=event.seed,
        max_pieces=limit,
        top_k=top_k,
        device=device,
        teacher=teacher,
    )
    fast_forward(session, event.piece)

    actual_hash = state_hash(session.state_features)
    if actual_hash != event.state_hash:
        session.close()
        raise RuntimeError(
            f"Matched-state replay hash mismatch for "
            f"{event.source} seed={event.seed} P{event.piece}: "
            f"expected {event.state_hash}, got {actual_hash}"
        )

    successors, _, _ = inspect_successors(session)
    actions = "|".join(action_text(s.action) for s in successors)
    if actions != event.candidate_actions:
        session.close()
        raise RuntimeError(
            f"Candidate replay mismatch for {event.source} "
            f"seed={event.seed} P{event.piece}.\n"
            f"expected: {event.candidate_actions}\n"
            f"actual  : {actions}"
        )

    return session


def run_branch(
    *,
    event: StateEvent,
    branch_rank: int,
    source_policy,
    rollout_policy,
    teacher,
    device,
    top_k: int,
    horizon: int,
) -> BranchResult:
    session = replay_selected_state(
        event=event,
        source_policy=source_policy,
        teacher=teacher,
        device=device,
        top_k=top_k,
        max_horizon=horizon,
    )

    try:
        successors, candidate_features, rewards = inspect_successors(session)
        idx = int(branch_rank) - 1
        if not 0 <= idx < len(successors):
            raise RuntimeError(
                f"Branch #{branch_rank} unavailable for event {event.event_id}."
            )

        # Re-evaluate the state with its source policy only for bookkeeping
        # during the forced branch. The continuation below is fixed.
        source_decision, source_q = decision_for_policy(
            session, source_policy, successors
        )

        successor = successors[idx]
        immediate_h, immediate_holes = candidate_board_metrics(
            candidate_features[idx]
        )

        before_lines = session.stats.lines
        before_tetrises = session.stats.tetrises
        before_value = session.stats.value

        force_successor(
            session,
            successor,
            candidate_index=idx,
            q_values=source_q,
            confidence=source_decision.confidence,
        )

        # CRITICAL CONTROL:
        # Every candidate branch now resumes under the same fixed policy.
        session.policy = rollout_policy

        max_h = int(session.stats.current_height)
        max_holes = int(session.stats.holes)

        for _ in range(int(horizon)):
            if session.done:
                break
            session.step()
            session.visual_drop = None
            max_h = max(max_h, int(session.stats.current_height))
            max_holes = max(max_holes, int(session.stats.holes))

        pieces_after = max(
            0,
            int(session.stats.pieces - event.piece),
        )

        return BranchResult(
            event_id=event.event_id,
            source=event.source,
            seed=event.seed,
            piece=event.piece,
            state_hash=event.state_hash,
            branch_rank=branch_rank,
            action=action_text(successor.action),
            teacher_top1=(branch_rank == 1),

            teacher_score=float(successor.teacher_score),
            normalized_reward=float(rewards[idx]),
            immediate_lines=int(successor.lines_cleared),
            immediate_height=int(immediate_h),
            immediate_holes=int(immediate_holes),

            horizon=int(horizon),
            survived=not bool(session.game_over),
            pieces_after_branch=int(pieces_after),
            cumulative_lines_from_branch=int(
                session.stats.lines - before_lines
            ),
            cumulative_tetrises_from_branch=int(
                session.stats.tetrises - before_tetrises
            ),
            cumulative_value_from_branch=int(
                session.stats.value - before_value
            ),
            current_height=int(session.stats.current_height),
            current_holes=int(session.stats.holes),
            max_height_from_branch=int(max_h),
            max_holes_from_branch=int(max_holes),
            done_reason=str(session.done_reason),
        )
    finally:
        session.close()


def quality(row: BranchResult) -> tuple:
    return (
        int(row.survived),
        int(row.cumulative_value_from_branch),
        int(row.cumulative_tetrises_from_branch),
        -int(row.max_height_from_branch),
        -int(row.max_holes_from_branch),
        -int(row.current_holes),
        -int(row.current_height),
    )


def classify(a: BranchResult, b: BranchResult) -> str:
    qa, qb = quality(a), quality(b)
    if qa > qb:
        return "WIN"
    if qa < qb:
        return "LOSS"
    return "TIE"


def parse_q_values(text: str) -> np.ndarray:
    return np.asarray(
        [float(x) for x in text.split("|") if x],
        dtype=np.float64,
    )


def pairwise_q_accuracy(
    q_values: np.ndarray,
    by_rank: dict[int, BranchResult],
) -> tuple[int, int]:
    correct = 0
    total = 0
    ranks = sorted(by_rank)

    for ai in range(len(ranks)):
        for bi in range(ai + 1, len(ranks)):
            ra, rb = ranks[ai], ranks[bi]
            qa = float(q_values[ra - 1])
            qb = float(q_values[rb - 1])
            real_a = quality(by_rank[ra])
            real_b = quality(by_rank[rb])

            if real_a == real_b or math.isclose(
                qa, qb, abs_tol=1e-12
            ):
                continue

            total += 1
            if (qa > qb) == (real_a > real_b):
                correct += 1

    return correct, total


def build_outcomes(
    events: list[StateEvent],
    branch_rows: list[BranchResult],
) -> list[MatchedOutcome]:
    grouped: dict[int, dict[int, BranchResult]] = {}
    for row in branch_rows:
        grouped.setdefault(row.event_id, {})[row.branch_rank] = row

    outcomes: list[MatchedOutcome] = []

    for event in events:
        by_rank = grouped[event.event_id]
        if len(by_rank) < 2:
            raise RuntimeError(
                f"Too few branches for event {event.event_id}."
            )

        best_quality = max(quality(r) for r in by_rank.values())
        best_ranks = sorted(
            rank
            for rank, row in by_rank.items()
            if quality(row) == best_quality
        )

        c_rank = event.champion_chosen_rank
        h_rank = event.challenger_chosen_rank

        c = by_rank[c_rank]
        h = by_rank[h_rank]
        teacher = by_rank[1]
        # Numeric regret must follow the same lexicographic winner set.
        # In particular, do not let a dead branch with a high pre-death value
        # define "best value" when survival wins the comparator.
        best_value = max(
            int(by_rank[rank].cumulative_value_from_branch)
            for rank in best_ranks
        )

        c_q = parse_q_values(event.champion_q_values)
        h_q = parse_q_values(event.challenger_q_values)
        c_correct, c_total = pairwise_q_accuracy(c_q, by_rank)
        h_correct, h_total = pairwise_q_accuracy(h_q, by_rank)

        outcomes.append(
            MatchedOutcome(
                event_id=event.event_id,
                source=event.source,
                seed=event.seed,
                piece=event.piece,
                state_hash=event.state_hash,
                confidence_bin=event.confidence_bin,
                models_disagree=event.models_disagree,
                best_ranks="|".join(map(str, best_ranks)),

                champion_rank=c_rank,
                challenger_rank=h_rank,

                champion_confidence=event.champion_confidence,
                challenger_confidence=event.challenger_confidence,
                champion_q_margin=(
                    event.champion_q_margin_vs_teacher
                ),
                challenger_q_margin=(
                    event.challenger_q_margin_vs_teacher
                ),

                champion_is_best=c_rank in best_ranks,
                challenger_is_best=h_rank in best_ranks,

                champion_vs_teacher=classify(c, teacher),
                challenger_vs_teacher=classify(h, teacher),
                challenger_vs_champion=classify(h, c),

                champion_value=int(c.cumulative_value_from_branch),
                challenger_value=int(h.cumulative_value_from_branch),
                teacher_value=int(
                    teacher.cumulative_value_from_branch
                ),
                best_value=int(best_value),

                champion_delta_vs_teacher=int(
                    c.cumulative_value_from_branch
                    - teacher.cumulative_value_from_branch
                ),
                challenger_delta_vs_teacher=int(
                    h.cumulative_value_from_branch
                    - teacher.cumulative_value_from_branch
                ),
                challenger_minus_champion_value=int(
                    h.cumulative_value_from_branch
                    - c.cumulative_value_from_branch
                ),

                champion_value_regret=int(
                    best_value - c.cumulative_value_from_branch
                ),
                challenger_value_regret=int(
                    best_value - h.cumulative_value_from_branch
                ),

                champion_pairwise_q_correct=c_correct,
                champion_pairwise_q_total=c_total,
                challenger_pairwise_q_correct=h_correct,
                challenger_pairwise_q_total=h_total,
            )
        )

    return outcomes


def safe_corr(xs, ys) -> Optional[float]:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if len(x) < 3:
        return None
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mean(values) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def wtl(labels: list[str]) -> tuple[int, int, int]:
    wins = sum(x == "WIN" for x in labels)
    ties = sum(x == "TIE" for x in labels)
    losses = len(labels) - wins - ties
    return wins, ties, losses


def model_metrics(
    outcomes: list[MatchedOutcome],
    *,
    which: str,
) -> dict:
    if which == "champion":
        ranks = [o.champion_rank for o in outcomes]
        best = [o.champion_is_best for o in outcomes]
        vs_teacher = [o.champion_vs_teacher for o in outcomes]
        confidence = [o.champion_confidence for o in outcomes]
        margin = [o.champion_q_margin for o in outcomes]
        delta = [o.champion_delta_vs_teacher for o in outcomes]
        regret = [o.champion_value_regret for o in outcomes]
        pair_correct = sum(
            o.champion_pairwise_q_correct for o in outcomes
        )
        pair_total = sum(
            o.champion_pairwise_q_total for o in outcomes
        )
    else:
        ranks = [o.challenger_rank for o in outcomes]
        best = [o.challenger_is_best for o in outcomes]
        vs_teacher = [o.challenger_vs_teacher for o in outcomes]
        confidence = [o.challenger_confidence for o in outcomes]
        margin = [o.challenger_q_margin for o in outcomes]
        delta = [o.challenger_delta_vs_teacher for o in outcomes]
        regret = [o.challenger_value_regret for o in outcomes]
        pair_correct = sum(
            o.challenger_pairwise_q_correct for o in outcomes
        )
        pair_total = sum(
            o.challenger_pairwise_q_total for o in outcomes
        )

    override_mask = [
        rank != 1 for rank in ranks
    ]
    override_conf = [
        c for c, use in zip(confidence, override_mask) if use
    ]
    override_margin = [
        m for m, use in zip(margin, override_mask) if use
    ]
    override_delta = [
        d for d, use in zip(delta, override_mask) if use
    ]

    wins, ties, losses = wtl(vs_teacher)

    return {
        "states": len(outcomes),
        "actual_best_count": int(sum(best)),
        "actual_best_rate": (
            float(np.mean(best)) if best else None
        ),
        "vs_teacher_wtl": [wins, ties, losses],
        "vs_teacher_win_rate": (
            wins / len(outcomes) if outcomes else None
        ),
        "mean_delta_value_vs_teacher": mean(delta),
        "mean_value_regret": mean(regret),
        "override_states": int(sum(override_mask)),
        "override_confidence_vs_delta_pearson": safe_corr(
            override_conf, override_delta
        ),
        "override_q_margin_vs_delta_pearson": safe_corr(
            override_margin, override_delta
        ),
        "pairwise_q_ordering_correct": int(pair_correct),
        "pairwise_q_ordering_total": int(pair_total),
        "pairwise_q_ordering_accuracy": (
            pair_correct / pair_total if pair_total else None
        ),
    }


def disagreement_metrics(
    outcomes: list[MatchedOutcome],
) -> dict:
    diff = [o for o in outcomes if o.models_disagree]
    labels = [o.challenger_vs_champion for o in diff]
    wins, ties, losses = wtl(labels)

    fixes = sum(
        o.challenger_is_best and not o.champion_is_best
        for o in diff
    )
    breaks = sum(
        o.champion_is_best and not o.challenger_is_best
        for o in diff
    )
    both_best = sum(
        o.champion_is_best and o.challenger_is_best
        for o in diff
    )
    neither_best = len(diff) - fixes - breaks - both_best

    return {
        "different_choice_states": len(diff),
        "challenger_vs_champion_wtl": [wins, ties, losses],
        "challenger_fix_count": int(fixes),
        "challenger_break_count": int(breaks),
        "both_best_count": int(both_best),
        "neither_best_count": int(neither_best),
        "mean_challenger_minus_champion_value": mean(
            o.challenger_minus_champion_value for o in diff
        ),
        "mean_regret_change_challenger_minus_champion": mean(
            (
                o.challenger_value_regret
                - o.champion_value_regret
            )
            for o in diff
        ),
    }


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    data = [
        asdict(x) if hasattr(x, "__dataclass_fields__") else dict(x)
        for x in rows
    ]
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(data[0].keys()),
        )
        writer.writeheader()
        writer.writerows(data)


def fmt_corr(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def write_outputs(
    *,
    output_dir: Path,
    metadata: dict,
    all_states: list[StateEvent],
    selected: list[StateEvent],
    branches: list[BranchResult],
    outcomes: list[MatchedOutcome],
    champion_metrics: dict,
    challenger_metrics: dict,
    disagreement: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "all_matched_state_candidates.csv", all_states)
    write_csv(output_dir / "selected_matched_states.csv", selected)
    write_csv(output_dir / "branch_results.csv", branches)
    write_csv(output_dir / "matched_outcomes.csv", outcomes)

    summary = {
        "metadata": metadata,
        "champion": champion_metrics,
        "challenger": challenger_metrics,
        "head_to_head": disagreement,
        "outcomes": [asdict(x) for x in outcomes],
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    c_wtl = champion_metrics["vs_teacher_wtl"]
    h_wtl = challenger_metrics["vs_teacher_wtl"]
    hh_wtl = disagreement["challenger_vs_champion_wtl"]

    lines = [
        "MATCHED-STATE CALIBRATION COMPARISON",
        "=" * 108,
        "",
        "Research/development only; NOT a promotion test.",
        (
            "Both checkpoints are evaluated on identical state243 and "
            "identical reachable top-k candidates."
        ),
        (
            "Every candidate branch resumes under ONE FIXED rollout policy, "
            "so realized branch outcomes are shared by both models."
        ),
        (
            "Corpus is confidence-stratified and source-balanced; it is NOT "
            "the natural override distribution."
        ),
        "",
        f"Seeds                : {metadata['seed_spec']}",
        f"Selected states      : {len(outcomes)}",
        f"State sources        : {metadata['state_sources']}",
        f"Rollout policy       : {metadata['rollout_label']}",
        f"Counterfactual horizon: {metadata['horizon']}",
        "",
        "31.2M CHAMPION — SAME STATES",
        "-" * 108,
        (
            f"Actual-best          : "
            f"{champion_metrics['actual_best_count']}/"
            f"{champion_metrics['states']} "
            f"({champion_metrics['actual_best_rate']*100:.1f}%)"
        ),
        f"vs Teacher W/T/L     : {c_wtl[0]}/{c_wtl[1]}/{c_wtl[2]}",
        (
            f"Mean ΔV vs Teacher   : "
            f"{champion_metrics['mean_delta_value_vs_teacher']:+.1f}"
        ),
        (
            f"Mean value regret    : "
            f"{champion_metrics['mean_value_regret']:.1f}"
        ),
        (
            f"Pairwise Q ordering  : "
            f"{champion_metrics['pairwise_q_ordering_accuracy']*100:.1f}% "
            f"({champion_metrics['pairwise_q_ordering_correct']}/"
            f"{champion_metrics['pairwise_q_ordering_total']})"
        ),
        (
            f"Corr(conf, ΔV)       : "
            f"{fmt_corr(champion_metrics['override_confidence_vs_delta_pearson'])}"
        ),
        (
            f"Corr(Q margin, ΔV)   : "
            f"{fmt_corr(champion_metrics['override_q_margin_vs_delta_pearson'])}"
        ),
        "",
        "41.2M CONTROL — SAME STATES",
        "-" * 108,
        (
            f"Actual-best          : "
            f"{challenger_metrics['actual_best_count']}/"
            f"{challenger_metrics['states']} "
            f"({challenger_metrics['actual_best_rate']*100:.1f}%)"
        ),
        f"vs Teacher W/T/L     : {h_wtl[0]}/{h_wtl[1]}/{h_wtl[2]}",
        (
            f"Mean ΔV vs Teacher   : "
            f"{challenger_metrics['mean_delta_value_vs_teacher']:+.1f}"
        ),
        (
            f"Mean value regret    : "
            f"{challenger_metrics['mean_value_regret']:.1f}"
        ),
        (
            f"Pairwise Q ordering  : "
            f"{challenger_metrics['pairwise_q_ordering_accuracy']*100:.1f}% "
            f"({challenger_metrics['pairwise_q_ordering_correct']}/"
            f"{challenger_metrics['pairwise_q_ordering_total']})"
        ),
        (
            f"Corr(conf, ΔV)       : "
            f"{fmt_corr(challenger_metrics['override_confidence_vs_delta_pearson'])}"
        ),
        (
            f"Corr(Q margin, ΔV)   : "
            f"{fmt_corr(challenger_metrics['override_q_margin_vs_delta_pearson'])}"
        ),
        "",
        "DIRECT HEAD-TO-HEAD ON STATES WHERE CHOICES DIFFER",
        "-" * 108,
        (
            f"Different choices    : "
            f"{disagreement['different_choice_states']}/{len(outcomes)}"
        ),
        (
            f"41.2M vs 31.2M W/T/L : "
            f"{hh_wtl[0]}/{hh_wtl[1]}/{hh_wtl[2]}"
        ),
        (
            f"41.2M fixes          : "
            f"{disagreement['challenger_fix_count']}"
        ),
        (
            f"41.2M breaks         : "
            f"{disagreement['challenger_break_count']}"
        ),
        (
            f"Both-best            : "
            f"{disagreement['both_best_count']}"
        ),
        (
            f"Neither-best         : "
            f"{disagreement['neither_best_count']}"
        ),
        (
            f"Mean ΔV (41-31)      : "
            f"{disagreement['mean_challenger_minus_champion_value']}"
        ),
        (
            f"Mean regret change   : "
            f"{disagreement['mean_regret_change_challenger_minus_champion']}"
        ),
        "",
        "INTERPRETATION RULE",
        "-" * 108,
        (
            "Evidence that extra training self-corrected ranking requires "
            "41.2M to improve same-state actual-best / regret / pairwise-Q "
            "ordering and to win more disagreement states than it loses."
        ),
        (
            "If whole-game performance improved but these matched-state "
            "ranking metrics do not, more same-recipe training improved play "
            "without clearly repairing ranking/calibration."
        ),
    ]

    (output_dir / "report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Matched-state calibration comparison: V8.8.6 31.2M "
            "vs same-recipe 41.2M."
        )
    )
    p.add_argument("--champion", default=DEFAULT_CHAMPION)
    p.add_argument("--challenger", default=DEFAULT_CHALLENGER)
    p.add_argument("--champion-label", default="V8.8.6 31.2M Champion")
    p.add_argument("--challenger-label", default="V8.8.6 41.2M Control")

    p.add_argument(
        "--state-sources",
        choices=["champion", "challenger", "both"],
        default="both",
        help=(
            "Trajectories used only to obtain common states. "
            "Default both reduces one-policy state-source bias."
        ),
    )
    p.add_argument(
        "--rollout-model",
        default=DEFAULT_CHAMPION,
        help=(
            "Fixed continuation policy used after every forced branch. "
            "Default is the formal 31.2M Champion."
        ),
    )
    p.add_argument(
        "--rollout-label",
        default="V8.8.6 31.2M Fixed Rollout",
    )

    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--horizon", type=int, default=250)

    p.add_argument("--confidence-min", type=float, default=0.600)
    p.add_argument(
        "--confidence-bins",
        default="0.60,0.70,0.80,0.90,0.99,1.01",
    )
    p.add_argument("--events-per-bin", type=int, default=4)
    p.add_argument("--max-events-per-seed", type=int, default=4)
    p.add_argument("--sample-seed", type=int, default=20260831)

    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument("--gate", type=float, default=0.600)
    p.add_argument(
        "--gate-semantics",
        choices=["auto", "normalized_q_margin", "raw_q_gap"],
        default="normalized_q_margin",
    )

    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--allow-protected-seeds", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    seeds = parse_seed_spec(args.seeds)
    edges = parse_float_list(args.confidence_bins)

    if args.max_pieces < 1:
        raise SystemExit("--max-pieces must be >= 1.")
    if args.horizon < 1:
        raise SystemExit("--horizon must be >= 1.")
    if not 1 <= args.top_k <= 4:
        raise SystemExit("--top-k must be 1..4.")
    if args.events_per_bin < 1:
        raise SystemExit("--events-per-bin must be >= 1.")
    if args.max_events_per_seed < 1:
        raise SystemExit("--max-events-per-seed must be >= 1.")

    protected = [s for s in seeds if s in PROTECTED_FINAL_SEEDS]
    if protected and not args.allow_protected_seeds:
        raise SystemExit(
            f"Protected final-report seeds blocked: {protected}"
        )

    device = choose_device(args.device)
    teacher = HeuristicTeacherV2()

    champion = load_policy(
        args.champion,
        label=args.champion_label,
        device=device,
        gate_override=args.gate,
        semantics_override=args.gate_semantics,
    )
    challenger = load_policy(
        args.challenger,
        label=args.challenger_label,
        device=device,
        gate_override=args.gate,
        semantics_override=args.gate_semantics,
    )
    rollout = load_policy(
        args.rollout_model,
        label=args.rollout_label,
        device=device,
        gate_override=args.gate,
        semantics_override=args.gate_semantics,
    )

    sources = []
    if args.state_sources in {"champion", "both"}:
        sources.append(("champion", champion))
    if args.state_sources in {"challenger", "both"}:
        sources.append(("challenger", challenger))

    print("=" * 108)
    print("MATCHED-STATE CALIBRATION COMPARISON")
    print("=" * 108)
    print(f"Champion     : {champion.label} | {champion.gate_short}")
    print(f"Challenger   : {challenger.label} | {challenger.gate_short}")
    print(f"State sources: {', '.join(name for name, _ in sources)}")
    print(f"Fixed rollout: {rollout.label} | {rollout.gate_short}")
    print(f"Seeds        : {args.seeds} (REUSED DEVELOPMENT-CONSUMED)")
    print(f"Piece cap    : {args.max_pieces}")
    print(f"Horizon      : {args.horizon}")
    print(f"Confidence   : >= {args.confidence_min:.2f}")
    print(f"Bins         : {edges}")
    print(f"Events/bin   : {args.events_per_bin}")
    print(f"Device       : {device}")
    print()
    print(
        "IMPORTANT: no new development or qualification seeds are consumed. "
        "This deliberately reuses 4761-4775."
    )

    started = time.perf_counter()

    print("\nPASS 1/3 — building common-state candidate corpus...")
    all_events: list[StateEvent] = []
    event_id = 0

    for source_name, source_policy in sources:
        rows, event_id = scan_source_trajectory(
            source_name=source_name,
            source_policy=source_policy,
            champion=champion,
            challenger=challenger,
            teacher=teacher,
            device=device,
            seeds=seeds,
            max_pieces=args.max_pieces,
            top_k=args.top_k,
            confidence_min=args.confidence_min,
            edges=edges,
            start_event_id=event_id,
        )
        all_events.extend(rows)

    all_events = deduplicate_same_state(all_events)

    selected = select_stratified_balanced(
        all_events,
        edges=edges,
        events_per_bin=args.events_per_bin,
        max_events_per_seed=args.max_events_per_seed,
        sample_seed=args.sample_seed,
    )

    print()
    print(f"Eligible unique states : {len(all_events)}")
    print(f"Selected matched states: {len(selected)}")
    if not selected:
        raise SystemExit("No matched states selected.")

    bin_counts = {}
    source_counts = {}
    disagree_count = 0
    for e in selected:
        bin_counts[e.confidence_bin] = (
            bin_counts.get(e.confidence_bin, 0) + 1
        )
        source_counts[e.source] = source_counts.get(e.source, 0) + 1
        disagree_count += int(e.models_disagree)

    print("Selected by bin       :", bin_counts)
    print("Selected by source    :", source_counts)
    print("Choice disagreements  :", disagree_count)

    print("\nPASS 2/3 — exact-state replay verification + fixed-policy branches...")
    branches: list[BranchResult] = []
    total_jobs = len(selected) * args.top_k
    job = 0

    source_lookup = {
        "champion": champion,
        "challenger": challenger,
    }

    for ei, event in enumerate(selected, 1):
        print(
            f"[state {ei:>2}/{len(selected)}] "
            f"{event.source} seed={event.seed} P{event.piece} "
            f"hash={event.state_hash} "
            f"31#{event.champion_chosen_rank} "
            f"41#{event.challenger_chosen_rank} "
            f"conf={event.anchor_confidence:.3f}"
        )

        source_policy = source_lookup[event.source]

        for rank in range(1, args.top_k + 1):
            job += 1
            t0 = time.perf_counter()

            row = run_branch(
                event=event,
                branch_rank=rank,
                source_policy=source_policy,
                rollout_policy=rollout,
                teacher=teacher,
                device=device,
                top_k=args.top_k,
                horizon=args.horizon,
            )
            branches.append(row)

            print(
                f"  branch#{rank} "
                f"V={row.cumulative_value_from_branch:>7} "
                f"T={row.cumulative_tetrises_from_branch:>3} "
                f"maxH={row.max_height_from_branch:>2} "
                f"holes={row.max_holes_from_branch:>2} "
                f"survive={row.survived} "
                f"| {time.perf_counter()-t0:.1f}s "
                f"[{job}/{total_jobs}]"
            )

    print("\nPASS 3/3 — matched ranking/calibration analysis...")
    outcomes = build_outcomes(selected, branches)
    champion_metrics = model_metrics(outcomes, which="champion")
    challenger_metrics = model_metrics(outcomes, which="challenger")
    disagreement = disagreement_metrics(outcomes)

    elapsed = time.perf_counter() - started
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "artifacts"
            / "matched_state_calibration"
            / f"compare_31m_41m_{seeds[0]}_{seeds[-1]}_{stamp}"
        )
    elif not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "champion": str(champion.path or args.champion),
        "challenger": str(challenger.path or args.challenger),
        "champion_label": champion.label,
        "challenger_label": challenger.label,
        "champion_gate": champion.gate,
        "challenger_gate": challenger.gate,
        "gate_semantics": args.gate_semantics,
        "state_sources": [name for name, _ in sources],
        "rollout_model": str(rollout.path or args.rollout_model),
        "rollout_label": rollout.label,
        "seed_spec": args.seeds,
        "seeds": seeds,
        "max_pieces": args.max_pieces,
        "top_k": args.top_k,
        "horizon": args.horizon,
        "confidence_min": args.confidence_min,
        "confidence_edges": edges,
        "events_per_bin": args.events_per_bin,
        "max_events_per_seed": args.max_events_per_seed,
        "sample_seed": args.sample_seed,
        "eligible_unique_states": len(all_events),
        "selected_states": len(selected),
        "elapsed_seconds": elapsed,
        "status": "RESEARCH/DEVELOPMENT ONLY - NO PROMOTION",
        "design_note": (
            "Both models see identical state243/top-k. Each top-k branch is "
            "forced from that exact state and all branches resume under one "
            "fixed rollout policy."
        ),
    }

    write_outputs(
        output_dir=output_dir,
        metadata=metadata,
        all_states=all_events,
        selected=selected,
        branches=branches,
        outcomes=outcomes,
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        disagreement=disagreement,
    )

    c_wtl = champion_metrics["vs_teacher_wtl"]
    h_wtl = challenger_metrics["vs_teacher_wtl"]
    hh_wtl = disagreement["challenger_vs_champion_wtl"]

    print()
    print("=" * 108)
    print("MATCHED-STATE CALIBRATION COMPARISON COMPLETE")
    print("=" * 108)
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Matched states: {len(outcomes)}")
    print()
    print(
        f"31.2M actual-best: "
        f"{champion_metrics['actual_best_count']}/{len(outcomes)} "
        f"({champion_metrics['actual_best_rate']*100:.1f}%)"
    )
    print(
        f"41.2M actual-best: "
        f"{challenger_metrics['actual_best_count']}/{len(outcomes)} "
        f"({challenger_metrics['actual_best_rate']*100:.1f}%)"
    )
    print(
        f"31.2M vs Teacher W/T/L: "
        f"{c_wtl[0]}/{c_wtl[1]}/{c_wtl[2]}"
    )
    print(
        f"41.2M vs Teacher W/T/L: "
        f"{h_wtl[0]}/{h_wtl[1]}/{h_wtl[2]}"
    )
    print(
        f"31.2M mean regret: "
        f"{champion_metrics['mean_value_regret']:.1f}"
    )
    print(
        f"41.2M mean regret: "
        f"{challenger_metrics['mean_value_regret']:.1f}"
    )
    print(
        f"31.2M pairwise Q ordering: "
        f"{champion_metrics['pairwise_q_ordering_accuracy']*100:.1f}%"
    )
    print(
        f"41.2M pairwise Q ordering: "
        f"{challenger_metrics['pairwise_q_ordering_accuracy']*100:.1f}%"
    )
    print(
        f"31.2M Corr(conf, ΔV): "
        f"{fmt_corr(champion_metrics['override_confidence_vs_delta_pearson'])}"
    )
    print(
        f"41.2M Corr(conf, ΔV): "
        f"{fmt_corr(challenger_metrics['override_confidence_vs_delta_pearson'])}"
    )
    print()
    print(
        f"Different-choice states: "
        f"{disagreement['different_choice_states']}/{len(outcomes)}"
    )
    print(
        f"41.2M vs 31.2M W/T/L on disagreements: "
        f"{hh_wtl[0]}/{hh_wtl[1]}/{hh_wtl[2]}"
    )
    print(
        f"41.2M fixes/breaks: "
        f"{disagreement['challenger_fix_count']}/"
        f"{disagreement['challenger_break_count']}"
    )
    print()
    print("Output:", output_dir)
    print(
        "Files: all_matched_state_candidates.csv, "
        "selected_matched_states.csv, branch_results.csv, "
        "matched_outcomes.csv, summary.json, report.txt"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
