from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
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
        "scan_override_counterfactuals.py requires tools\\watch_models.py."
    ) from exc

from singleplayer.game.executor import execute_placement
from singleplayer.network.state_encoder import encode_state

DEFAULT_MODEL = r"models\v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
DEFAULT_LABEL = "V8.8.6 31.2M"
PROTECTED_FINAL_SEEDS = set(range(6, 21))
CONSUMED_DEV_SEEDS = set(range(4601, 4621)) | set(range(4701, 4721))


@dataclass
class OverrideEvent:
    event_id: int
    seed: int
    piece: int
    confidence: float
    confidence_bin: str
    current_piece: str
    hold_piece: str
    next4: str
    chosen_rank: int
    chosen_action: str
    teacher_action: str
    q_chosen: float
    q_teacher: float
    q_margin_vs_teacher: float
    teacher_score_chosen: float
    teacher_score_top1: float
    immediate_reward_chosen: float
    immediate_reward_teacher: float
    immediate_lines_chosen: int
    immediate_lines_teacher: int
    before_height: int
    before_holes: int
    chosen_height: int
    chosen_holes: int
    teacher_height: int
    teacher_holes: int
    delta_height_vs_teacher: int
    delta_holes_vs_teacher: int
    skipped_immediate_lines: int
    risk_tags: str


@dataclass
class BranchHorizon:
    event_id: int
    seed: int
    piece: int
    confidence: float
    confidence_bin: str
    branch_rank: int
    actual_selected: bool
    teacher_top1: bool
    action: str
    q_value: float
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
    q_switches_after_branch: int
    q_switch_rate_after_branch: float
    recovered_holes_le_3_after: Optional[int]
    recovered_holes_le_1_after: Optional[int]
    done_reason: str


@dataclass
class EventOutcome:
    event_id: int
    seed: int
    piece: int
    confidence: float
    confidence_bin: str
    chosen_rank: int
    q_margin_vs_teacher: float
    risk_tags: str
    eval_horizon: int
    actual_vs_teacher: str
    actual_is_best: bool
    teacher_is_best: bool
    best_ranks: str
    actual_value: int
    teacher_value: int
    actual_minus_teacher_value: int
    actual_tetrises: int
    teacher_tetrises: int
    actual_minus_teacher_tetrises: int
    actual_max_height: int
    teacher_max_height: int
    actual_minus_teacher_max_height: int
    actual_max_holes: int
    teacher_max_holes: int
    actual_minus_teacher_max_holes: int
    actual_survived: bool
    teacher_survived: bool


def parse_seed_spec(spec: str) -> list[int]:
    out: list[int] = []
    for raw in spec.split(","):
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


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not values or min(values) < 0:
        raise ValueError("Expected comma-separated integers >= 0.")
    return values


def parse_float_list(text: str) -> list[float]:
    values = sorted({float(x.strip()) for x in text.split(",") if x.strip()})
    if len(values) < 2:
        raise ValueError("Confidence bins need at least two boundaries.")
    return values


def action_text(action) -> str:
    return f"{'H ' if bool(action.use_hold) else ''}R{int(action.rotation)} X{int(action.x)}"


def candidate_board_metrics(candidate_features: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(candidate_features, dtype=np.float32).reshape(-1)
    board = (arr[:200].reshape(20, 10) > 0.5).astype(np.uint8)
    return board_metrics(board)


def confidence_bin_label(value: float, edges: list[float]) -> str:
    v = float(value)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        if lo <= v < hi or (last and math.isclose(v, hi, abs_tol=1e-9)):
            return f"[{lo:.2f},{hi:.2f}{']' if last else ')'}"
    return f"<{edges[0]:.2f}" if v < edges[0] else f">={edges[-1]:.2f}"


def force_successor(session, successor, candidate_index, q_values, confidence):
    piece_before = str(session.state.current_piece)
    result = execute_placement(session.adapter, successor.action)
    session.state = result["state"]
    session.state_features = encode_state(session.state).astype(np.float32, copy=True)
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
        source="Teacher" if candidate_index == 0 else f"Q -> #{candidate_index + 1}",
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


def inspect_decision(session):
    successors = preview_top_k_successors(
        adapter=session.adapter, teacher=session.teacher, state=session.state, top_k=session.top_k
    )
    if not successors:
        raise RuntimeError("No reachable successors.")
    q_values = session._q_values(successors)
    chosen_index, confidence = session._choose_index(q_values)
    candidate_features, rewards, _, _ = compact_candidate_arrays(successors)
    return successors, q_values, rewards, candidate_features, chosen_index, confidence


def risk_tags_for_event(chosen_h, chosen_holes, teacher_h, teacher_holes, chosen_lines, teacher_lines):
    tags = []
    if chosen_h > teacher_h:
        tags.append("HIGHER_THAN_TEACHER")
    if chosen_holes > teacher_holes:
        tags.append("MORE_HOLES_THAN_TEACHER")
    if chosen_lines < teacher_lines:
        tags.append("SKIPPED_IMMEDIATE_CLEAR")
    if not tags:
        tags.append("NO_IMMEDIATE_RISK_SIGNAL")
    return "|".join(tags)


def scan_override_events(policy, teacher, device, seeds, max_pieces, top_k, confidence_min, confidence_edges):
    session = ModelSession(policy, seed=seeds[0], max_pieces=max_pieces, top_k=top_k, device=device, teacher=teacher)
    events = []
    event_id = 0
    try:
        for seed_idx, seed in enumerate(seeds, 1):
            session.max_pieces = max_pieces
            session.reset(seed)
            started = time.perf_counter()
            seed_count = 0
            while not session.done and session.stats.pieces < max_pieces:
                state = session.state
                before_h = int(session.stats.current_height)
                before_holes = int(session.stats.holes)
                successors, q_values, rewards, candidate_features, chosen_idx, confidence = inspect_decision(session)
                metrics = [candidate_board_metrics(candidate_features[i]) for i in range(len(successors))]
                if chosen_idx != 0 and confidence >= confidence_min:
                    event_id += 1
                    seed_count += 1
                    chosen = successors[chosen_idx]
                    teacher_top = successors[0]
                    chosen_h, chosen_holes = metrics[chosen_idx]
                    teacher_h, teacher_holes = metrics[0]
                    events.append(OverrideEvent(
                        event_id=event_id, seed=seed, piece=session.stats.pieces + 1,
                        confidence=float(confidence), confidence_bin=confidence_bin_label(confidence, confidence_edges),
                        current_piece=str(state.current_piece), hold_piece=str(state.hold_piece or "-"),
                        next4=" ".join(str(x) for x in state.next_pieces[:4]),
                        chosen_rank=chosen_idx + 1, chosen_action=action_text(chosen.action), teacher_action=action_text(teacher_top.action),
                        q_chosen=float(q_values[chosen_idx]), q_teacher=float(q_values[0]),
                        q_margin_vs_teacher=float(q_values[chosen_idx] - q_values[0]),
                        teacher_score_chosen=float(chosen.teacher_score), teacher_score_top1=float(teacher_top.teacher_score),
                        immediate_reward_chosen=float(rewards[chosen_idx]), immediate_reward_teacher=float(rewards[0]),
                        immediate_lines_chosen=int(chosen.lines_cleared), immediate_lines_teacher=int(teacher_top.lines_cleared),
                        before_height=before_h, before_holes=before_holes,
                        chosen_height=int(chosen_h), chosen_holes=int(chosen_holes),
                        teacher_height=int(teacher_h), teacher_holes=int(teacher_holes),
                        delta_height_vs_teacher=int(chosen_h - teacher_h), delta_holes_vs_teacher=int(chosen_holes - teacher_holes),
                        skipped_immediate_lines=int(teacher_top.lines_cleared - chosen.lines_cleared),
                        risk_tags=risk_tags_for_event(chosen_h, chosen_holes, teacher_h, teacher_holes, int(chosen.lines_cleared), int(teacher_top.lines_cleared)),
                    ))
                force_successor(session, successors[chosen_idx], chosen_idx, q_values, confidence)
            print(f"[baseline {seed_idx:>2}/{len(seeds)}] seed {seed} | P{session.stats.pieces} overrides>={confidence_min:.2f}: {seed_count} | {time.perf_counter()-started:.1f}s")
    finally:
        session.close()
    return events


def select_events_stratified(events, edges, events_per_bin, max_events_per_seed, sample_seed, risk_only):
    rng = random.Random(sample_seed)
    pool_events = [e for e in events if (not risk_only or e.risk_tags != "NO_IMMEDIATE_RISK_SIGNAL")]
    bins = {}
    for e in pool_events:
        bins.setdefault(e.confidence_bin, []).append(e)
    labels = [confidence_bin_label((edges[i] + edges[i+1]) / 2, edges) for i in range(len(edges)-1)]
    selected = []
    per_seed = {}
    for label in labels:
        pool = list(bins.get(label, []))
        rng.shuffle(pool)
        picked = 0
        for e in pool:
            if picked >= events_per_bin:
                break
            if per_seed.get(e.seed, 0) >= max_events_per_seed:
                continue
            selected.append(e)
            per_seed[e.seed] = per_seed.get(e.seed, 0) + 1
            picked += 1
    return sorted(selected, key=lambda e: (e.seed, e.piece))


def fast_forward(session, target_piece):
    target = target_piece - 1
    while session.stats.pieces < target and not session.done:
        session.step()
        session.visual_drop = None
    if session.done:
        raise RuntimeError(f"Ended before seed={session.seed} piece={target_piece}: {session.done_reason}")


def run_one_branch(event, branch_rank, policy, teacher, device, top_k, horizons):
    session = ModelSession(policy, seed=event.seed, max_pieces=0, top_k=top_k, device=device, teacher=teacher)
    try:
        fast_forward(session, event.piece)
        successors, q_values, rewards, candidate_features, actual_idx, confidence = inspect_decision(session)
        if actual_idx + 1 != event.chosen_rank:
            raise RuntimeError(f"Replay mismatch seed {event.seed} P{event.piece}: baseline #{event.chosen_rank}, replay #{actual_idx+1}")
        idx = branch_rank - 1
        successor = successors[idx]
        immediate_h, immediate_holes = candidate_board_metrics(candidate_features[idx])
        before_lines = session.stats.lines
        before_tetris = session.stats.tetrises
        before_value = session.stats.value
        force_successor(session, successor, idx, q_values, confidence)
        switch_base = session.stats.interventions
        max_h = session.stats.current_height
        max_holes = session.stats.holes
        recovered3 = 0 if session.stats.holes <= 3 else None
        recovered1 = 0 if session.stats.holes <= 1 else None
        horizon_set = set(horizons)
        max_horizon = max(horizons)
        rows = []
        def snap(h):
            switches = session.stats.interventions - switch_base
            return BranchHorizon(
                event_id=event.event_id, seed=event.seed, piece=event.piece, confidence=event.confidence,
                confidence_bin=event.confidence_bin, branch_rank=branch_rank,
                actual_selected=(branch_rank == event.chosen_rank), teacher_top1=(branch_rank == 1), action=action_text(successor.action),
                q_value=float(q_values[idx]), teacher_score=float(successor.teacher_score), normalized_reward=float(rewards[idx]),
                immediate_lines=int(successor.lines_cleared), immediate_height=int(immediate_h), immediate_holes=int(immediate_holes),
                horizon=h, survived=not session.game_over, pieces_after_branch=min(h, max(0, session.stats.pieces - event.piece)),
                cumulative_lines_from_branch=int(session.stats.lines - before_lines),
                cumulative_tetrises_from_branch=int(session.stats.tetrises - before_tetris),
                cumulative_value_from_branch=int(session.stats.value - before_value),
                current_height=int(session.stats.current_height), current_holes=int(session.stats.holes),
                max_height_from_branch=int(max_h), max_holes_from_branch=int(max_holes),
                q_switches_after_branch=int(switches), q_switch_rate_after_branch=(0.0 if h <= 0 else float(switches / h)),
                recovered_holes_le_3_after=recovered3, recovered_holes_le_1_after=recovered1, done_reason=str(session.done_reason),
            )
        if 0 in horizon_set:
            rows.append(snap(0))
        for step in range(1, max_horizon + 1):
            if not session.done:
                session.step(); session.visual_drop = None
                max_h = max(max_h, session.stats.current_height)
                max_holes = max(max_holes, session.stats.holes)
                if recovered3 is None and session.stats.holes <= 3: recovered3 = step
                if recovered1 is None and session.stats.holes <= 1: recovered1 = step
            if step in horizon_set:
                rows.append(snap(step))
        return rows
    finally:
        session.close()


def quality(row):
    return (
        int(row.survived), row.cumulative_value_from_branch, row.cumulative_tetrises_from_branch,
        -row.max_height_from_branch, -row.max_holes_from_branch, -row.current_holes, -row.current_height,
    )


def classify(a, b):
    return "WIN" if quality(a) > quality(b) else ("LOSS" if quality(a) < quality(b) else "TIE")


def build_outcomes(events, rows, eval_horizon):
    grouped = {}
    for r in rows:
        if r.horizon == eval_horizon:
            grouped.setdefault(r.event_id, []).append(r)
    out = []
    for e in events:
        erows = grouped[e.event_id]
        by_rank = {r.branch_rank: r for r in erows}
        actual, teacher = by_rank[e.chosen_rank], by_rank[1]
        bestq = max(quality(r) for r in erows)
        best_ranks = sorted(r.branch_rank for r in erows if quality(r) == bestq)
        out.append(EventOutcome(
            event_id=e.event_id, seed=e.seed, piece=e.piece, confidence=e.confidence, confidence_bin=e.confidence_bin,
            chosen_rank=e.chosen_rank, q_margin_vs_teacher=e.q_margin_vs_teacher, risk_tags=e.risk_tags,
            eval_horizon=eval_horizon, actual_vs_teacher=classify(actual, teacher), actual_is_best=e.chosen_rank in best_ranks,
            teacher_is_best=1 in best_ranks, best_ranks="|".join(map(str, best_ranks)),
            actual_value=actual.cumulative_value_from_branch, teacher_value=teacher.cumulative_value_from_branch,
            actual_minus_teacher_value=actual.cumulative_value_from_branch - teacher.cumulative_value_from_branch,
            actual_tetrises=actual.cumulative_tetrises_from_branch, teacher_tetrises=teacher.cumulative_tetrises_from_branch,
            actual_minus_teacher_tetrises=actual.cumulative_tetrises_from_branch - teacher.cumulative_tetrises_from_branch,
            actual_max_height=actual.max_height_from_branch, teacher_max_height=teacher.max_height_from_branch,
            actual_minus_teacher_max_height=actual.max_height_from_branch - teacher.max_height_from_branch,
            actual_max_holes=actual.max_holes_from_branch, teacher_max_holes=teacher.max_holes_from_branch,
            actual_minus_teacher_max_holes=actual.max_holes_from_branch - teacher.max_holes_from_branch,
            actual_survived=actual.survived, teacher_survived=teacher.survived,
        ))
    return out


def safe_corr(xs, ys):
    if len(xs) < 3: return None
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12: return None
    return float(np.corrcoef(x, y)[0,1])


def aggregate_bins(outcomes):
    grouped = {}
    for o in outcomes: grouped.setdefault(o.confidence_bin, []).append(o)
    rows = []
    for label, rr in sorted(grouped.items()):
        n = len(rr); wins = sum(x.actual_vs_teacher == "WIN" for x in rr); ties = sum(x.actual_vs_teacher == "TIE" for x in rr); losses = n-wins-ties
        rows.append({
            "confidence_bin": label, "events": n, "actual_vs_teacher_wins": wins, "actual_vs_teacher_ties": ties,
            "actual_vs_teacher_losses": losses, "actual_vs_teacher_win_rate": wins/n,
            "actual_best_count": sum(x.actual_is_best for x in rr), "actual_best_rate": sum(x.actual_is_best for x in rr)/n,
            "mean_actual_minus_teacher_value": float(np.mean([x.actual_minus_teacher_value for x in rr])),
            "mean_actual_minus_teacher_tetrises": float(np.mean([x.actual_minus_teacher_tetrises for x in rr])),
            "mean_actual_minus_teacher_max_height": float(np.mean([x.actual_minus_teacher_max_height for x in rr])),
            "mean_actual_minus_teacher_max_holes": float(np.mean([x.actual_minus_teacher_max_holes for x in rr])),
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    data = [asdict(x) if hasattr(x, "__dataclass_fields__") else dict(x) for x in rows]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)


def write_outputs(output_dir, metadata, all_events, selected, branch_rows, outcomes, bin_rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_override_events.csv", all_events)
    write_csv(output_dir / "selected_events.csv", selected)
    write_csv(output_dir / "branch_horizons.csv", branch_rows)
    write_csv(output_dir / "event_outcomes.csv", outcomes)
    write_csv(output_dir / "confidence_bins.csv", bin_rows)
    wins = sum(o.actual_vs_teacher == "WIN" for o in outcomes); ties = sum(o.actual_vs_teacher == "TIE" for o in outcomes); losses = len(outcomes)-wins-ties
    actual_best = sum(o.actual_is_best for o in outcomes); teacher_best = sum(o.teacher_is_best for o in outcomes)
    cc = safe_corr([o.confidence for o in outcomes], [o.actual_minus_teacher_value for o in outcomes])
    qc = safe_corr([o.q_margin_vs_teacher for o in outcomes], [o.actual_minus_teacher_value for o in outcomes])
    summary = {"metadata": metadata, "totals": {
        "all_overrides_found": len(all_events), "counterfactual_events": len(outcomes),
        "actual_vs_teacher_wins": wins, "actual_vs_teacher_ties": ties, "actual_vs_teacher_losses": losses,
        "actual_vs_teacher_win_rate": wins/len(outcomes) if outcomes else None,
        "actual_is_best_count": actual_best, "actual_is_best_rate": actual_best/len(outcomes) if outcomes else None,
        "teacher_is_best_count": teacher_best, "teacher_is_best_rate": teacher_best/len(outcomes) if outcomes else None,
        "confidence_vs_realized_value_delta_pearson": cc, "q_margin_vs_realized_value_delta_pearson": qc,
    }, "confidence_bins": bin_rows, "outcomes": [asdict(o) for o in outcomes]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["CHAMPION Q-OVERRIDE COUNTERFACTUAL SCAN", "="*100, "", "Research only; not a promotion test.",
             "cumulative_* metrics INCLUDE the forced branch placement.", "",
             f"All qualifying overrides : {len(all_events)}", f"Sampled events          : {len(outcomes)}",
             f"Actual vs Teacher W/T/L : {wins}/{ties}/{losses}", f"Actual best among top-k : {actual_best}/{len(outcomes)}" if outcomes else "Actual best among top-k : n/a",
             f"Corr(confidence, ΔV)    : {cc:+.4f}" if cc is not None else "Corr(confidence, ΔV)    : n/a",
             f"Corr(Q margin, ΔV)      : {qc:+.4f}" if qc is not None else "Corr(Q margin, ΔV)      : n/a", "", "CONFIDENCE BINS", "-"*100]
    for r in bin_rows:
        lines.append(f"{r['confidence_bin']}: n={r['events']} W/T/L={r['actual_vs_teacher_wins']}/{r['actual_vs_teacher_ties']}/{r['actual_vs_teacher_losses']} actual-best={r['actual_best_rate']:.3f} meanΔV={r['mean_actual_minus_teacher_value']:+.1f} meanΔT={r['mean_actual_minus_teacher_tetrises']:+.2f} meanΔmaxH={r['mean_actual_minus_teacher_max_height']:+.2f} meanΔmaxHoles={r['mean_actual_minus_teacher_max_holes']:+.2f}")
    lines += ["", "EVENT OUTCOMES", "-"*100]
    for o in outcomes:
        lines.append(f"E{o.event_id:04d} seed={o.seed} P{o.piece} conf={o.confidence:.3f} {o.confidence_bin} chosen#{o.chosen_rank} vsTeacher={o.actual_vs_teacher} best={o.best_ranks} ΔV={o.actual_minus_teacher_value:+d} ΔT={o.actual_minus_teacher_tetrises:+d} ΔmaxH={o.actual_minus_teacher_max_height:+d} ΔmaxHoles={o.actual_minus_teacher_max_holes:+d} | {o.risk_tags}")
    (output_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(description="Mass counterfactual validation of V8.8.6 Q overrides on fresh development seeds.")
    p.add_argument("model", nargs="?", default=DEFAULT_MODEL); p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument("--seeds", default="4721-4725"); p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--confidence-min", type=float, default=0.60)
    p.add_argument("--confidence-bins", default="0.60,0.70,0.80,0.90,0.99,1.01")
    p.add_argument("--events-per-bin", type=int, default=2); p.add_argument("--max-events-per-seed", type=int, default=3)
    p.add_argument("--sample-seed", type=int, default=20260829); p.add_argument("--risk-only", action="store_true")
    p.add_argument("--horizons", default="0,20,50,100,250"); p.add_argument("--eval-horizon", type=int, default=250)
    p.add_argument("--top-k", type=int, default=4); p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--gate", type=float, default=None); p.add_argument("--gate-semantics", choices=["auto","normalized_q_margin","raw_q_gap"], default="auto")
    p.add_argument("--output-dir", type=Path, default=None); p.add_argument("--allow-protected-seeds", action="store_true"); p.add_argument("--allow-consumed-dev-seeds", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    seeds = parse_seed_spec(args.seeds); horizons = parse_int_list(args.horizons); edges = parse_float_list(args.confidence_bins)
    if args.eval_horizon not in horizons: raise SystemExit("--eval-horizon must also appear in --horizons.")
    if args.max_pieces < 1: raise SystemExit("--max-pieces must be >= 1.")
    protected = [s for s in seeds if s in PROTECTED_FINAL_SEEDS]
    if protected and not args.allow_protected_seeds: raise SystemExit(f"Protected final-report seeds blocked: {protected}")
    consumed = [s for s in seeds if s in CONSUMED_DEV_SEEDS]
    if consumed and not args.allow_consumed_dev_seeds: raise SystemExit(f"Previously consumed development seeds blocked: {consumed}")
    device = choose_device(args.device); policy = load_policy(args.model, label=args.label, device=device, gate_override=args.gate, semantics_override=args.gate_semantics); teacher = HeuristicTeacherV2()
    print("="*108); print("CHAMPION Q-OVERRIDE COUNTERFACTUAL SCAN"); print("="*108)
    print(f"Model: {policy.label} | {policy.gate_short} | Device: {device}")
    print(f"Seeds: {args.seeds} | cap={args.max_pieces} | confidence>={args.confidence_min:.2f}")
    print(f"Bins: {edges} | events/bin={args.events_per_bin} | horizons={horizons} | eval={args.eval_horizon}")
    started = time.perf_counter()
    print("\nPASS 1/2 — finding real Champion Q overrides...")
    all_events = scan_override_events(policy, teacher, device, seeds, args.max_pieces, args.top_k, args.confidence_min, edges)
    selected = select_events_stratified(all_events, edges, args.events_per_bin, args.max_events_per_seed, args.sample_seed, args.risk_only)
    print(f"\nQualifying overrides found: {len(all_events)}\nSelected for rollout: {len(selected)}")
    if not selected: raise SystemExit("No events selected.")
    print("\nPASS 2/2 — counterfactual top-k rollouts...")
    branch_rows = []; total_jobs = len(selected) * args.top_k; job = 0
    for ei, e in enumerate(selected, 1):
        print(f"[event {ei}/{len(selected)}] seed {e.seed} P{e.piece} conf={e.confidence:.3f} actual#{e.chosen_rank} {e.risk_tags}")
        for rank in range(1, args.top_k+1):
            job += 1; t0 = time.perf_counter(); rows = run_one_branch(e, rank, policy, teacher, device, args.top_k, horizons); branch_rows.extend(rows)
            final = next(r for r in rows if r.horizon == args.eval_horizon)
            print(f"  branch#{rank} {'ACTUAL' if rank==e.chosen_rank else '      '} V={final.cumulative_value_from_branch:>7} T={final.cumulative_tetrises_from_branch:>3} maxH={final.max_height_from_branch:>2} maxHoles={final.max_holes_from_branch:>2} survive={final.survived} | {time.perf_counter()-t0:.1f}s [{job}/{total_jobs}]")
    outcomes = build_outcomes(selected, branch_rows, args.eval_horizon); bin_rows = aggregate_bins(outcomes)
    elapsed = time.perf_counter()-started; stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (PROJECT_ROOT / "artifacts" / "override_counterfactual_scans" / f"scan_{seeds[0]}_{seeds[-1]}_{stamp}")
    if not output_dir.is_absolute(): output_dir = PROJECT_ROOT / output_dir
    metadata = {"created_at": datetime.now().isoformat(timespec="seconds"), "model": str(policy.path or args.model), "label": policy.label, "gate": policy.gate, "gate_semantics": policy.gate_semantics, "seed_spec": args.seeds, "seeds": seeds, "max_pieces": args.max_pieces, "confidence_min": args.confidence_min, "confidence_edges": edges, "events_per_bin": args.events_per_bin, "max_events_per_seed": args.max_events_per_seed, "sample_seed": args.sample_seed, "risk_only": args.risk_only, "horizons": horizons, "eval_horizon": args.eval_horizon, "elapsed_seconds": elapsed, "branch_metric_note": "cumulative_* includes the forced branch placement itself", "comparator_note": "research-only lexicographic: survival, cumulative value, cumulative Tetris, lower max height, lower max holes, lower current holes, lower current height"}
    write_outputs(output_dir, metadata, all_events, selected, branch_rows, outcomes, bin_rows)
    wins=sum(o.actual_vs_teacher=="WIN" for o in outcomes); ties=sum(o.actual_vs_teacher=="TIE" for o in outcomes); losses=len(outcomes)-wins-ties; best=sum(o.actual_is_best for o in outcomes)
    print("\n"+"="*108); print("OVERRIDE COUNTERFACTUAL SCAN COMPLETE"); print("="*108)
    print(f"Elapsed: {elapsed:.2f}s\nAll overrides: {len(all_events)}\nCounterfactual events: {len(outcomes)}\nActual vs Teacher W/T/L: {wins}/{ties}/{losses}\nActual best among top-k: {best}/{len(outcomes)} ({best/len(outcomes)*100:.1f}%)\nOutput: {output_dir}")
    print("Files: all_override_events.csv, selected_events.csv, branch_horizons.csv, event_outcomes.csv, confidence_bins.csv, summary.json, report.txt")
    print("="*108)


if __name__ == "__main__":
    main()
