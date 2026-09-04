from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tools.watch_models import (
        HeuristicTeacherV2,
        ModelSession,
        choose_device,
        load_policy,
    )
except ImportError as exc:
    raise RuntimeError(
        "scan_champion_failures.py requires tools\\watch_models.py.\n"
        "Place this file at:\n"
        "  F:\\tetris-learning-ai\\tools\\scan_champion_failures.py\n"
        "and run it from the project root."
    ) from exc


DEFAULT_MODEL = r"models\v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt"
DEFAULT_LABEL = "V8.8.6 31.2M"

PROTECTED_FINAL_SEEDS = set(range(6, 21))
ALREADY_CONSUMED_DEV_SEEDS = set(range(4601, 4621))


@dataclass
class DangerEvent:
    seed: int
    piece: int
    event: str
    severity: int

    height_before: int
    height_after: int
    holes_before: int
    holes_after: int

    lines_total: int
    tetrises_total: int
    value_total: int

    current_piece: str
    action: str
    policy_source: str
    chosen_rank: int
    confidence: float
    lines_cleared: int

    note: str


@dataclass
class SeedSummary:
    seed: int
    pieces: int
    lines: int
    tetrises: int
    value: int

    avg_height: float
    max_height: int
    max_holes: int

    q_switches: int
    switch_rate: float

    game_over: bool
    stop_reason: str

    danger_events: int
    severe_events: int

    max_consecutive_high_height: int
    max_consecutive_holes: int
    longest_recovery_from_holes: int

    first_height_14_piece: Optional[int]
    first_height_16_piece: Optional[int]
    first_holes_5_piece: Optional[int]
    first_holes_8_piece: Optional[int]
    game_over_piece: Optional[int]

    worst_event_piece: Optional[int]
    worst_event_severity: int
    tags: list[str]


def parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []

    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue

        if "-" in part:
            left, right = part.split("-", 1)
            a = int(left.strip())
            b = int(right.strip())
            if b < a:
                a, b = b, a
            seeds.extend(range(a, b + 1))
        else:
            seeds.append(int(part))

    if not seeds:
        raise ValueError("No seeds parsed.")

    return list(dict.fromkeys(seeds))


def severity_for(
    *,
    event: str,
    height_before: int,
    height_after: int,
    holes_before: int,
    holes_after: int,
) -> int:
    if event == "GAME_OVER":
        return 100

    score = 0

    if event == "HOLES_JUMP":
        score += 12 * max(0, holes_after - holes_before)
    elif event == "HEIGHT_JUMP":
        score += 8 * max(0, height_after - height_before)
    elif event == "HIGH_HEIGHT":
        score += max(0, height_after - 12) * 8
    elif event == "HIGH_HOLES":
        score += max(0, holes_after - 3) * 10
    elif event == "Q_INTERVENTION_DANGER":
        score += 25
    elif event == "RECOVERY_FAILURE":
        score += 30

    score += max(0, height_after - 14) * 4
    score += max(0, holes_after - 5) * 5

    return int(score)


def make_event(
    *,
    seed: int,
    session: ModelSession,
    event: str,
    height_before: int,
    holes_before: int,
    note: str,
) -> DangerEvent:
    severity = severity_for(
        event=event,
        height_before=height_before,
        height_after=session.stats.current_height,
        holes_before=holes_before,
        holes_after=session.stats.holes,
    )

    return DangerEvent(
        seed=int(seed),
        piece=int(session.stats.pieces),
        event=event,
        severity=severity,

        height_before=int(height_before),
        height_after=int(session.stats.current_height),
        holes_before=int(holes_before),
        holes_after=int(session.stats.holes),

        lines_total=int(session.stats.lines),
        tetrises_total=int(session.stats.tetrises),
        value_total=int(session.stats.value),

        current_piece=str(session.last.piece),
        action=str(session.last.action_text),
        policy_source=str(session.last.source),
        chosen_rank=int(session.last.chosen_index + 1),
        confidence=float(session.last.confidence),
        lines_cleared=int(session.last.lines_cleared),

        note=str(note),
    )


def seed_interest_key(row: SeedSummary) -> tuple:
    return (
        int(row.game_over),
        row.worst_event_severity,
        row.max_holes,
        row.max_height,
        row.severe_events,
        row.longest_recovery_from_holes,
        row.max_consecutive_high_height,
        row.max_consecutive_holes,
    )


def write_outputs(
    output_dir: Path,
    *,
    metadata: dict,
    summaries: list[SeedSummary],
    events: list[DangerEvent],
    top_n: int,
    model_path: str,
    model_label: str,
    pre_roll: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "summaries": [asdict(s) for s in summaries],
        "events": [asdict(e) for e in events],
    }

    (output_dir / "scan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_fields = [
        "seed",
        "pieces",
        "lines",
        "tetrises",
        "value",
        "avg_height",
        "max_height",
        "max_holes",
        "q_switches",
        "switch_rate",
        "game_over",
        "stop_reason",
        "danger_events",
        "severe_events",
        "max_consecutive_high_height",
        "max_consecutive_holes",
        "longest_recovery_from_holes",
        "first_height_14_piece",
        "first_height_16_piece",
        "first_holes_5_piece",
        "first_holes_8_piece",
        "game_over_piece",
        "worst_event_piece",
        "worst_event_severity",
        "tags",
    ]

    with (output_dir / "summary.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            d = asdict(row)
            d["tags"] = "|".join(row.tags)
            writer.writerow(d)

    event_fields = [
        "seed",
        "piece",
        "event",
        "severity",
        "height_before",
        "height_after",
        "holes_before",
        "holes_after",
        "lines_total",
        "tetrises_total",
        "value_total",
        "current_piece",
        "action",
        "policy_source",
        "chosen_rank",
        "confidence",
        "lines_cleared",
        "note",
    ]

    with (output_dir / "events.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=event_fields)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))

    ranked = sorted(
        summaries,
        key=seed_interest_key,
        reverse=True,
    )

    lines = [
        "V8.8.6 CHAMPION FAILURE MINING — INTERESTING SEEDS",
        "=" * 92,
        "Ranking is for investigation only; it is not a promotion score.",
        "",
    ]

    for rank, row in enumerate(ranked[:top_n], 1):
        lines.append(
            f"#{rank:02d} Seed {row.seed} — "
            f"pieces={row.pieces}, "
            f"maxH={row.max_height}, "
            f"maxHoles={row.max_holes}, "
            f"worstSeverity={row.worst_event_severity}"
        )
        lines.append(
            f"    gameOver={row.game_over} "
            f"stop={row.stop_reason} "
            f"dangerEvents={row.danger_events} "
            f"severe={row.severe_events}"
        )
        lines.append(
            f"    longest high-height={row.max_consecutive_high_height} "
            f"pieces, holes-streak={row.max_consecutive_holes}, "
            f"hole-recovery={row.longest_recovery_from_holes}"
        )
        lines.append(
            f"    first H14={row.first_height_14_piece}, "
            f"H16={row.first_height_16_piece}, "
            f"holes5={row.first_holes_5_piece}, "
            f"holes8={row.first_holes_8_piece}, "
            f"GO={row.game_over_piece}"
        )
        lines.append(
            f"    worst event piece={row.worst_event_piece}"
        )
        lines.append(
            f"    tags: {', '.join(row.tags)}"
        )
        lines.append("")

    (output_dir / "interesting_seeds.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # Highest severity event per seed.
    by_seed: dict[int, list[DangerEvent]] = {}
    for event in events:
        by_seed.setdefault(event.seed, []).append(event)

    cmd_lines = [
        "@echo off",
        "REM Auto-generated by scan_champion_failures.py",
        "REM Run from F:\\tetris-learning-ai",
        "",
    ]

    for row in ranked[:top_n]:
        seed_events = by_seed.get(row.seed, [])
        if seed_events:
            target = max(
                seed_events,
                key=lambda e: (e.severity, e.piece),
            )
            start_piece = max(1, target.piece - pre_roll)
        elif row.game_over_piece is not None:
            start_piece = max(1, row.game_over_piece - pre_roll)
        else:
            start_piece = 1

        cmd_lines.append(
            f'echo Seed {row.seed} - start piece {start_piece}'
        )
        cmd_lines.append(
            f'.venv\\Scripts\\python.exe -m tools.watch_models '
            f'"{model_path}" '
            f'--labels "{model_label}" '
            f'--seed {row.seed} '
            f'--start-piece {start_piece}'
        )
        cmd_lines.append("")

    (output_dir / "replay_hotspots.cmd").write_text(
        "\n".join(cmd_lines),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Mine V8.8.6 Champion failure/risk events across fresh "
            "development seeds using the same policy path as watch_models."
        )
    )

    p.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
    )
    p.add_argument(
        "--label",
        default=DEFAULT_LABEL,
    )
    p.add_argument(
        "--seeds",
        default="4701-4720",
        help="Default: fresh development block 4701-4720.",
    )
    p.add_argument(
        "--max-pieces",
        type=int,
        default=5000,
        help="Default 5000. Use 0 for unlimited until top-out.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
    )
    p.add_argument(
        "--gate",
        type=float,
        default=None,
    )
    p.add_argument(
        "--gate-semantics",
        choices=[
            "auto",
            "normalized_q_margin",
            "raw_q_gap",
        ],
        default="auto",
    )

    p.add_argument(
        "--height-jump",
        type=int,
        default=3,
        help="Record HEIGHT_JUMP when height increases by at least this much.",
    )
    p.add_argument(
        "--holes-jump",
        type=int,
        default=2,
        help="Record HOLES_JUMP when holes increase by at least this much.",
    )
    p.add_argument(
        "--high-height",
        type=int,
        default=14,
        help="Danger-height threshold. Default 14.",
    )
    p.add_argument(
        "--severe-height",
        type=int,
        default=16,
        help="Severe-height threshold. Default 16.",
    )
    p.add_argument(
        "--high-holes",
        type=int,
        default=5,
        help="Danger-hole threshold. Default 5.",
    )
    p.add_argument(
        "--severe-holes",
        type=int,
        default=8,
        help="Severe-hole threshold. Default 8.",
    )
    p.add_argument(
        "--recovery-window",
        type=int,
        default=30,
        help=(
            "If holes remain at/above --high-holes for this many pieces, "
            "record RECOVERY_FAILURE. Default 30."
        ),
    )
    p.add_argument(
        "--q-danger-window",
        type=int,
        default=6,
        help=(
            "After a Q intervention, if height rises >=3 or holes rise >=2 "
            "within this many pieces, record Q_INTERVENTION_DANGER. "
            "This is correlation only, not causal attribution."
        ),
    )

    p.add_argument(
        "--top-n",
        type=int,
        default=20,
    )
    p.add_argument(
        "--pre-roll",
        type=int,
        default=30,
        help=(
            "Replay command starts this many pieces before the hotspot. "
            "Default 30."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--allow-protected-seeds",
        action="store_true",
    )
    p.add_argument(
        "--allow-consumed-dev-seeds",
        action="store_true",
        help=(
            "Allow 4601-4620, which were already consumed by the "
            "V8.8-vs-V8.8.6 divergence scan."
        ),
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    seeds = parse_seed_spec(args.seeds)

    protected = [s for s in seeds if s in PROTECTED_FINAL_SEEDS]
    if protected and not args.allow_protected_seeds:
        raise SystemExit(
            f"Protected final seeds blocked: {protected}"
        )

    consumed = [
        s for s in seeds
        if s in ALREADY_CONSUMED_DEV_SEEDS
    ]
    if consumed and not args.allow_consumed_dev_seeds:
        raise SystemExit(
            "Seeds 4601-4620 were already consumed by the previous "
            "development divergence scan.\n"
            f"Requested consumed seeds: {consumed}\n"
            "Use fresh seeds (default 4701-4720)."
        )

    if args.max_pieces < 0:
        raise SystemExit("--max-pieces must be >= 0.")

    if not 1 <= args.top_k <= 4:
        raise SystemExit("--top-k must be 1..4.")

    if args.height_jump < 1 or args.holes_jump < 1:
        raise SystemExit("Jump thresholds must be >= 1.")

    if args.recovery_window < 1:
        raise SystemExit("--recovery-window must be >= 1.")

    device = choose_device(args.device)

    policy = load_policy(
        args.model,
        label=args.label,
        device=device,
        gate_override=args.gate,
        semantics_override=args.gate_semantics,
    )

    model_path = (
        str(policy.path.relative_to(PROJECT_ROOT))
        if policy.path is not None
        and PROJECT_ROOT in policy.path.parents
        else str(policy.path or args.model)
    )

    print("=" * 108)
    print("V8.8.6 CHAMPION FAILURE MINING")
    print("=" * 108)
    print(f"Device       : {device}")
    print(f"Model        : {policy.label}")
    print(f"Policy       : {policy.gate_short}")
    print(f"Seeds        : {args.seeds} ({len(seeds)})")
    print(
        f"Cap          : "
        f"{'UNLIMITED' if args.max_pieces == 0 else args.max_pieces}"
    )
    print(
        f"Thresholds   : Hjump>={args.height_jump}, "
        f"holesJump>={args.holes_jump}, "
        f"highH>={args.high_height}, "
        f"severeH>={args.severe_height}, "
        f"highHoles>={args.high_holes}, "
        f"severeHoles>={args.severe_holes}"
    )
    print()

    teacher = HeuristicTeacherV2()
    session = ModelSession(
        policy,
        seed=seeds[0],
        max_pieces=args.max_pieces,
        top_k=args.top_k,
        device=device,
        teacher=teacher,
    )

    all_events: list[DangerEvent] = []
    summaries: list[SeedSummary] = []

    scan_started = time.perf_counter()

    try:
        for seed_idx, seed in enumerate(seeds, 1):
            seed_started = time.perf_counter()
            session.max_pieces = args.max_pieces
            session.reset(seed)

            seed_events: list[DangerEvent] = []

            consecutive_high_height = 0
            max_consecutive_high_height = 0

            consecutive_holes = 0
            max_consecutive_holes = 0

            hole_episode_start: Optional[int] = None
            longest_recovery = 0
            recovery_failure_recorded_for_episode = False

            first_h14 = None
            first_h16 = None
            first_holes5 = None
            first_holes8 = None

            # Track recent Q interventions to identify correlated deterioration.
            recent_q_interventions: list[dict] = []

            while not session.done:
                h_before = session.stats.current_height
                holes_before = session.stats.holes

                session.step()

                piece = session.stats.pieces
                h_after = session.stats.current_height
                holes_after = session.stats.holes

                if piece <= 0:
                    break

                # First threshold crossings.
                if first_h14 is None and h_after >= args.high_height:
                    first_h14 = piece

                if first_h16 is None and h_after >= args.severe_height:
                    first_h16 = piece

                if first_holes5 is None and holes_after >= args.high_holes:
                    first_holes5 = piece

                if first_holes8 is None and holes_after >= args.severe_holes:
                    first_holes8 = piece

                # Per-step jumps.
                if h_after - h_before >= args.height_jump:
                    seed_events.append(
                        make_event(
                            seed=seed,
                            session=session,
                            event="HEIGHT_JUMP",
                            height_before=h_before,
                            holes_before=holes_before,
                            note=(
                                f"height +{h_after - h_before} "
                                f"({h_before}->{h_after})"
                            ),
                        )
                    )

                if holes_after - holes_before >= args.holes_jump:
                    seed_events.append(
                        make_event(
                            seed=seed,
                            session=session,
                            event="HOLES_JUMP",
                            height_before=h_before,
                            holes_before=holes_before,
                            note=(
                                f"holes +{holes_after - holes_before} "
                                f"({holes_before}->{holes_after})"
                            ),
                        )
                    )

                # Entering danger zones.
                if (
                    h_after >= args.high_height
                    and h_before < args.high_height
                ):
                    seed_events.append(
                        make_event(
                            seed=seed,
                            session=session,
                            event="HIGH_HEIGHT",
                            height_before=h_before,
                            holes_before=holes_before,
                            note=f"entered height danger zone >= {args.high_height}",
                        )
                    )

                if (
                    holes_after >= args.high_holes
                    and holes_before < args.high_holes
                ):
                    seed_events.append(
                        make_event(
                            seed=seed,
                            session=session,
                            event="HIGH_HOLES",
                            height_before=h_before,
                            holes_before=holes_before,
                            note=f"entered holes danger zone >= {args.high_holes}",
                        )
                    )

                # Sustained high-height streak.
                if h_after >= args.high_height:
                    consecutive_high_height += 1
                else:
                    consecutive_high_height = 0

                max_consecutive_high_height = max(
                    max_consecutive_high_height,
                    consecutive_high_height,
                )

                # Sustained holes / recovery timing.
                if holes_after >= args.high_holes:
                    consecutive_holes += 1

                    if hole_episode_start is None:
                        hole_episode_start = piece
                        recovery_failure_recorded_for_episode = False

                    episode_len = piece - hole_episode_start + 1

                    if (
                        episode_len >= args.recovery_window
                        and not recovery_failure_recorded_for_episode
                    ):
                        seed_events.append(
                            make_event(
                                seed=seed,
                                session=session,
                                event="RECOVERY_FAILURE",
                                height_before=h_before,
                                holes_before=holes_before,
                                note=(
                                    f"holes stayed >= {args.high_holes} "
                                    f"for {episode_len} pieces"
                                ),
                            )
                        )
                        recovery_failure_recorded_for_episode = True

                else:
                    if hole_episode_start is not None:
                        recovered_after = piece - hole_episode_start
                        longest_recovery = max(
                            longest_recovery,
                            recovered_after,
                        )

                    hole_episode_start = None
                    recovery_failure_recorded_for_episode = False
                    consecutive_holes = 0

                max_consecutive_holes = max(
                    max_consecutive_holes,
                    consecutive_holes,
                )

                # Q intervention correlation tracking.
                if session.last.chosen_index != 0:
                    recent_q_interventions.append({
                        "piece": piece,
                        "height": h_before,
                        "holes": holes_before,
                        "reported": False,
                    })

                still_recent = []
                for item in recent_q_interventions:
                    age = piece - item["piece"]

                    if age <= args.q_danger_window:
                        if (
                            not item["reported"]
                            and (
                                h_after - item["height"] >= 3
                                or holes_after - item["holes"] >= 2
                            )
                        ):
                            seed_events.append(
                                make_event(
                                    seed=seed,
                                    session=session,
                                    event="Q_INTERVENTION_DANGER",
                                    height_before=item["height"],
                                    holes_before=item["holes"],
                                    note=(
                                        "risk increased within "
                                        f"{age} pieces after Q intervention "
                                        f"at piece {item['piece']}; "
                                        "correlation only"
                                    ),
                                )
                            )
                            item["reported"] = True

                        still_recent.append(item)

                recent_q_interventions = still_recent

                if session.game_over:
                    seed_events.append(
                        make_event(
                            seed=seed,
                            session=session,
                            event="GAME_OVER",
                            height_before=h_before,
                            holes_before=holes_before,
                            note="environment terminated with game over/top-out",
                        )
                    )
                    break

            if hole_episode_start is not None:
                longest_recovery = max(
                    longest_recovery,
                    session.stats.pieces - hole_episode_start + 1,
                )

            severe_events = sum(
                1 for e in seed_events
                if e.severity >= 30
            )

            worst = (
                max(
                    seed_events,
                    key=lambda e: (e.severity, e.piece),
                )
                if seed_events
                else None
            )

            tags: list[str] = []

            if session.game_over:
                tags.append("GAME_OVER")

            if session.stats.max_height >= args.severe_height:
                tags.append("SEVERE_HEIGHT")

            if session.stats.max_holes >= args.severe_holes:
                tags.append("SEVERE_HOLES")

            if any(
                e.event == "RECOVERY_FAILURE"
                for e in seed_events
            ):
                tags.append("HOLE_RECOVERY_FAILURE")

            if any(
                e.event == "Q_INTERVENTION_DANGER"
                for e in seed_events
            ):
                tags.append("Q_INTERVENTION_RISK_CORRELATION")

            if not tags:
                tags.append("STABLE")

            stop_reason = session.done_reason or (
                "GAME OVER" if session.game_over else "UNKNOWN"
            )

            summary = SeedSummary(
                seed=int(seed),
                pieces=int(session.stats.pieces),
                lines=int(session.stats.lines),
                tetrises=int(session.stats.tetrises),
                value=int(session.stats.value),

                avg_height=float(session.stats.avg_height),
                max_height=int(session.stats.max_height),
                max_holes=int(session.stats.max_holes),

                q_switches=int(session.stats.interventions),
                switch_rate=float(session.stats.switch_rate),

                game_over=bool(session.game_over),
                stop_reason=str(stop_reason),

                danger_events=len(seed_events),
                severe_events=severe_events,

                max_consecutive_high_height=max_consecutive_high_height,
                max_consecutive_holes=max_consecutive_holes,
                longest_recovery_from_holes=longest_recovery,

                first_height_14_piece=first_h14,
                first_height_16_piece=first_h16,
                first_holes_5_piece=first_holes5,
                first_holes_8_piece=first_holes8,
                game_over_piece=(
                    session.stats.pieces
                    if session.game_over
                    else None
                ),

                worst_event_piece=(
                    worst.piece if worst else None
                ),
                worst_event_severity=(
                    worst.severity if worst else 0
                ),
                tags=tags,
            )

            summaries.append(summary)
            all_events.extend(seed_events)

            elapsed = time.perf_counter() - seed_started
            total_elapsed = time.perf_counter() - scan_started
            avg = total_elapsed / seed_idx
            eta = avg * (len(seeds) - seed_idx)

            print(
                f"[{seed_idx:>3}/{len(seeds)}] seed {seed} | "
                f"P{summary.pieces} "
                f"L{summary.lines} "
                f"T{summary.tetrises} "
                f"V{summary.value} "
                f"avgH{summary.avg_height:.2f} "
                f"maxH{summary.max_height} "
                f"maxHoles{summary.max_holes} "
                f"switch{summary.switch_rate * 100:.1f}% "
                f"events{summary.danger_events} "
                f"severe{summary.severe_events} "
                f"{'GO' if summary.game_over else summary.stop_reason} "
                f"| {elapsed:.2f}s | ETA {eta/60:.1f}m"
            )

    except KeyboardInterrupt:
        print()
        print("Interrupted. Saving completed seeds...")

    finally:
        try:
            session.close()
        except Exception:
            pass

    elapsed_total = time.perf_counter() - scan_started

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "artifacts"
            / "champion_failure_scans"
            / f"scan_{seeds[0]}_{seeds[-1]}_{stamp}"
        )
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "model": model_path,
        "label": policy.label,
        "gate": policy.gate,
        "gate_semantics": policy.gate_semantics,
        "env_steps": policy.env_steps,
        "seed_spec": args.seeds,
        "seeds_requested": seeds,
        "seeds_completed": [s.seed for s in summaries],
        "max_pieces": args.max_pieces,
        "thresholds": {
            "height_jump": args.height_jump,
            "holes_jump": args.holes_jump,
            "high_height": args.high_height,
            "severe_height": args.severe_height,
            "high_holes": args.high_holes,
            "severe_holes": args.severe_holes,
            "recovery_window": args.recovery_window,
            "q_danger_window": args.q_danger_window,
        },
        "elapsed_seconds": elapsed_total,
    }

    write_outputs(
        output_dir,
        metadata=metadata,
        summaries=summaries,
        events=all_events,
        top_n=args.top_n,
        model_path=model_path,
        model_label=policy.label,
        pre_roll=args.pre_roll,
    )

    ranked = sorted(
        summaries,
        key=seed_interest_key,
        reverse=True,
    )

    print()
    print("=" * 108)
    print("CHAMPION FAILURE MINING COMPLETE")
    print("=" * 108)
    print(f"Completed : {len(summaries)}/{len(seeds)} seeds")
    print(f"Elapsed   : {elapsed_total:.2f}s")
    print(f"Output    : {output_dir}")
    print("Files     : scan.json, summary.csv, events.csv, interesting_seeds.txt, replay_hotspots.cmd")

    if ranked:
        top = ranked[0]
        print()
        print("Top investigation candidate:")
        print(
            f"  Seed {top.seed} | "
            f"pieces={top.pieces} "
            f"maxH={top.max_height} "
            f"maxHoles={top.max_holes} "
            f"worstSeverity={top.worst_event_severity} "
            f"worstPiece={top.worst_event_piece}"
        )
        print(
            f"  tags={','.join(top.tags)}"
        )

        start_piece = max(
            1,
            (top.worst_event_piece or 1) - args.pre_roll,
        )

        print()
        print("Replay hotspot:")
        print(
            f'.venv\\Scripts\\python.exe -m tools.watch_models '
            f'"{model_path}" '
            f'--labels "{policy.label}" '
            f'--seed {top.seed} '
            f'--start-piece {start_piece}'
        )

    print("=" * 108)


if __name__ == "__main__":
    main()
