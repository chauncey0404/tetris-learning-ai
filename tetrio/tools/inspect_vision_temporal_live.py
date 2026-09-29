from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2

from tetrio.vision.board import draw_board_overlay, read_visual_board
from tetrio.vision.image_io import imwrite_unicode
from tetrio.vision.layout import (
    OcrObservation,
    PlayfieldCandidate,
    RapidUsernameReader,
    ResolvedPlayfield,
    detect_playfields,
    draw_layout_overlay,
    resolve_playfield_roles,
)
from tetrio.vision.piece_reader import (
    draw_piece_preview_overlay,
    read_piece_previews,
)
from tetrio.vision.screen_capture import (
    CaptureRegion,
    WindowsGdiScreenCapture,
    clamp_capture_region,
)
from tetrio.vision.state_tracker import TemporalStateTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live, vision-only Phase-B gate for the TETR.IO-like simulator. "
            "This tool never sends keyboard input."
        )
    )
    p.add_argument("--self-username", default="MAYSHOWGUNMORE77")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--seconds", type=float, default=90.0)
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--heartbeat-seconds", type=float, default=1.0)
    p.add_argument("--min-ready-frames", type=int, default=8)
    p.add_argument("--min-lock-events", type=int, default=5)
    p.add_argument("--min-preview-shifts", type=int, default=5)
    p.add_argument(
        "--require-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one temporally-confirmed HOLD change for PASS.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\temporal_live"),
    )
    return p.parse_args()


def _draw_temporal_text(image, obs):
    out = image.copy()
    active = "-" if obs.active is None else (
        f"{obs.active.piece} c={obs.active.confidence:.2f}"
        + (" REC" if obs.active.recovered else "")
    )
    preview = "-" if obs.preview_queue is None else "".join(obs.preview_queue)
    lines = [
        f"TEMP {obs.phase} ready={int(obs.stable_pre_action)} {obs.reason}",
        f"active={active} hold={obs.hold_piece or '-'} next={preview}",
        f"spawn={int(obs.spawn_event)} lock={int(obs.lock_event)} "
        f"qshift={int(obs.preview_shift_event)} holdchg={int(obs.hold_changed)}",
    ]
    y = 30
    for line in lines:
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += 26
    return out


def _resolve_self(image, *, self_username, no_ocr, rapid):
    candidates = detect_playfields(image)
    if not candidates:
        return None, []

    # Single-board simulator modes are unambiguous.  Avoid paying the OCR cost
    # just to have resolve_playfield_roles later ignore it anyway.
    if len(candidates) == 1:
        configured = (
            self_username.strip()
            if isinstance(self_username, str) and self_username.strip()
            else None
        )
        item = ResolvedPlayfield(
            candidate=candidates[0],
            role="SELF",
            username=configured,
            username_ocr_confidence=0.0,
            self_match_score=1.0 if configured else 0.0,
        )
        return item, [item]

    if no_ocr:
        def reader(_image, _candidate):
            return OcrObservation(None, 0.0)
    else:
        assert rapid is not None
        reader = rapid.read

    resolved = resolve_playfield_roles(
        image,
        candidates,
        self_username=self_username,
        ocr_reader=reader,
    )
    self_boards = [item for item in resolved if item.role == "SELF"]
    if len(self_boards) != 1:
        return None, resolved
    return self_boards[0], resolved


def tracking_region_for_candidate(
    candidate: PlayfieldCandidate,
    screen_width: int,
    screen_height: int,
) -> CaptureRegion:
    """ROI containing SELF board plus board-relative HOLD and NEXT panels."""
    bx, by, bw, bh = candidate.bbox
    x1 = int(round(bx - 0.70 * bw))
    x2 = int(round(bx + 1.62 * bw))
    y1 = int(round(by - 0.04 * bh))
    y2 = int(round(by + 1.03 * bh))
    return CaptureRegion(
        x=max(0, x1),
        y=max(0, y1),
        width=max(1, min(screen_width, x2) - max(0, x1)),
        height=max(1, min(screen_height, y2) - max(0, y1)),
    )


def localize_candidate(
    candidate: PlayfieldCandidate,
    region: CaptureRegion,
) -> PlayfieldCandidate:
    return PlayfieldCandidate(
        x=float(candidate.x - region.x),
        y=float(candidate.y - region.y),
        w=float(candidate.w),
        h=float(candidate.h),
        cell_size=float(candidate.cell_size),
        grid_score=float(candidate.grid_score),
        matched_vertical_lines=int(candidate.matched_vertical_lines),
        confidence=float(candidate.confidence),
    )


def _local_resolved(item: ResolvedPlayfield, region: CaptureRegion) -> ResolvedPlayfield:
    return ResolvedPlayfield(
        candidate=localize_candidate(item.candidate, region),
        role=item.role,
        username=item.username,
        username_ocr_confidence=float(item.username_ocr_confidence),
        self_match_score=float(item.self_match_score),
    )


def _gate_passed(args, counters: Counter) -> bool:
    if counters["ready_frames"] < args.min_ready_frames:
        return False
    if counters["lock_events"] < args.min_lock_events:
        return False
    if counters["preview_shift_events"] < args.min_preview_shifts:
        return False
    if args.require_hold and counters["hold_changes"] < 1:
        return False
    return True


def _summary(
    args,
    capture,
    counters,
    reasons,
    timings,
    started,
    frames,
    stopped_reason,
    region,
    layout_acquire_seconds,
):
    elapsed = max(0.0, time.perf_counter() - started)
    passed = _gate_passed(args, counters)
    processed = max(1, int(counters["processed_frames"]))
    return {
        "format": "tetrio_visual_temporal_live_gate_v3",
        "status": "PASS" if passed else "FAIL",
        "stopped_reason": stopped_reason,
        "elapsed_seconds": elapsed,
        "frames": int(frames),
        "effective_fps": 0.0 if elapsed <= 0 else frames / elapsed,
        "screen": capture.geometry.to_dict(),
        "tracking_region": region.to_dict(),
        "layout_acquire_seconds": float(layout_acquire_seconds),
        "criteria": {
            "min_ready_frames": args.min_ready_frames,
            "min_lock_events": args.min_lock_events,
            "min_preview_shifts": args.min_preview_shifts,
            "require_hold": bool(args.require_hold),
        },
        "counters": dict(counters),
        "reasons": dict(reasons),
        "mean_stage_ms": {
            key: float(value / processed)
            for key, value in timings.items()
        },
    }


def _acquire_layout(capture, *, self_username, no_ocr, rapid, timeout=30.0):
    started = time.perf_counter()
    attempts = 0
    while time.perf_counter() - started < timeout:
        image = capture.grab()
        attempts += 1
        item, resolved = _resolve_self(
            image,
            self_username=self_username,
            no_ocr=no_ocr,
            rapid=rapid,
        )
        if item is not None:
            return item, resolved, time.perf_counter() - started, attempts
        if attempts == 1 or attempts % 3 == 0:
            print("WAIT: SELF board unresolved during one-time layout acquisition", flush=True)
        time.sleep(0.25)
    return None, [], time.perf_counter() - started, attempts


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.seconds <= 0:
        raise SystemExit("--seconds must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = args.output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    rapid = None if args.no_ocr else RapidUsernameReader()
    tracker = TemporalStateTracker(TrackerConfig())

    print("=" * 112)
    print("TETR.IO VISUAL TEMPORAL TRACKER — LIVE PHASE-B GATE V3")
    print("=" * 112)
    print(f"Target FPS     : {args.fps:.1f}")
    print(f"Tracking time  : {args.seconds:.0f}s after layout acquisition")
    print(
        "PASS gate      : "
        f"READY>={args.min_ready_frames}, LOCK>={args.min_lock_events}, "
        f"NEXT shift>={args.min_preview_shifts}, "
        f"HOLD change={'yes' if args.require_hold else 'no'}"
    )
    print("Input          : VISION ONLY; keyboard output is disabled")
    print("Layout policy  : detect SELF once, then lock the board-relative ROI")
    print()
    for remain in range(max(0, args.countdown), 0, -1):
        print(f"Starting in {remain}...", flush=True)
        time.sleep(1.0)

    with WindowsGdiScreenCapture() as capture:
        print(f"Virtual screen : {capture.geometry}")
        print("Acquiring SELF layout once...", flush=True)
        item, _resolved_full, acquire_seconds, attempts = _acquire_layout(
            capture,
            self_username=args.self_username,
            no_ocr=args.no_ocr,
            rapid=rapid,
        )
        if item is None:
            raise SystemExit(
                "SELF board could not be resolved within 30 seconds. "
                "Keep the simulator fully visible and do not use --no-ocr in multi-board mode."
            )

        region = tracking_region_for_candidate(
            item.candidate,
            capture.geometry.width,
            capture.geometry.height,
        )
        region = clamp_capture_region(region, capture.geometry)
        local_item = _local_resolved(item, region)
        local_resolved = [local_item]

        print(
            f"Layout locked   : bbox={item.candidate.bbox} "
            f"acquire={acquire_seconds:.2f}s attempts={attempts}"
        )
        print(f"Tracking ROI   : {region}")
        print("Now play normally and use HOLD at least once.")
        print()

        interval = 1.0 / args.fps
        started = time.perf_counter()
        next_tick = started
        last_heartbeat = -1e9
        counters: Counter = Counter()
        reasons: Counter = Counter()
        timings: Counter = Counter()
        frame_rows = []
        frame_index = 0
        stopped_reason = "timeout"
        last_signature = None

        try:
            while True:
                loop_start = time.perf_counter()
                if loop_start - started >= args.seconds:
                    stopped_reason = "timeout"
                    break

                t0 = time.perf_counter()
                image = capture.grab_region(region)
                t1 = time.perf_counter()
                board = read_visual_board(image, local_item.candidate)
                t2 = time.perf_counter()
                previews = read_piece_previews(image, local_item.candidate)
                t3 = time.perf_counter()
                tracked = tracker.update(
                    board,
                    previews,
                    # Fixed local bbox: layout detector jitter can no longer
                    # reset temporal history during a stationary live gate.
                    layout_bbox=local_item.candidate.bbox,
                )
                t4 = time.perf_counter()

                frame_index += 1
                counters["captured_frames"] += 1
                counters["processed_frames"] += 1
                timings["capture"] += (t1 - t0) * 1000.0
                timings["board"] += (t2 - t1) * 1000.0
                timings["pieces"] += (t3 - t2) * 1000.0
                timings["tracker"] += (t4 - t3) * 1000.0
                timings["processing"] += (t4 - t0) * 1000.0

                reasons[tracked.reason] += 1
                if tracked.stable_pre_action:
                    counters["ready_frames"] += 1
                if tracked.spawn_event:
                    counters["spawn_events"] += 1
                if tracked.lock_event:
                    counters["lock_events"] += 1
                if tracked.preview_shift_event:
                    counters["preview_shift_events"] += 1
                if tracked.hold_changed:
                    counters["hold_changes"] += 1
                if tracked.active is not None and tracked.active.recovered:
                    counters["active_recovered_frames"] += 1

                active = "-" if tracked.active is None else tracked.active.piece
                signature = (
                    tracked.phase,
                    tracked.reason,
                    tracked.stable_pre_action,
                    active,
                    tracked.hold_piece,
                    tracked.preview_queue,
                    tracked.spawn_event,
                    tracked.lock_event,
                    tracked.preview_shift_event,
                    tracked.hold_changed,
                )
                now = time.perf_counter()
                important = (
                    signature != last_signature
                    or tracked.spawn_event
                    or tracked.lock_event
                    or tracked.preview_shift_event
                    or tracked.hold_changed
                )
                heartbeat = now - last_heartbeat >= args.heartbeat_seconds
                if important or heartbeat:
                    processed = max(1, counters["processed_frames"])
                    mean_processing = timings["processing"] / processed
                    observed_fps = frame_index / max(now - started, 1e-9)
                    print(
                        f"[{frame_index:05d}] {tracked.phase:10s} "
                        f"ready={int(tracked.stable_pre_action)} "
                        f"active={active} hold={tracked.hold_piece or '-'} "
                        f"next={tracked.preview_queue} "
                        f"spawn={int(tracked.spawn_event)} "
                        f"lock={int(tracked.lock_event)} "
                        f"shift={int(tracked.preview_shift_event)} "
                        f"holdchg={int(tracked.hold_changed)} "
                        f"transient={tracked.transient_count} "
                        f"reason={tracked.reason} "
                        f"fps={observed_fps:.1f} proc={mean_processing:.1f}ms",
                        flush=True,
                    )
                    last_heartbeat = now
                last_signature = signature

                frame_rows.append({
                    "frame": frame_index,
                    "t": now - started,
                    "layout_bbox": list(local_item.candidate.bbox),
                    "tracked": tracked.to_dict(),
                    "stage_ms": {
                        "capture": (t1 - t0) * 1000.0,
                        "board": (t2 - t1) * 1000.0,
                        "pieces": (t3 - t2) * 1000.0,
                        "tracker": (t4 - t3) * 1000.0,
                        "processing": (t4 - t0) * 1000.0,
                    },
                })

                save_snapshot = (
                    tracked.spawn_event
                    or tracked.lock_event
                    or tracked.preview_shift_event
                    or tracked.hold_changed
                    or (tracked.stable_pre_action and counters["ready_frames"] == 1)
                )
                if save_snapshot:
                    overlay = draw_layout_overlay(image, local_resolved)
                    overlay = draw_board_overlay(overlay, local_item.candidate, board)
                    overlay = draw_piece_preview_overlay(overlay, previews)
                    overlay = _draw_temporal_text(overlay, tracked)
                    out = snapshots / f"frame_{frame_index:05d}_{tracked.phase.lower()}.png"
                    imwrite_unicode(out, overlay)

                if _gate_passed(args, counters):
                    stopped_reason = "gate_passed"
                    break

                next_tick += interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -interval * 3:
                    next_tick = time.perf_counter()

        except KeyboardInterrupt:
            stopped_reason = "ctrl_c"

        summary = _summary(
            args,
            capture,
            counters,
            reasons,
            timings,
            started,
            frame_index,
            stopped_reason,
            region,
            acquire_seconds,
        )
        (args.output_dir / "frames.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frame_rows),
            encoding="utf-8",
        )
        (args.output_dir / "live_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print()
    print("=" * 112)
    print(f"LIVE TEMPORAL GATE: {summary['status']}")
    print("=" * 112)
    print(
        f"READY={counters['ready_frames']}  "
        f"LOCK={counters['lock_events']}  "
        f"SPAWN={counters['spawn_events']}  "
        f"NEXT_SHIFT={counters['preview_shift_events']}  "
        f"HOLD_CHANGE={counters['hold_changes']}"
    )
    print(
        f"Frames={frame_index}  effective_fps={summary['effective_fps']:.2f}  "
        f"stop={stopped_reason}"
    )
    ms = summary["mean_stage_ms"]
    print(
        "Mean stage ms  : "
        f"capture={ms.get('capture', 0.0):.1f} "
        f"board={ms.get('board', 0.0):.1f} "
        f"pieces={ms.get('pieces', 0.0):.1f} "
        f"tracker={ms.get('tracker', 0.0):.1f} "
        f"processing={ms.get('processing', 0.0):.1f}"
    )
    print(f"Report: {args.output_dir / 'live_summary.json'}")
    print(f"Trace : {args.output_dir / 'frames.jsonl'}")

    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
