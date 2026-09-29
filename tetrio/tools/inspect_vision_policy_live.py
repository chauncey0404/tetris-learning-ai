from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2

from tetrio.live_v1_1 import LiveV11Policy, LiveV11PolicyConfig
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
from tetrio.vision.observation import ObservationNotReady, build_model_observation
from tetrio.vision.piece_reader import draw_piece_preview_overlay, read_piece_previews
from tetrio.vision.screen_capture import (
    CaptureRegion,
    WindowsGdiScreenCapture,
    clamp_capture_region,
)
from tetrio.vision.state_tracker import TemporalStateTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Vision -> Temporal Tracker -> Expert-v1.1 500K dry-run. "
            "No keyboard input is ever sent."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_1_future_500k.pt"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--self-username", default="MAYSHOWGUNMORE77")
    p.add_argument("--no-ocr", action="store_true")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--seconds", type=float, default=180.0)
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--decisions", type=int, default=3)
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\policy_live"),
    )
    return p.parse_args()


def _resolve_self(image, *, self_username, no_ocr, rapid):
    candidates = detect_playfields(image)
    if not candidates:
        return None
    if len(candidates) == 1:
        configured = (
            self_username.strip()
            if isinstance(self_username, str) and self_username.strip()
            else None
        )
        return ResolvedPlayfield(
            candidate=candidates[0],
            role="SELF",
            username=configured,
            username_ocr_confidence=0.0,
            self_match_score=1.0 if configured else 0.0,
        )

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
    own = [item for item in resolved if item.role == "SELF"]
    return own[0] if len(own) == 1 else None


def _acquire_layout(capture, *, self_username, no_ocr, rapid, timeout=30.0):
    started = time.perf_counter()
    attempts = 0
    while time.perf_counter() - started < timeout:
        image = capture.grab()
        attempts += 1
        item = _resolve_self(
            image,
            self_username=self_username,
            no_ocr=no_ocr,
            rapid=rapid,
        )
        if item is not None:
            return item, time.perf_counter() - started, attempts
        if attempts == 1 or attempts % 3 == 0:
            print("WAIT: SELF board unresolved during layout acquisition", flush=True)
        time.sleep(0.25)
    return None, time.perf_counter() - started, attempts


def tracking_region_for_candidate(
    candidate: PlayfieldCandidate,
    screen_width: int,
    screen_height: int,
) -> CaptureRegion:
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


def localize_candidate(candidate: PlayfieldCandidate, region: CaptureRegion) -> PlayfieldCandidate:
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


def _draw_decision(image, observation, decision):
    out = image.copy()
    chosen = decision.chosen
    path = ",".join(decision.movement_path)
    if len(path) > 90:
        path = path[:87] + "..."
    lines = [
        f"POLICY DRY RUN frame={observation.frame_index} fp={observation.fingerprint}",
        f"state active={observation.active_piece} hold={observation.hold_piece or '-'} next={''.join(observation.preview_queue)}",
        f"chosen hold={int(chosen.use_hold)} {chosen.branch_mode} piece={chosen.state.piece} r={int(chosen.state.rotation)%4} x={chosen.state.x} y={chosen.state.y}",
        f"score base={chosen.base_score:.3f} residual={chosen.residual:+.3f} final={chosen.final_score:.3f} candidates={len(decision.candidates)} shortlist={len(decision.shortlist_indices)}",
        f"path={path}",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (20, 20, 20), 1, cv2.LINE_AA)
        y += 23
    return out


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.seconds <= 0 or args.decisions <= 0:
        raise SystemExit("--fps, --seconds and --decisions must be > 0")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = args.output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("TETR.IO LIVE VISION -> EXPERT V1.1 500K — POLICY DRY RUN")
    print("=" * 118)
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Device         : {args.device}")
    print(f"Target FPS     : {args.fps:.1f}")
    print(f"Decision gate  : {args.decisions} unique stable pre-action snapshots")
    print("Keyboard       : DISABLED (this tool never sends input)")
    print("Simulator truth: DISABLED (vision is the only state source)")
    print("Loading model...")

    policy = LiveV11Policy(
        args.checkpoint,
        device=args.device,
        config=LiveV11PolicyConfig(
            backend=args.backend,
            fast_max_states=args.fast_max_states,
            reference_max_states=args.reference_max_states,
            top_overall=args.top_overall,
            top_per_branch=args.top_per_branch,
        ),
    )
    print(
        f"Model loaded   : format={policy.checkpoint.get('format')} "
        f"epoch={policy.checkpoint.get('epoch')}"
    )
    print()

    for remain in range(max(0, args.countdown), 0, -1):
        print(f"Starting in {remain}...", flush=True)
        time.sleep(1.0)

    rapid = None if args.no_ocr else RapidUsernameReader()
    tracker = TemporalStateTracker(TrackerConfig())
    decisions = []
    seen: set[str] = set()
    frame_index = 0
    stop_reason = "timeout"

    with WindowsGdiScreenCapture() as capture:
        print(f"Virtual screen : {capture.geometry}")
        print("Acquiring SELF layout once...", flush=True)
        item, acquire_seconds, attempts = _acquire_layout(
            capture,
            self_username=args.self_username,
            no_ocr=args.no_ocr,
            rapid=rapid,
        )
        if item is None:
            raise SystemExit("SELF board could not be resolved within 30 seconds")

        region = clamp_capture_region(
            tracking_region_for_candidate(
                item.candidate,
                capture.geometry.width,
                capture.geometry.height,
            ),
            capture.geometry,
        )
        local_item = _local_resolved(item, region)
        print(
            f"Layout locked   : bbox={item.candidate.bbox} "
            f"acquire={acquire_seconds:.2f}s attempts={attempts}"
        )
        print(f"Tracking ROI   : {region}")
        print("Play manually. Each new READY state is scored once; no key will be sent.")
        print()

        started = time.perf_counter()
        next_tick = started
        interval = 1.0 / args.fps

        while time.perf_counter() - started < args.seconds:
            image = capture.grab_region(region)
            board = read_visual_board(image, local_item.candidate)
            previews = read_piece_previews(image, local_item.candidate)
            tracked = tracker.update(
                board,
                previews,
                layout_bbox=local_item.candidate.bbox,
            )
            frame_index += 1

            if tracked.stable_pre_action:
                try:
                    observation = build_model_observation(tracked)
                except ObservationNotReady as exc:
                    print(f"[{frame_index:05d}] READY rejected by adapter: {exc}")
                    observation = None

                if observation is not None and observation.fingerprint not in seen:
                    seen.add(observation.fingerprint)
                    ordinal = len(decisions) + 1
                    print(
                        f"[{frame_index:05d}] DECISION {ordinal}/{args.decisions} "
                        f"active={observation.active_piece} "
                        f"hold={observation.hold_piece or '-'} "
                        f"next={observation.preview_queue} fp={observation.fingerprint}",
                        flush=True,
                    )
                    try:
                        decision = policy.decide(observation)
                    except Exception as exc:
                        report = {
                            "format": "tetrio_live_v1_1_policy_dry_run",
                            "status": "FAIL",
                            "failure": f"{type(exc).__name__}:{exc}",
                            "frame": frame_index,
                            "observation": observation.to_dict(),
                            "decisions": decisions,
                        }
                        (args.output_dir / "policy_live_summary.json").write_text(
                            json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        raise

                    row = {
                        "frame": frame_index,
                        "observation": observation.to_dict(),
                        "decision": decision.to_dict(),
                    }
                    decisions.append(row)
                    chosen = decision.chosen
                    print(
                        "  CHOSEN "
                        f"hold={int(chosen.use_hold)} mode={chosen.branch_mode} "
                        f"piece={chosen.state.piece} r={int(chosen.state.rotation)%4} "
                        f"x={chosen.state.x} y={chosen.state.y} lines={chosen.lines} "
                        f"base={chosen.base_score:.3f} residual={chosen.residual:+.3f} "
                        f"final={chosen.final_score:.3f}"
                    )
                    print(
                        f"  candidates={len(decision.candidates)} "
                        f"shortlist={len(decision.shortlist_indices)} "
                        f"path={decision.movement_path} "
                        f"total={decision.timings_ms['total']:.1f}ms",
                        flush=True,
                    )

                    overlay = draw_layout_overlay(image, [local_item])
                    overlay = draw_board_overlay(overlay, local_item.candidate, board)
                    overlay = draw_piece_preview_overlay(overlay, previews)
                    overlay = _draw_decision(overlay, observation, decision)
                    imwrite_unicode(
                        snapshots / f"decision_{ordinal:02d}_frame_{frame_index:05d}.png",
                        overlay,
                    )

                    if len(decisions) >= args.decisions:
                        stop_reason = "decision_gate_passed"
                        break
                    print("  Continue playing manually for the next unique state.\n")

            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -interval * 3:
                # Neural/future inference can intentionally take longer than a
                # frame. Restart cadence rather than attempting catch-up bursts.
                next_tick = time.perf_counter()

    passed = len(decisions) >= args.decisions
    summary = {
        "format": "tetrio_live_v1_1_policy_dry_run",
        "status": "PASS" if passed else "FAIL",
        "stop_reason": stop_reason,
        "checkpoint": str(args.checkpoint),
        "checkpoint_format": policy.checkpoint.get("format"),
        "checkpoint_epoch": policy.checkpoint.get("epoch"),
        "device": str(policy.device),
        "frames": frame_index,
        "unique_decisions": len(decisions),
        "required_decisions": args.decisions,
        "decisions": decisions,
    }
    (args.output_dir / "policy_live_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print(f"LIVE V1.1 POLICY DRY RUN: {summary['status']}")
    print("=" * 118)
    print(
        f"Decisions={len(decisions)}/{args.decisions} frames={frame_index} "
        f"stop={stop_reason}"
    )
    print(f"Report: {args.output_dir / 'policy_live_summary.json'}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
