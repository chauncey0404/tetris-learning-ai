from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from tetrio.live_pipeline import (
    build_speculative_preparation,
    decide_prepared,
    decisions_equivalent,
    make_speculation_seed,
    observation_generation_key,
    synthetic_completed_observation,
    template_match_reason,
    warmup_live_policy,
)
from tetrio.live_v1_1 import LiveV11Policy, LiveV11PolicyConfig
from tetrio.tools.inspect_vision_policy_live import (
    _acquire_layout,
    _local_resolved,
    tracking_region_for_candidate,
)
from tetrio.vision.board import read_visual_board
from tetrio.vision.observation import ObservationNotReady, build_model_observation
from tetrio.vision.layout import RapidUsernameReader
from tetrio.vision.piece_reader import read_piece_previews
from tetrio.vision.retarget import (
    RetargetResult,
    build_retarget_request,
    retarget_request_stale_reason,
    run_retarget_request,
)
from tetrio.vision.screen_capture import WindowsGdiScreenCapture, clamp_capture_region
from tetrio.vision.state_tracker import TemporalStateTracker, TrackerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Phase C1.5 v3 gate: generation-deduped asynchronous vision/planner, speculative next-state "
            "CPU preparation, and target-directed current-state retarget. Keyboard disabled."
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
    p.add_argument("--decisions", type=int, default=4)
    p.add_argument("--retarget-probe-frames", type=int, default=4)
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument("--max-full-foreground-ms", type=float, default=350.0)
    p.add_argument("--max-prepared-foreground-ms", type=float, default=50.0)
    p.add_argument("--max-spec-prep-ms", type=float, default=1500.0)
    p.add_argument("--max-retarget-ms", type=float, default=350.0)
    p.add_argument("--max-retarget-wall-ms", type=float, default=350.0)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"artifacts\tetrio\vision\pipeline_c1_5"),
    )
    return p.parse_args()


def _fmt_timings(timings: dict[str, float]) -> str:
    return " ".join(f"{k}={float(v):.1f}ms" for k, v in timings.items())


def _submit_plan(planner_pool, policy, observation, prepared):
    mode = "FULL"
    cache_reason = None
    used_prepared = None
    if prepared is not None:
        cache_reason = template_match_reason(observation, prepared.template)
        if cache_reason is None:
            mode = "PREPARED"
            used_prepared = prepared
            future = planner_pool.submit(
                decide_prepared,
                policy,
                observation,
                prepared,
                resolve_exact_path=False,
            )
        else:
            future = planner_pool.submit(
                policy.decide,
                observation,
                resolve_exact_path=False,
            )
    else:
        future = planner_pool.submit(
            policy.decide,
            observation,
            resolve_exact_path=False,
        )
    return future, {
        "observation": observation,
        "mode": mode,
        "cache_reason": cache_reason,
        "prepared": used_prepared,
        "submitted_at": time.perf_counter(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fps <= 0 or args.seconds <= 0:
        raise SystemExit("--fps and --seconds must be > 0")

    config = LiveV11PolicyConfig(
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        top_overall=args.top_overall,
        top_per_branch=args.top_per_branch,
    )

    print("=" * 122)
    print("TETR.IO PHASE C1.5 V3 — GENERATION DEDUP + ASYNC SPECULATION + ASYNC TARGETED RETARGET")
    print("=" * 122)
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Device         : {args.device}")
    print(f"Target FPS     : {args.fps:.1f}")
    print(f"Decision gate  : {args.decisions}")
    print("Keyboard       : DISABLED")
    print("Simulator truth: DISABLED")
    print("Exact path     : DEFERRED; current-state retarget only")
    print()

    print("Loading model...", flush=True)
    policy = LiveV11Policy(args.checkpoint, device=args.device, config=config)
    print(
        f"Model loaded   : format={policy.checkpoint.get('format')} "
        f"epoch={policy.checkpoint.get('epoch')}"
    )
    print("Warm-up        : V1.1 neural/search warm-up; spawn exact-path BFS skipped...", flush=True)
    warm = warmup_live_policy(policy)
    print(f"Warm-up done   : {_fmt_timings(warm.timings_ms)}")
    print()

    for remain in range(max(0, args.countdown), 0, -1):
        print(f"Starting in {remain}...", flush=True)
        time.sleep(1.0)

    tracker = TemporalStateTracker(TrackerConfig())
    rapid = None if args.no_ocr else RapidUsernameReader()
    reserved_generations: set[tuple[str, str | None, tuple[str, ...]]] = set()
    decision_rows: list[dict] = []
    counters = {
        "decisions": 0,
        "prepared_completed": 0,
        "prepared_cache_hits": 0,
        "prepared_cache_misses": 0,
        "retarget_safe": 0,
        "retarget_fail": 0,
        "retarget_hold_skipped": 0,
        "planner_queued": 0,
        "generation_dedup_skips": 0,
        "retarget_retries": 0,
    }
    first_prepared = None
    latest_prepared = None
    prep_future = None
    queued_seed = None
    planner_future = None
    planner_meta = None
    queued_observation = None
    queued_generation = None
    pending_retarget = None
    retarget_future = None
    retarget_meta = None
    stop_reason = "timeout"

    planner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tetrio-v11")
    # One CPU worker can prepare NEXT while the other performs current-state
    # retarget.  At most one speculative preparation is submitted at a time,
    # leaving capacity for the latency-sensitive retarget job.
    cpu_pool = ProcessPoolExecutor(max_workers=2)

    try:
        with WindowsGdiScreenCapture() as capture:
            print(f"Virtual screen : {capture.geometry}")
            print("Acquiring SELF layout once...", flush=True)
            item, acquire_seconds, attempts = _acquire_layout(
                capture,
                self_username=args.self_username,
                no_ocr=args.no_ocr,
                rapid=rapid,
                timeout=30.0,
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
            print("Play normally. Capture keeps running while V1.1 plans in another thread.")
            print()

            started = time.perf_counter()
            interval = 1.0 / args.fps
            next_tick = started
            frame_index = 0

            while time.perf_counter() - started < args.seconds:
                # Collect speculative CPU preparation without blocking capture.
                if prep_future is not None and prep_future.done():
                    try:
                        prepared = prep_future.result()
                        counters["prepared_completed"] += 1
                        latest_prepared = prepared
                        if first_prepared is None:
                            first_prepared = prepared
                        print(
                            "  SPEC_PREP READY "
                            f"source={prepared.template.source_fingerprint} "
                            f"known_next={prepared.template.preview_prefix} "
                            f"missing={prepared.template.missing_preview} "
                            f"candidates={len(prepared.candidates)} paths=0 "
                            f"{_fmt_timings(prepared.timings_ms)}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"  SPEC_PREP FAIL {type(exc).__name__}: {exc}", flush=True)
                    prep_future = None
                    if queued_seed is not None:
                        prep_future = cpu_pool.submit(
                            build_speculative_preparation,
                            queued_seed,
                            config,
                        )
                        print(
                            f"  SPEC_PREP START queued source={queued_seed.source_fingerprint}",
                            flush=True,
                        )
                        queued_seed = None

                # Capture/vision never waits for the policy thread.
                image = capture.grab_region(region)
                board = read_visual_board(image, local_item.candidate)
                previews = read_piece_previews(image, local_item.candidate)
                tracked = tracker.update(
                    board,
                    previews,
                    layout_bbox=local_item.candidate.bbox,
                )
                frame_index += 1

                # Finish a planner result against the *latest* visual state.
                if planner_future is not None and planner_future.done():
                    try:
                        decision = planner_future.result()
                    except Exception as exc:
                        print(f"  PLANNER FAIL {type(exc).__name__}: {exc}", flush=True)
                        if planner_meta is not None:
                            reserved_generations.discard(planner_meta.get("generation_key"))
                        planner_future = None
                        planner_meta = None
                    else:
                        meta = planner_meta
                        planner_future = None
                        planner_meta = None
                        observation = meta["observation"]
                        mode = meta["mode"]
                        cache_reason = meta["cache_reason"]
                        if mode == "PREPARED":
                            counters["prepared_cache_hits"] += 1
                        elif cache_reason is not None:
                            counters["prepared_cache_misses"] += 1

                        counters["decisions"] += 1
                        ordinal = counters["decisions"]
                        chosen = decision.chosen
                        wall_ms = (time.perf_counter() - meta["submitted_at"]) * 1000.0
                        print(
                            f"[{frame_index:05d}] DECISION {ordinal}/{args.decisions} "
                            f"mode={mode} active={observation.active_piece} "
                            f"hold={observation.hold_piece or '-'} next={observation.preview_queue}"
                        )
                        if cache_reason is not None:
                            print(f"  SPEC_CACHE MISS reason={cache_reason}")
                        print(
                            "  CHOSEN "
                            f"hold={int(chosen.use_hold)} mode={chosen.branch_mode} "
                            f"piece={chosen.state.piece} r={int(chosen.state.rotation)%4} "
                            f"x={chosen.state.x} y={chosen.state.y} lines={chosen.lines} "
                            f"final={chosen.final_score:.3f} path=DEFERRED"
                        )
                        print(
                            f"  TIMING {_fmt_timings(decision.timings_ms)} "
                            f"planner_wall={wall_ms:.1f}ms",
                            flush=True,
                        )

                        row = {
                            "frame": frame_index,
                            "mode": mode,
                            "cache_miss_reason": cache_reason,
                            "observation": observation.to_dict(),
                            "decision": decision.to_dict(),
                            "planner_wall_ms": wall_ms,
                            "generation_key": list(meta["generation_key"][:2]) + [list(meta["generation_key"][2])],
                            "retarget": None,
                            "retarget_wall_ms": None,
                        }
                        decision_rows.append(row)

                        # Start next-state CPU work immediately; it contains no
                        # simulator truth and no keyboard action.
                        seed = make_speculation_seed(observation, decision)
                        if prep_future is None:
                            prep_future = cpu_pool.submit(
                                build_speculative_preparation,
                                seed,
                                config,
                            )
                            print(
                                f"  SPEC_PREP START source={seed.source_fingerprint} "
                                "(CPU worker; capture continues)",
                                flush=True,
                            )
                        else:
                            queued_seed = seed
                            print(
                                f"  SPEC_PREP QUEUED latest source={seed.source_fingerprint}",
                                flush=True,
                            )

                        if bool(chosen.use_hold):
                            retarget = RetargetResult(
                                False,
                                "hold_must_be_executed_and_reobserved_before_retarget",
                                (), None, (),
                            )
                            counters["retarget_hold_skipped"] += 1
                            row["retarget"] = retarget.to_dict()
                            print(f"  RETARGET WAIT/ABORT reason={retarget.reason}")
                        else:
                            # Do not run the CPU search in the capture loop.
                            # The next loop iteration will validate the newest
                            # visual state and submit a pickle-safe request.
                            pending_retarget = {
                                "observation": observation,
                                "decision": decision,
                                "row": row,
                                "attempts": 0,
                            }

                # Collect an asynchronous retarget result against the newest
                # visual frame. A path is accepted only if its exact visual
                # start state is still current; otherwise it is recomputed from
                # the newer state instead of executing a stale path.
                if retarget_future is not None and retarget_future.done():
                    meta = retarget_meta
                    try:
                        retarget = retarget_future.result()
                    except Exception as exc:
                        retarget = RetargetResult(
                            False,
                            f"retarget_worker_error:{type(exc).__name__}",
                            (), None, (),
                        )
                    retarget_future = None
                    retarget_meta = None
                    wall_ms = (time.perf_counter() - meta["submitted_at"]) * 1000.0
                    stale_reason = retarget_request_stale_reason(meta["request"], tracked)
                    retryable = (
                        stale_reason == "active_visual_state_advanced"
                        or retarget.reason in {"visual_state_unmappable", "fresh_active_unresolved"}
                    )
                    if retarget.safe and stale_reason is None:
                        meta["row"]["retarget"] = retarget.to_dict()
                        meta["row"]["retarget_wall_ms"] = wall_ms
                        counters["retarget_safe"] += 1
                        print(
                            f"  RETARGET SAFE states={len(retarget.current_states)} "
                            f"nodes={retarget.search_nodes} search={retarget.search_ms:.1f}ms "
                            f"wall={wall_ms:.1f}ms path={retarget.movement_path}",
                            flush=True,
                        )
                    elif retryable and meta["pending"]["attempts"] < max(1, args.retarget_probe_frames):
                        counters["retarget_retries"] += 1
                        pending_retarget = meta["pending"]
                    else:
                        reason = stale_reason or retarget.reason
                        final = retarget
                        if stale_reason is not None:
                            final = RetargetResult(
                                False, stale_reason, (), None, retarget.target_cells,
                                search_nodes=retarget.search_nodes,
                                search_ms=retarget.search_ms,
                            )
                        meta["row"]["retarget"] = final.to_dict()
                        meta["row"]["retarget_wall_ms"] = wall_ms
                        counters["retarget_fail"] += 1
                        print(
                            f"  RETARGET WAIT/ABORT reason={reason} "
                            f"nodes={retarget.search_nodes} search={retarget.search_ms:.1f}ms "
                            f"wall={wall_ms:.1f}ms",
                            flush=True,
                        )

                # Submit/retry a retarget request only after a fresh frame has
                # passed the cheap generation/board checks. The heavy search is
                # isolated in the CPU process pool so capture remains at 8 FPS.
                if pending_retarget is not None and retarget_future is None:
                    pending_retarget["attempts"] += 1
                    request, failure = build_retarget_request(
                        pending_retarget["observation"],
                        pending_retarget["decision"],
                        tracked,
                    )
                    if request is not None:
                        retarget_future = cpu_pool.submit(
                            run_retarget_request,
                            request,
                            max_states=args.reference_max_states,
                        )
                        retarget_meta = {
                            "request": request,
                            "pending": pending_retarget,
                            "row": pending_retarget["row"],
                            "submitted_at": time.perf_counter(),
                        }
                        pending_retarget = None
                    else:
                        assert failure is not None
                        terminal = failure.reason in {
                            "locked_board_changed_decision_stale",
                            "active_generation_changed",
                        }
                        if terminal or pending_retarget["attempts"] >= max(1, args.retarget_probe_frames):
                            pending_retarget["row"]["retarget"] = failure.to_dict()
                            pending_retarget["row"]["retarget_wall_ms"] = 0.0
                            counters["retarget_fail"] += 1
                            print(f"  RETARGET WAIT/ABORT reason={failure.reason}", flush=True)
                            pending_retarget = None

                # Discover model-ready snapshots continuously. Plan at most
                # once per Active/Hold/NEXT generation. The full observation
                # fingerprint includes the board, which can jitter while the
                # same falling piece remains alive; that must not trigger a
                # second policy decision for the same transaction.
                if (
                    counters["decisions"] < args.decisions
                    and tracked.stable_pre_action
                ):
                    try:
                        observation = build_model_observation(tracked)
                    except ObservationNotReady:
                        observation = None
                    if observation is not None:
                        generation_key = observation_generation_key(observation)
                        if generation_key in reserved_generations:
                            counters["generation_dedup_skips"] += 1
                        else:
                            reserved_generations.add(generation_key)
                            if planner_future is None:
                                planner_future, planner_meta = _submit_plan(
                                    planner_pool,
                                    policy,
                                    observation,
                                    latest_prepared,
                                )
                                planner_meta["generation_key"] = generation_key
                                if latest_prepared is not None:
                                    latest_prepared = None
                                print(
                                    f"[{frame_index:05d}] PLANNER START "
                                    f"active={observation.active_piece} fp={observation.fingerprint}",
                                    flush=True,
                                )
                            else:
                                # Keep only the newest unseen generation. If a
                                # later generation arrives before this queued one
                                # starts, release the old reservation so a future
                                # stable snapshot may be considered again.
                                if queued_generation is not None:
                                    reserved_generations.discard(queued_generation)
                                queued_observation = observation
                                queued_generation = generation_key
                                counters["planner_queued"] += 1
                                print(
                                    f"[{frame_index:05d}] PLANNER QUEUED latest "
                                    f"active={observation.active_piece} fp={observation.fingerprint}",
                                    flush=True,
                                )

                if (
                    planner_future is None
                    and queued_observation is not None
                    and counters["decisions"] < args.decisions
                ):
                    observation = queued_observation
                    generation_key = queued_generation
                    queued_observation = None
                    queued_generation = None
                    planner_future, planner_meta = _submit_plan(
                        planner_pool,
                        policy,
                        observation,
                        latest_prepared,
                    )
                    assert generation_key is not None
                    planner_meta["generation_key"] = generation_key
                    if latest_prepared is not None:
                        latest_prepared = None
                    print(
                        f"[{frame_index:05d}] PLANNER START queued "
                        f"active={observation.active_piece} fp={observation.fingerprint}",
                        flush=True,
                    )

                if (
                    counters["decisions"] >= args.decisions
                    and planner_future is None
                    and pending_retarget is None
                    and retarget_future is None
                ):
                    stop_reason = "decision_gate_reached"
                    break

                next_tick += interval
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -interval * 3:
                    next_tick = time.perf_counter()

        # Observe at least one speculative result for the contract.
        if first_prepared is None and prep_future is not None:
            prepared = prep_future.result()
            counters["prepared_completed"] += 1
            first_prepared = prepared
            latest_prepared = prepared
            print(
                "SPEC_PREP FINAL "
                f"candidates={len(prepared.candidates)} paths=0 "
                f"{_fmt_timings(prepared.timings_ms)}"
            )

        parity = None
        parity_full_ms = None
        parity_prepared_ms = None
        if first_prepared is not None:
            synthetic = synthetic_completed_observation(first_prepared)
            print("Prepared parity : comparing landing decision only (path deferred)...", flush=True)
            full = policy.decide(synthetic, resolve_exact_path=False)
            fast = decide_prepared(
                policy,
                synthetic,
                first_prepared,
                resolve_exact_path=False,
            )
            parity = decisions_equivalent(full, fast, require_path=False)
            parity_full_ms = float(full.timings_ms["total"])
            parity_prepared_ms = float(fast.timings_ms["total"])
            print(
                f"Prepared parity : {'PASS' if parity else 'FAIL'} "
                f"full={parity_full_ms:.1f}ms prepared={parity_prepared_ms:.1f}ms "
                f"geometry={fast.chosen.state.geometry_key()} hold={int(fast.chosen.use_hold)}"
            )

        nohold_retarget_opportunities = sum(
            not bool(row["decision"]["chosen"]["use_hold"])
            for row in decision_rows
        )
        retarget_gate = (
            counters["retarget_safe"] >= 1
            if nohold_retarget_opportunities > 0
            else True
        )

        full_times = [
            float(row["decision"]["timings_ms"]["total"])
            for row in decision_rows
            if row["mode"] == "FULL"
        ]
        safe_retarget_times = [
            float(row["retarget"]["search_ms"])
            for row in decision_rows
            if row.get("retarget") and row["retarget"].get("safe")
        ]
        safe_retarget_wall_times = [
            float(row["retarget_wall_ms"])
            for row in decision_rows
            if row.get("retarget") and row["retarget"].get("safe")
            and row.get("retarget_wall_ms") is not None
        ]
        spec_ms = None if first_prepared is None else float(first_prepared.timings_ms["total"])
        perf_gate = (
            (not full_times or max(full_times) <= args.max_full_foreground_ms)
            and (parity_prepared_ms is None or parity_prepared_ms <= args.max_prepared_foreground_ms)
            and (spec_ms is None or spec_ms <= args.max_spec_prep_ms)
            and (not safe_retarget_times or max(safe_retarget_times) <= args.max_retarget_ms)
            and (not safe_retarget_wall_times or max(safe_retarget_wall_times) <= args.max_retarget_wall_ms)
        )

        passed = (
            counters["decisions"] >= args.decisions
            and counters["prepared_completed"] >= 1
            and parity is True
            and retarget_gate
            and perf_gate
        )
        summary = {
            "format": "tetrio_phase_c1_5_generation_dedup_async_retarget_gate_v3",
            "status": "PASS" if passed else "FAIL",
            "stop_reason": stop_reason,
            "checkpoint": str(args.checkpoint),
            "warmup": warm.to_dict(),
            "counters": counters,
            "prepared_parity": parity,
            "retarget_gate": retarget_gate,
            "performance_gate": perf_gate,
            "performance": {
                "full_foreground_ms": full_times,
                "prepared_parity_full_ms": parity_full_ms,
                "prepared_foreground_ms": parity_prepared_ms,
                "spec_prep_ms": spec_ms,
                "safe_retarget_ms": safe_retarget_times,
                "safe_retarget_wall_ms": safe_retarget_wall_times,
                "limits": {
                    "max_full_foreground_ms": args.max_full_foreground_ms,
                    "max_prepared_foreground_ms": args.max_prepared_foreground_ms,
                    "max_spec_prep_ms": args.max_spec_prep_ms,
                    "max_retarget_ms": args.max_retarget_ms,
                    "max_retarget_wall_ms": args.max_retarget_wall_ms,
                },
            },
            "decisions": decision_rows,
            "first_prepared": None if first_prepared is None else first_prepared.to_dict(),
        }
        (args.output_dir / "c1_5_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 122)
        print(f"PHASE C1.5 V3 GATE: {summary['status']}")
        print("=" * 122)
        print(
            f"Decisions={counters['decisions']} "
            f"PrepCompleted={counters['prepared_completed']} "
            f"CacheHit={counters['prepared_cache_hits']} "
            f"CacheMiss={counters['prepared_cache_misses']} "
            f"RetargetSafe={counters['retarget_safe']} "
            f"RetargetFail={counters['retarget_fail']} "
            f"HoldRetargetSkipped={counters['retarget_hold_skipped']}"
        )
        print(
            f"PreparedParity={parity} RetargetGate={retarget_gate} "
            f"PerformanceGate={perf_gate}"
        )
        if full_times:
            print(f"Full foreground max : {max(full_times):.1f}ms")
        if parity_prepared_ms is not None:
            print(f"Prepared foreground : {parity_prepared_ms:.1f}ms")
        if spec_ms is not None:
            print(f"Spec prep background : {spec_ms:.1f}ms")
        if safe_retarget_times:
            print(f"Retarget search max  : {max(safe_retarget_times):.1f}ms")
        if safe_retarget_wall_times:
            print(f"Retarget worker wall : {max(safe_retarget_wall_times):.1f}ms")
        print(f"Generation dedup skips: {counters['generation_dedup_skips']}")
        print(f"Report: {args.output_dir / 'c1_5_summary.json'}")
        if not passed:
            raise SystemExit(2)
    finally:
        planner_pool.shutdown(wait=True, cancel_futures=False)
        cpu_pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
