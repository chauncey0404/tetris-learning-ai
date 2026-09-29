from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from tetrio.network.checkpoint import load_expert_v1
from tetrio.tools.watch_expert_v0 import (
    BranchPlan,
    Decision,
    ExpertV0Rollout,
    RolloutViewer,
    ScoredCandidate,
    aggregate_results,
    board_metrics,
    parse_seed_spec,
)


@dataclass(frozen=True)
class JointCandidate(ScoredCandidate):
    use_hold: bool = False
    branch_mode: str = "no_hold"


class ExpertV1Rollout(ExpertV0Rollout):
    """Closed-loop Expert-v1: both Hold branches compete on every move."""

    def refresh_decision(self) -> Decision | None:
        if self.game_over:
            self.pending_decision = None
            return None
        if self.max_pieces > 0 and self.stats.pieces >= self.max_pieces:
            self.game_over = True
            self.terminal_reason = "LIMIT"
            self.pending_decision = None
            return None

        state_tensor = self._state_tensor()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            state_latent = self.model.scorer.encode_state(state_tensor)

        audit = (
            self.backend == "fast"
            and self.reference_audit_every > 0
            and self.stats.pieces % self.reference_audit_every == 0
        )

        all_candidates: list[JointCandidate] = []
        candidate_branches: list[BranchPlan] = []
        any_audit = False
        branch_best = {}

        for use_hold in (False, True):
            branch = self._plan_branch(use_hold)
            candidates, audited = self._build_scored_candidates(
                branch,
                state_latent,
                audit=audit,
            )
            any_audit = any_audit or audited

            converted = [
                JointCandidate(
                    state=c.state,
                    board_after=c.board_after,
                    lines=c.lines,
                    score=c.score,
                    use_hold=use_hold,
                    branch_mode=branch.mode,
                )
                for c in candidates
            ]
            start = len(all_candidates)
            all_candidates.extend(converted)
            candidate_branches.extend([branch] * len(converted))
            if converted:
                branch_best[use_hold] = max(c.score for c in converted)

        if not all_candidates:
            self.game_over = True
            self.terminal_reason = "NO_REACHABLE_PLACEMENTS_BOTH_BRANCHES"
            self.pending_decision = None
            return None

        scores = np.asarray(
            [c.score for c in all_candidates],
            dtype=np.float32,
        )
        chosen_index = int(np.argmax(scores))
        chosen = all_candidates[chosen_index]
        chosen_branch = candidate_branches[chosen_index]

        # A readable branch confidence, not a separately trained Hold head:
        # softmax over each branch's best score.
        if False in branch_best and True in branch_best:
            nh = float(branch_best[False])
            hh = float(branch_best[True])
            delta = max(-30.0, min(30.0, hh - nh))
            hold_probability = 1.0 / (1.0 + math.exp(-delta))
        elif True in branch_best:
            hold_probability = 1.0
        else:
            hold_probability = 0.0

        _, holes_before = board_metrics(self.board)
        candidate_holes = [
            board_metrics(c.board_after)[1]
            for c in all_candidates
        ]
        min_candidate_holes = min(candidate_holes)
        chosen_holes = int(candidate_holes[chosen_index])

        safer_index = None
        if min_candidate_holes < chosen_holes:
            safer_pool = [
                i for i, h in enumerate(candidate_holes)
                if h == min_candidate_holes
            ]
            safer_index = max(
                safer_pool,
                key=lambda i: all_candidates[i].score,
            )

        decision = Decision(
            active=self.active,
            hold_before=self.hold,
            preview_before=self.preview(),
            hold_probability=hold_probability,
            use_hold=bool(chosen.use_hold),
            branch=chosen_branch,
            candidates=tuple(all_candidates),
            chosen_index=chosen_index,
            audited_reference=any_audit,
            holes_before=int(holes_before),
            chosen_holes_after=chosen_holes,
            chosen_hole_delta=chosen_holes - int(holes_before),
            min_candidate_holes=int(min_candidate_holes),
            safer_candidate_index=safer_index,
        )
        self.pending_decision = decision
        return decision


class ExpertV1Viewer(RolloutViewer):
    """V1 uses the shared V3.4 animation/history renderer."""

    VIEWER_TITLE = (
        "Tetris Learning AI - TETR.IO Expert v1 Autonomous "
        "V3.4 Visible Rotation + History Replay"
    )
    PANEL_TITLE = "EXPERT v1 AUTONOMOUS"
    RESULT_FORMAT = "tetrio_expert_v1_autonomous_rollout_gui"
    SCREENSHOT_DIR = Path(
        r"artifacts\tetrio\expert_v1_rollout_screenshots"
    )

    def _draw_side(self, rect):
        super()._draw_side(rect)
        self._text(
            "V1: Hold/No-Hold are jointly ranked; no binary Hold head",
            rect.x + 14,
            rect.bottom - 27,
            font=self.font_small,
            color=(160, 166, 178),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch TETR.IO Expert-v1 unified Hold/No-Hold rollout."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_joint_100k.pt"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=9031)
    p.add_argument("--seeds", default="")
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--reference-audit-every", type=int, default=250)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument(
        "--no-fall-animation",
        action="store_true",
        help="Disable the V3.4-style purely visual falling animation.",
    )
    p.add_argument("--width", type=int, default=1220)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--state-batch", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=512)
    p.add_argument(
        "--save-json",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_autonomous_rollout.json"),
    )
    return p.parse_args()


def _new_session(model, device, seed, args):
    return ExpertV1Rollout(
        model,
        device=device,
        seed=seed,
        max_pieces=args.max_pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        # Parent field remains unused by V1 refresh_decision.
        hold_threshold=0.5,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    device = torch.device(args.device)
    model, checkpoint = load_expert_v1(
        args.checkpoint,
        device=device,
    )
    model.eval()

    if args.headless:
        import json
        from tetrio.rollout.batched import (
            BatchedRolloutConfig,
            run_batched_v1,
        )

        seeds = parse_seed_spec(args.seeds, args.seed)
        batch_result = run_batched_v1(
            model,
            seeds=seeds,
            device=device,
            config=BatchedRolloutConfig(
                max_pieces=args.max_pieces,
                backend=args.backend,
                fast_max_states=args.fast_max_states,
                reference_max_states=args.reference_max_states,
                reference_audit_every=args.reference_audit_every,
                workers=args.workers,
                state_batch=args.state_batch,
                progress_every=args.progress_every,
            ),
        )
        results = batch_result["results"]
        report = {
            "format": "tetrio_expert_v1_autonomous_rollout",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "results": results,
            "aggregate": batch_result["aggregate"],
            "runtime": batch_result["runtime"],
            "status": "DEVELOPMENT ROLLOUT (not Champion qualification)",
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report: {args.save_json}")
        return

    session = _new_session(model, device, args.seed, args)
    viewer = ExpertV1Viewer(
        session,
        width=args.width,
        height=args.height,
        speed=args.speed,
        save_json=args.save_json,
        fall_animation=not args.no_fall_animation,
    )
    viewer.run()


if __name__ == "__main__":
    main()
