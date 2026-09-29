from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from tetrio.future.features import feature_index
from tetrio.future.lookahead import (
    FutureCandidateInput,
    FutureFeatureConfig,
    build_row_future_features,
)
from tetrio.network.checkpoint import load_expert_v1_1
from tetrio.tools.build_expert_v1_1_future_cache import select_inference_shortlist
from tetrio.tools.watch_expert_v0 import (
    Decision,
    RolloutViewer,
    ScoredCandidate,
    aggregate_results,
    board_metrics,
    parse_seed_spec,
)
from tetrio.tools.watch_expert_v1 import (
    ExpertV1Rollout,
    ExpertV1Viewer,
)


TDEST = feature_index("t_opportunity_destroyed")
TDEFER = feature_index("t_cashout_deferred")


class ExpertV11Rollout(ExpertV1Rollout):
    """Expert-v1.1 closed loop: V1 joint candidates + future residual reranking."""

    def __init__(self, *args, top_overall=8, top_per_branch=4, **kwargs):
        self.top_overall = int(top_overall)
        self.top_per_branch = int(top_per_branch)
        self.future_config = FutureFeatureConfig()
        self.last_v11_diag = {}
        super().__init__(*args, **kwargs)

    def refresh_decision(self) -> Decision | None:
        # Parent computes the complete V1 joint candidate set and base scores,
        # but does not mutate the board.
        base_decision = super().refresh_decision()
        if base_decision is None:
            return None

        all_candidates = list(base_decision.candidates)
        base_scores = np.asarray(
            [c.score for c in all_candidates],
            dtype=np.float32,
        )
        holds = np.asarray(
            [getattr(c, "use_hold", False) for c in all_candidates],
            dtype=bool,
        )
        shortlist = select_inference_shortlist(
            base_scores,
            holds,
            top_overall=self.top_overall,
            top_per_branch=self.top_per_branch,
        )

        future_inputs = tuple(
            FutureCandidateInput(
                board_after=all_candidates[i].board_after,
                piece=all_candidates[i].state.piece,
                rotation=all_candidates[i].state.rotation,
                x=all_candidates[i].state.x,
                y=all_candidates[i].state.y,
                use_hold=bool(holds[i]),
                lines=all_candidates[i].lines,
            )
            for i in shortlist
        )
        features = build_row_future_features(
            board_before=self.board,
            active=self.active,
            hold=self.hold,
            preview=self.preview(),
            candidates=future_inputs,
            config=self.future_config,
        )

        bscore = torch.from_numpy(base_scores[shortlist][None, :]).to(
            self.device,
            dtype=torch.float32,
        )
        f = torch.from_numpy(features[None, ...]).to(
            self.device,
            dtype=torch.float32,
        )
        h = torch.from_numpy(holds[shortlist][None, :]).to(
            self.device,
            dtype=torch.bool,
        )
        mask = torch.ones(
            (1, len(shortlist)),
            device=self.device,
            dtype=torch.bool,
        )

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=(
                torch.bfloat16
                if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            enabled=self.device.type == "cuda",
        ):
            final_short, residual = self.model.final_scores(
                base_scores=bscore,
                raw_features=f,
                candidate_use_hold=h,
                mask=mask,
            )

        final_short_np = final_short[0].float().cpu().numpy()
        residual_np = residual[0].float().cpu().numpy()
        winner_local = int(np.argmax(final_short_np))
        winner_global = int(shortlist[winner_local])

        # Non-shortlisted candidates cannot win the V1.1 policy, but keeping the
        # full tuple preserves global structural diagnostics/candidate counts.
        final_global = np.full_like(base_scores, -1e9)
        for local_i, global_i in enumerate(shortlist):
            final_global[global_i] = final_short_np[local_i]

        rescored = tuple(
            ScoredCandidate(
                state=c.state,
                board_after=c.board_after,
                lines=c.lines,
                score=float(final_global[i]),
            )
            if not hasattr(c, "use_hold")
            else replace(c, score=float(final_global[i]))
            for i, c in enumerate(all_candidates)
        )

        chosen = rescored[winner_global]
        chosen_hold = bool(holds[winner_global])

        branch_scores = {}
        for local_i, global_i in enumerate(shortlist):
            branch = bool(holds[global_i])
            branch_scores[branch] = max(
                branch_scores.get(branch, -1e9),
                float(final_short_np[local_i]),
            )
        if False in branch_scores and True in branch_scores:
            delta = max(-30.0, min(30.0, branch_scores[True] - branch_scores[False]))
            hold_probability = 1.0 / (1.0 + math.exp(-delta))
        elif True in branch_scores:
            hold_probability = 1.0
        else:
            hold_probability = 0.0

        _, holes_before = board_metrics(self.board)
        candidate_holes = [
            board_metrics(c.board_after)[1]
            for c in all_candidates
        ]
        min_holes = min(candidate_holes)
        chosen_holes = int(candidate_holes[winner_global])

        safer_index = None
        if min_holes < chosen_holes:
            safer_pool = [
                i for i, holes in enumerate(candidate_holes)
                if holes == min_holes
            ]
            safer_index = max(
                safer_pool,
                key=lambda i: float(final_global[i]),
            )

        # Recover the correct BranchPlan from the parent's per-candidate mode.
        chosen_base = all_candidates[winner_global]
        branch = self._plan_branch(chosen_hold)

        decision = Decision(
            active=self.active,
            hold_before=self.hold,
            preview_before=self.preview(),
            hold_probability=hold_probability,
            use_hold=chosen_hold,
            branch=branch,
            candidates=rescored,
            chosen_index=winner_global,
            audited_reference=base_decision.audited_reference,
            holes_before=int(holes_before),
            chosen_holes_after=chosen_holes,
            chosen_hole_delta=chosen_holes - int(holes_before),
            min_candidate_holes=int(min_holes),
            safer_candidate_index=safer_index,
        )

        self.last_v11_diag = {
            "shortlist": len(shortlist),
            "base_winner_global": int(np.argmax(base_scores)),
            "v11_winner_global": winner_global,
            "base_score": float(base_scores[winner_global]),
            "residual": float(residual_np[winner_local]),
            "final_score": float(final_short_np[winner_local]),
            "t_destroyed": int(features[winner_local, TDEST] > 0.5),
            "t_deferred": int(features[winner_local, TDEFER] > 0.5),
        }
        self.pending_decision = decision
        return decision


class ExpertV11Viewer(ExpertV1Viewer):
    VIEWER_TITLE = (
        "Tetris Learning AI - TETR.IO Expert v1.1 Future Reranker "
        "V3.4 Visible Rotation + History Replay"
    )
    PANEL_TITLE = "EXPERT v1.1 FUTURE-AWARE"
    RESULT_FORMAT = "tetrio_expert_v1_1_autonomous_rollout_gui"
    SCREENSHOT_DIR = Path(
        r"artifacts\tetrio\expert_v1_1_rollout_screenshots"
    )

    def _draw_side(self, rect):
        RolloutViewer._draw_side(self, rect)
        if hasattr(self.session, "last_v11_diag") and self.session.last_v11_diag:
            d = self.session.last_v11_diag
            self._text(
                f"V1.1 shortlist={d['shortlist']} "
                f"base={d['base_score']:.3f} residual={d['residual']:+.3f} "
                f"Tdestroy={d['t_destroyed']} Tdefer={d['t_deferred']}",
                rect.x + 14,
                rect.bottom - 27,
                font=self.font_small,
                color=(160, 166, 178),
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch Expert-v1.1 future-aware autonomous rollout."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v1_1_future_100k.pt"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=9051)
    p.add_argument("--seeds", default="")
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument("--backend", choices=("fast", "reference"), default="fast")
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument("--reference-audit-every", type=int, default=250)
    p.add_argument("--top-overall", type=int, default=8)
    p.add_argument("--top-per-branch", type=int, default=4)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument("--no-fall-animation", action="store_true")
    p.add_argument("--width", type=int, default=1220)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--state-batch", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=512)
    p.add_argument(
        "--save-json",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v1_1_autonomous_rollout.json"),
    )
    return p.parse_args()


def _session(model, device, seed, args):
    return ExpertV11Rollout(
        model,
        device=device,
        seed=seed,
        max_pieces=args.max_pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        hold_threshold=0.5,
        top_overall=args.top_overall,
        top_per_branch=args.top_per_branch,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    device = torch.device(args.device)
    model, ckpt = load_expert_v1_1(args.checkpoint, device=device)

    if args.headless:
        from tetrio.rollout.batched import (
            BatchedRolloutConfig,
            run_batched_v11,
        )

        seeds = parse_seed_spec(args.seeds, args.seed)
        batch_result = run_batched_v11(
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
                top_overall=args.top_overall,
                top_per_branch=args.top_per_branch,
            ),
        )
        results = batch_result["results"]
        report = {
            "format": "tetrio_expert_v1_1_autonomous_rollout",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": ckpt.get("epoch"),
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

    s = _session(model, device, args.seed, args)
    viewer = ExpertV11Viewer(
        s,
        width=args.width,
        height=args.height,
        speed=args.speed,
        save_json=args.save_json,
        fall_animation=not args.no_fall_animation,
    )
    viewer.run()


if __name__ == "__main__":
    main()
