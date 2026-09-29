from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from tetrio.vision.observation import ModelObservation


@dataclass(frozen=True)
class LiveV11PolicyConfig:
    backend: str = "fast"
    fast_max_states: int = 10_000
    reference_max_states: int = 50_000
    top_overall: int = 8
    top_per_branch: int = 4

    def __post_init__(self) -> None:
        if self.backend not in ("fast", "reference"):
            raise ValueError("backend must be 'fast' or 'reference'")
        if self.fast_max_states < 1 or self.reference_max_states < 1:
            raise ValueError("reachability max_states values must be >= 1")
        if self.top_overall < 1 or self.top_per_branch < 1:
            raise ValueError("shortlist sizes must be >= 1")


@dataclass(frozen=True)
class BranchPlan:
    selected_piece: str
    use_hold: bool
    hold_after: str | None
    next_active: str
    consume_count: int
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiveCandidate:
    state: Any
    board_after: np.ndarray
    lines: int
    use_hold: bool
    branch_mode: str
    base_score: float
    residual: float | None = None
    final_score: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "piece": str(self.state.piece),
            "rotation": int(self.state.rotation) % 4,
            "x": int(self.state.x),
            "y": int(self.state.y),
            "lines": int(self.lines),
            "use_hold": bool(self.use_hold),
            "branch_mode": str(self.branch_mode),
            "base_score": float(self.base_score),
            "residual": None if self.residual is None else float(self.residual),
            "final_score": None if self.final_score is None else float(self.final_score),
        }


@dataclass(frozen=True)
class LiveDecision:
    observation_fingerprint: str
    active_piece: str
    hold_piece: str | None
    preview_queue: tuple[str, ...]
    candidates: tuple[LiveCandidate, ...]
    shortlist_indices: tuple[int, ...]
    chosen_index: int
    branch: BranchPlan
    movement_path: tuple[str, ...]
    timings_ms: dict[str, float]

    @property
    def chosen(self) -> LiveCandidate:
        return self.candidates[self.chosen_index]

    @property
    def top3(self) -> tuple[tuple[int, LiveCandidate], ...]:
        ranked = sorted(
            self.shortlist_indices,
            key=lambda i: float(self.candidates[i].final_score),
            reverse=True,
        )
        return tuple((i, self.candidates[i]) for i in ranked[:3])

    def to_dict(self) -> dict[str, Any]:
        branch_counts = {
            "no_hold": sum(not c.use_hold for c in self.candidates),
            "hold": sum(c.use_hold for c in self.candidates),
        }
        return {
            "observation_fingerprint": self.observation_fingerprint,
            "active_piece": self.active_piece,
            "hold_piece": self.hold_piece,
            "preview_queue": list(self.preview_queue),
            "candidate_count": len(self.candidates),
            "branch_candidate_counts": branch_counts,
            "shortlist_indices": list(self.shortlist_indices),
            "shortlist_count": len(self.shortlist_indices),
            "chosen_index": int(self.chosen_index),
            "chosen": self.chosen.summary(),
            "branch": self.branch.to_dict(),
            "movement_path": list(self.movement_path),
            "top3": [
                {"index": int(index), **candidate.summary()}
                for index, candidate in self.top3
            ],
            "timings_ms": {k: float(v) for k, v in self.timings_ms.items()},
        }


def branch_plans(observation: ModelObservation) -> tuple[BranchPlan, BranchPlan]:
    preview = observation.preview_queue
    no_hold = BranchPlan(
        selected_piece=observation.active_piece,
        use_hold=False,
        hold_after=observation.hold_piece,
        next_active=preview[0],
        consume_count=1,
        mode="no_hold",
    )
    if observation.hold_piece is None:
        hold = BranchPlan(
            selected_piece=preview[0],
            use_hold=True,
            hold_after=observation.active_piece,
            next_active=preview[1],
            consume_count=2,
            mode="hold_empty",
        )
    else:
        hold = BranchPlan(
            selected_piece=observation.hold_piece,
            use_hold=True,
            hold_after=observation.active_piece,
            next_active=preview[0],
            consume_count=1,
            mode="hold_swap",
        )
    return no_hold, hold


def _reference_best_placements(board: np.ndarray, piece: str, max_states: int):
    from tetrio.reachability import enumerate_tetrio_reachable_placements

    best = {}
    for placement in enumerate_tetrio_reachable_placements(
        board,
        piece,
        max_states=int(max_states),
    ):
        key = placement.landing_state.geometry_key()
        previous = best.get(key)
        if previous is None or (
            len(placement.path),
            tuple(action.value for action in placement.path),
        ) < (
            len(previous.path),
            tuple(action.value for action in previous.path),
        ):
            best[key] = placement
    return [
        best[key]
        for key in sorted(
            best,
            key=lambda k: (int(k[3]) % 4, int(k[1]), int(k[2])),
        )
    ]


def _landing_states(
    board: np.ndarray,
    piece: str,
    config: LiveV11PolicyConfig,
):
    if config.backend == "reference":
        return [
            placement.landing_state
            for placement in _reference_best_placements(
                board,
                piece,
                config.reference_max_states,
            )
        ]

    from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast

    fast = enumerate_tetrio_reachable_geometries_fast(
        board,
        piece,
        max_states=int(config.fast_max_states),
    )
    if fast:
        return fast

    # Production fail-safe already used by the validated rollout backend.
    return [
        placement.landing_state
        for placement in _reference_best_placements(
            board,
            piece,
            config.reference_max_states,
        )
    ]


def _exact_path_for_geometry(
    board: np.ndarray,
    state,
    max_states: int,
) -> tuple[str, ...]:
    target = state.geometry_key()
    matches = [
        placement
        for placement in _reference_best_placements(
            board,
            str(state.piece),
            max_states,
        )
        if placement.landing_state.geometry_key() == target
    ]
    if not matches:
        raise RuntimeError(
            "Selected policy geometry has no reference path: "
            f"{target!r}"
        )
    chosen = min(
        matches,
        key=lambda p: (
            len(p.path),
            tuple(action.value for action in p.path),
        ),
    )
    return tuple(action.value for action in chosen.path)


class LiveV11Policy:
    """Pure single-state Expert-v1.1 policy for vision-driven inference.

    This class never mutates the game, never sends keyboard input, and never
    reads simulator truth.  Its only state input is ``ModelObservation``.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        config: LiveV11PolicyConfig = LiveV11PolicyConfig(),
    ) -> None:
        import torch
        from tetrio.network.checkpoint import load_expert_v1_1

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self.config = config
        self.checkpoint_path = Path(checkpoint)
        self.model, self.checkpoint = load_expert_v1_1(
            self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

    def _state_latent(self, observation: ModelObservation):
        from tetrio.network.encoding import (
            dense_state_batch,
            pack_board,
            piece_id,
        )

        dense = dense_state_batch(
            np.stack([pack_board(observation.board_array())]),
            np.asarray([piece_id(observation.active_piece)], dtype=np.uint8),
            np.asarray([piece_id(observation.hold_piece)], dtype=np.uint8),
            np.asarray(
                [[piece_id(p) for p in observation.preview_queue]],
                dtype=np.uint8,
            ),
        )
        state_tensor = self.torch.from_numpy(dense).to(
            device=self.device,
            non_blocking=self.device.type == "cuda",
        )
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            return self.model.scorer.encode_state(state_tensor)

    def _score_branch(
        self,
        *,
        board: np.ndarray,
        plan: BranchPlan,
        state_latent,
    ) -> tuple[LiveCandidate, ...]:
        from tetrio.network.encoding import dense_candidate_batch, pack_board, piece_id
        from tetrio.ruleset import TETRIO_MOVEMENT
        from tetris_ai.core.movement import clear_lines, lock_piece

        landings = _landing_states(board, plan.selected_piece, self.config)
        if not landings:
            return ()

        boards = []
        lines = []
        for landing in landings:
            locked = lock_piece(board, landing, TETRIO_MOVEMENT)
            after, cleared = clear_lines(locked, TETRIO_MOVEMENT)
            boards.append(np.asarray(after, dtype=np.uint8))
            lines.append(int(cleared))

        dense = dense_candidate_batch(
            np.stack([pack_board(x) for x in boards]),
            np.asarray([piece_id(plan.selected_piece)] * len(landings), dtype=np.uint8),
            np.asarray([x.rotation for x in landings], dtype=np.uint8),
            np.asarray([x.x for x in landings], dtype=np.int8),
            np.asarray([x.y for x in landings], dtype=np.int8),
            np.asarray([int(plan.use_hold)] * len(landings), dtype=np.uint8),
            np.asarray(lines, dtype=np.uint8),
        )
        candidate_tensor = self.torch.from_numpy(dense).to(
            device=self.device,
            non_blocking=self.device.type == "cuda",
        ).unsqueeze(0)

        # Keep the exact legacy V1/V1.1 call shape: each HOLD branch is scored
        # separately.  BF16 near-ties can change when GEMM shapes change.
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            scores = self.model.scorer.score_from_state_latent(
                state_latent,
                candidate_tensor,
            )[0]
        scores_np = scores.float().cpu().numpy()

        return tuple(
            LiveCandidate(
                state=landing,
                board_after=board_after,
                lines=line_count,
                use_hold=plan.use_hold,
                branch_mode=plan.mode,
                base_score=float(score),
            )
            for landing, board_after, line_count, score in zip(
                landings,
                boards,
                lines,
                scores_np,
            )
        )

    def decide(
        self,
        observation: ModelObservation,
        *,
        resolve_exact_path: bool = True,
    ) -> LiveDecision:
        from tetrio.future.lookahead import (
            FutureCandidateInput,
            FutureFeatureConfig,
            build_row_future_features,
        )
        from tetrio.tools.build_expert_v1_1_future_cache import (
            select_inference_shortlist,
        )

        timings: dict[str, float] = {}
        total_start = time.perf_counter()
        board = observation.board_array()

        t0 = time.perf_counter()
        latent = self._state_latent(observation)
        timings["state_encode"] = (time.perf_counter() - t0) * 1000.0

        plans = branch_plans(observation)
        t0 = time.perf_counter()
        branch_candidates = [
            self._score_branch(board=board, plan=plan, state_latent=latent)
            for plan in plans
        ]
        candidates = tuple(x for branch in branch_candidates for x in branch)
        timings["enumerate_and_base_score"] = (time.perf_counter() - t0) * 1000.0
        if not candidates:
            raise RuntimeError("No reachable placements in either HOLD branch")

        base_scores = np.asarray([c.base_score for c in candidates], dtype=np.float32)
        holds = np.asarray([int(c.use_hold) for c in candidates], dtype=np.uint8)
        shortlist = select_inference_shortlist(
            base_scores,
            holds,
            top_overall=self.config.top_overall,
            top_per_branch=self.config.top_per_branch,
        )
        if not shortlist:
            raise RuntimeError("V1.1 shortlist is empty")

        t0 = time.perf_counter()
        future_inputs = tuple(
            FutureCandidateInput(
                board_after=candidates[i].board_after,
                piece=str(candidates[i].state.piece),
                rotation=int(candidates[i].state.rotation),
                x=int(candidates[i].state.x),
                y=int(candidates[i].state.y),
                use_hold=bool(candidates[i].use_hold),
                lines=int(candidates[i].lines),
            )
            for i in shortlist
        )
        features = build_row_future_features(
            board_before=board,
            active=observation.active_piece,
            hold=observation.hold_piece,
            preview=observation.preview_queue,
            candidates=future_inputs,
            config=FutureFeatureConfig(
                fast_max_states=self.config.fast_max_states,
                reference_max_states=self.config.reference_max_states,
                tactical_preview_depth=2,
                exact_immediate_t=False,
            ),
        )
        timings["future_features"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        base_t = self.torch.from_numpy(base_scores[shortlist][None, :]).to(
            device=self.device,
            dtype=self.torch.float32,
        )
        feat_t = self.torch.from_numpy(
            np.asarray(features, dtype=np.float32)[None, :, :]
        ).to(device=self.device, dtype=self.torch.float32)
        hold_t = self.torch.from_numpy(holds[shortlist][None, :]).to(
            device=self.device
        ).bool()
        mask_t = self.torch.ones(
            (1, len(shortlist)),
            device=self.device,
            dtype=self.torch.bool,
        )
        with self.torch.inference_mode(), self.torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            final_t, residual_t = self.model.final_scores(
                base_scores=base_t,
                raw_features=feat_t,
                candidate_use_hold=hold_t,
                mask=mask_t,
            )
        final_short = final_t[0].float().cpu().numpy()
        residual_short = residual_t[0].float().cpu().numpy()
        timings["future_rerank"] = (time.perf_counter() - t0) * 1000.0

        updated = list(candidates)
        for local_i, global_i in enumerate(shortlist):
            updated[global_i] = replace(
                updated[global_i],
                residual=float(residual_short[local_i]),
                final_score=float(final_short[local_i]),
            )
        candidates = tuple(updated)
        winner_local = int(np.argmax(final_short))
        winner_global = int(shortlist[winner_local])
        chosen = candidates[winner_global]

        branch = next(
            plan
            for plan in plans
            if plan.use_hold == chosen.use_hold and plan.mode == chosen.branch_mode
        )

        if resolve_exact_path:
            t0 = time.perf_counter()
            movement_path = _exact_path_for_geometry(
                board,
                chosen.state,
                self.config.reference_max_states,
            )
            timings["exact_path"] = (time.perf_counter() - t0) * 1000.0
        else:
            # Live closed-loop execution must retarget from the latest visual
            # active state.  Resolving a spawn-reference path here costs close
            # to one second on the current reference BFS and is stale by the
            # time it could be executed.
            movement_path = ()
            timings["exact_path_deferred"] = 0.0
        timings["total"] = (time.perf_counter() - total_start) * 1000.0

        return LiveDecision(
            observation_fingerprint=observation.fingerprint,
            active_piece=observation.active_piece,
            hold_piece=observation.hold_piece,
            preview_queue=observation.preview_queue,
            candidates=candidates,
            shortlist_indices=tuple(int(i) for i in shortlist),
            chosen_index=winner_global,
            branch=branch,
            movement_path=movement_path,
            timings_ms=timings,
        )
