from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any

import numpy as np

from tetrio.live_v1_1 import (
    BranchPlan,
    LiveCandidate,
    LiveDecision,
    LiveV11Policy,
    LiveV11PolicyConfig,
    _landing_states,
    _reference_best_placements,
    branch_plans,
)
from tetrio.vision.observation import ModelObservation




def observation_generation_key(observation: ModelObservation) -> tuple[str, str | None, tuple[str, ...]]:
    """Queue/HOLD identity for one live falling-piece generation.

    Deliberately ignores the locked-board fingerprint so a single falling piece
    is not re-planned just because the visual tracker briefly changes its board
    estimate while that same Active/Hold/NEXT generation is still on screen.
    A real lock advances NEXT (or a manual/AI HOLD changes Active/Hold), which
    yields a new key and permits one new plan.
    """
    return (
        str(observation.active_piece),
        None if observation.hold_piece is None else str(observation.hold_piece),
        tuple(str(x) for x in observation.preview_queue),
    )

@dataclass(frozen=True)
class SpeculationSeed:
    source_fingerprint: str
    source_active: str
    source_hold: str | None
    source_preview: tuple[str, ...]
    board_after: tuple[tuple[int, ...], ...]
    placed_piece: str
    use_hold: bool
    branch_mode: str


@dataclass(frozen=True)
class PredictedNextTemplate:
    source_fingerprint: str
    board: tuple[tuple[int, ...], ...]
    active_piece: str
    hold_piece: str | None
    preview_prefix: tuple[str, ...]
    missing_preview: int
    source_branch_mode: str

    def board_array(self) -> np.ndarray:
        return np.asarray(self.board, dtype=np.uint8)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["board"] = [list(row) for row in self.board]
        out["preview_prefix"] = list(self.preview_prefix)
        return out


@dataclass(frozen=True)
class PreparedCandidate:
    state: Any
    board_after: np.ndarray
    lines: int
    use_hold: bool
    branch_mode: str
    future_features: np.ndarray
    spawn_path: tuple[str, ...] | None

    def geometry_key(self) -> tuple:
        return self.state.geometry_key()


@dataclass(frozen=True)
class SpeculativePreparation:
    template: PredictedNextTemplate
    candidates: tuple[PreparedCandidate, ...]
    branch_plans: tuple[BranchPlan, BranchPlan]
    timings_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": self.template.to_dict(),
            "candidate_count": len(self.candidates),
            "branch_candidate_counts": {
                "no_hold": sum(not x.use_hold for x in self.candidates),
                "hold": sum(x.use_hold for x in self.candidates),
            },
            "path_ready_count": sum(x.spawn_path is not None for x in self.candidates),
            "timings_ms": {k: float(v) for k, v in self.timings_ms.items()},
        }


def make_speculation_seed(
    observation: ModelObservation,
    decision: LiveDecision,
) -> SpeculationSeed:
    chosen = decision.chosen
    return SpeculationSeed(
        source_fingerprint=observation.fingerprint,
        source_active=observation.active_piece,
        source_hold=observation.hold_piece,
        source_preview=observation.preview_queue,
        board_after=tuple(
            tuple(int(v) for v in row)
            for row in np.asarray(chosen.board_after, dtype=np.uint8)
        ),
        placed_piece=str(chosen.state.piece),
        use_hold=bool(chosen.use_hold),
        branch_mode=str(chosen.branch_mode),
    )


def predict_next_template(seed: SpeculationSeed) -> PredictedNextTemplate:
    from tetrio.future.state_transition import advance_after_lock

    future = advance_after_lock(
        active=seed.source_active,
        hold=seed.source_hold,
        preview=seed.source_preview,
        use_hold=seed.use_hold,
        placed_piece=seed.placed_piece,
    )
    prefix = tuple(future.preview)
    if len(prefix) < 3:
        raise RuntimeError(
            "Speculative next state exposes fewer than three known preview pieces; "
            "cannot precompute V1.1 future features safely."
        )
    if len(prefix) > 5:
        prefix = prefix[:5]
    return PredictedNextTemplate(
        source_fingerprint=seed.source_fingerprint,
        board=seed.board_after,
        active_piece=future.active,
        hold_piece=future.hold,
        preview_prefix=prefix,
        missing_preview=5 - len(prefix),
        source_branch_mode=seed.branch_mode,
    )


def _branch_plans_for_template(template: PredictedNextTemplate) -> tuple[BranchPlan, BranchPlan]:
    # branch_plans only consumes preview[0:2]; every supported template has >=3.
    proxy = type(
        "_ProxyObservation",
        (),
        {
            "active_piece": template.active_piece,
            "hold_piece": template.hold_piece,
            "preview_queue": template.preview_prefix,
        },
    )()
    return branch_plans(proxy)


def _spawn_path_map(board: np.ndarray, piece: str, max_states: int) -> dict[tuple, tuple[str, ...]]:
    out: dict[tuple, tuple[str, ...]] = {}
    for placement in _reference_best_placements(board, piece, max_states):
        key = placement.landing_state.geometry_key()
        path = tuple(action.value for action in placement.path)
        old = out.get(key)
        if old is None or (len(path), path) < (len(old), old):
            out[key] = path
    return out


def build_speculative_preparation(
    seed: SpeculationSeed,
    config: LiveV11PolicyConfig = LiveV11PolicyConfig(),
) -> SpeculativePreparation:
    """CPU-only next-state preparation. Safe to run in a worker process.

    It never fabricates unseen NEXT pieces. Candidate geometry, resulting boards,
    V1.1 future features and spawn-reference paths depend only on the predicted
    board plus the known preview prefix exposed by the current NEXT5.
    """
    from tetrio.future.lookahead import (
        FutureCandidateInput,
        FutureFeatureConfig,
        build_row_future_features,
    )
    from tetrio.ruleset import TETRIO_MOVEMENT
    from tetris_ai.core.movement import clear_lines, lock_piece

    total_start = time.perf_counter()
    template = predict_next_template(seed)
    board = template.board_array()
    plans = _branch_plans_for_template(template)

    t0 = time.perf_counter()
    skeleton: list[tuple[Any, np.ndarray, int, BranchPlan]] = []
    for plan in plans:
        landings = _landing_states(board, plan.selected_piece, config)
        for landing in landings:
            locked = lock_piece(board, landing, TETRIO_MOVEMENT)
            after, cleared = clear_lines(locked, TETRIO_MOVEMENT)
            skeleton.append(
                (landing, np.asarray(after, dtype=np.uint8), int(cleared), plan)
            )
    enumerate_ms = (time.perf_counter() - t0) * 1000.0
    if not skeleton:
        raise RuntimeError("Speculative preparation found no candidates")

    t0 = time.perf_counter()
    feature_inputs = tuple(
        FutureCandidateInput(
            board_after=after,
            piece=str(state.piece),
            rotation=int(state.rotation),
            x=int(state.x),
            y=int(state.y),
            use_hold=bool(plan.use_hold),
            lines=int(lines),
        )
        for state, after, lines, plan in skeleton
    )
    features = build_row_future_features(
        board_before=board,
        active=template.active_piece,
        hold=template.hold_piece,
        preview=template.preview_prefix,
        candidates=feature_inputs,
        config=FutureFeatureConfig(
            fast_max_states=config.fast_max_states,
            reference_max_states=config.reference_max_states,
            tactical_preview_depth=2,
            exact_immediate_t=False,
        ),
    )
    feature_ms = (time.perf_counter() - t0) * 1000.0

    # Do not precompute reference spawn paths for every candidate.  The live
    # controller will always re-observe the falling piece and retarget from its
    # *current* visual state to the chosen landing.  Precomputing all spawn
    # paths cost ~2 seconds per speculative state in the live gate and those
    # paths are stale by execution time.
    path_ms = 0.0

    candidates = tuple(
        PreparedCandidate(
            state=state,
            board_after=after,
            lines=lines,
            use_hold=bool(plan.use_hold),
            branch_mode=plan.mode,
            future_features=np.asarray(features[i], dtype=np.float32),
            spawn_path=None,
        )
        for i, (state, after, lines, plan) in enumerate(skeleton)
    )
    return SpeculativePreparation(
        template=template,
        candidates=candidates,
        branch_plans=plans,
        timings_ms={
            "enumerate_candidates": enumerate_ms,
            "future_features_all": feature_ms,
            "reference_paths": path_ms,
            "total": (time.perf_counter() - total_start) * 1000.0,
        },
    )


def template_match_reason(
    observation: ModelObservation,
    template: PredictedNextTemplate,
) -> str | None:
    if not np.array_equal(observation.board_array(), template.board_array()):
        return "board_mismatch"
    if observation.active_piece != template.active_piece:
        return "active_mismatch"
    if observation.hold_piece != template.hold_piece:
        return "hold_mismatch"
    prefix = template.preview_prefix
    if tuple(observation.preview_queue[: len(prefix)]) != prefix:
        return "preview_prefix_mismatch"
    return None


def observation_matches_template(
    observation: ModelObservation,
    template: PredictedNextTemplate,
) -> bool:
    return template_match_reason(observation, template) is None


def _score_prepared_branch(
    policy: LiveV11Policy,
    *,
    state_latent,
    branch_candidates: tuple[PreparedCandidate, ...],
) -> tuple[LiveCandidate, ...]:
    from tetrio.network.encoding import dense_candidate_batch, pack_board, piece_id

    if not branch_candidates:
        return ()
    piece = str(branch_candidates[0].state.piece)
    dense = dense_candidate_batch(
        np.stack([pack_board(x.board_after) for x in branch_candidates]),
        np.asarray([piece_id(piece)] * len(branch_candidates), dtype=np.uint8),
        np.asarray([x.state.rotation for x in branch_candidates], dtype=np.uint8),
        np.asarray([x.state.x for x in branch_candidates], dtype=np.int8),
        np.asarray([x.state.y for x in branch_candidates], dtype=np.int8),
        np.asarray([int(x.use_hold) for x in branch_candidates], dtype=np.uint8),
        np.asarray([x.lines for x in branch_candidates], dtype=np.uint8),
    )
    tensor = policy.torch.from_numpy(dense).to(
        device=policy.device,
        non_blocking=policy.device.type == "cuda",
    ).unsqueeze(0)
    with policy.torch.inference_mode(), policy.torch.autocast(
        device_type=policy.device.type,
        dtype=policy.amp_dtype,
        enabled=policy.device.type == "cuda",
    ):
        scores = policy.model.scorer.score_from_state_latent(state_latent, tensor)[0]
    scores_np = scores.float().cpu().numpy()
    return tuple(
        LiveCandidate(
            state=x.state,
            board_after=x.board_after,
            lines=x.lines,
            use_hold=x.use_hold,
            branch_mode=x.branch_mode,
            base_score=float(score),
        )
        for x, score in zip(branch_candidates, scores_np)
    )


def decide_prepared(
    policy: LiveV11Policy,
    observation: ModelObservation,
    prepared: SpeculativePreparation,
    *,
    resolve_exact_path: bool = True,
) -> LiveDecision:
    """Finish an exact V1.1 decision using precomputed CPU-heavy work."""
    from tetrio.tools.build_expert_v1_1_future_cache import select_inference_shortlist

    mismatch = template_match_reason(observation, prepared.template)
    if mismatch is not None:
        raise RuntimeError(f"prepared observation mismatch: {mismatch}")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    latent = policy._state_latent(observation)
    timings["state_encode"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    scored_parts: list[tuple[LiveCandidate, ...]] = []
    for use_hold in (False, True):
        branch = tuple(x for x in prepared.candidates if x.use_hold == use_hold)
        scored_parts.append(
            _score_prepared_branch(policy, state_latent=latent, branch_candidates=branch)
        )
    candidates = tuple(x for part in scored_parts for x in part)
    timings["prepared_base_score"] = (time.perf_counter() - t0) * 1000.0
    if len(candidates) != len(prepared.candidates):
        raise RuntimeError("prepared candidate cardinality drift")

    base_scores = np.asarray([x.base_score for x in candidates], dtype=np.float32)
    holds = np.asarray([int(x.use_hold) for x in candidates], dtype=np.uint8)
    shortlist = select_inference_shortlist(
        base_scores,
        holds,
        top_overall=policy.config.top_overall,
        top_per_branch=policy.config.top_per_branch,
    )
    if not shortlist:
        raise RuntimeError("prepared V1.1 shortlist is empty")

    t0 = time.perf_counter()
    features = np.stack(
        [prepared.candidates[i].future_features for i in shortlist], axis=0
    ).astype(np.float32, copy=False)
    timings["prepared_feature_lookup"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    base_t = policy.torch.from_numpy(base_scores[shortlist][None, :]).to(
        device=policy.device, dtype=policy.torch.float32
    )
    feat_t = policy.torch.from_numpy(features[None, :, :]).to(
        device=policy.device, dtype=policy.torch.float32
    )
    hold_t = policy.torch.from_numpy(holds[shortlist][None, :]).to(
        device=policy.device
    ).bool()
    mask_t = policy.torch.ones(
        (1, len(shortlist)), device=policy.device, dtype=policy.torch.bool
    )
    with policy.torch.inference_mode(), policy.torch.autocast(
        device_type=policy.device.type,
        dtype=policy.amp_dtype,
        enabled=policy.device.type == "cuda",
    ):
        final_t, residual_t = policy.model.final_scores(
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
        p
        for p in branch_plans(observation)
        if p.use_hold == chosen.use_hold and p.mode == chosen.branch_mode
    )
    if resolve_exact_path:
        t0 = time.perf_counter()
        movement_path = prepared.candidates[winner_global].spawn_path
        if movement_path is None:
            from tetrio.live_v1_1 import _exact_path_for_geometry
            movement_path = _exact_path_for_geometry(
                observation.board_array(),
                chosen.state,
                policy.config.reference_max_states,
            )
            timings["exact_path_fallback"] = (time.perf_counter() - t0) * 1000.0
        else:
            timings["exact_path_lookup"] = (time.perf_counter() - t0) * 1000.0
    else:
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
        movement_path=tuple(movement_path),
        timings_ms=timings,
    )


def synthetic_completed_observation(
    prepared: SpeculativePreparation,
    *,
    filler: tuple[str, ...] = ("I", "O", "T", "S", "Z", "J", "L"),
) -> ModelObservation:
    prefix = list(prepared.template.preview_prefix)
    i = 0
    while len(prefix) < 5:
        prefix.append(filler[i % len(filler)])
        i += 1
    preview = tuple(prefix[:5])
    # Fingerprint is diagnostic only for policy execution; construct a stable
    # synthetic tag without pretending the unseen pieces came from vision.
    return ModelObservation(
        frame_index=-1,
        board=prepared.template.board,
        active_piece=prepared.template.active_piece,
        hold_piece=prepared.template.hold_piece,
        preview_queue=preview,
        board_confidence=1.0,
        active_confidence=1.0,
        hold_confidence=1.0,
        preview_confidence=1.0,
        fingerprint="synthetic-prepared-parity",
    )


def warmup_live_policy(policy: LiveV11Policy) -> LiveDecision:
    """Pay import/CUDA/first-search costs before live tracking begins."""
    empty = tuple(tuple(0 for _ in range(10)) for _ in range(40))
    observation = ModelObservation(
        frame_index=-1,
        board=empty,
        active_piece="T",
        hold_piece="J",
        preview_queue=("I", "O", "S", "Z", "L"),
        board_confidence=1.0,
        active_confidence=1.0,
        hold_confidence=1.0,
        preview_confidence=1.0,
        fingerprint="warmup-empty-board",
    )
    return policy.decide(observation, resolve_exact_path=False)


def decisions_equivalent(
    a: LiveDecision,
    b: LiveDecision,
    atol: float = 1e-5,
    *,
    require_path: bool = True,
) -> bool:
    if a.chosen.state.geometry_key() != b.chosen.state.geometry_key():
        return False
    if bool(a.chosen.use_hold) != bool(b.chosen.use_hold):
        return False
    if a.branch.mode != b.branch.mode:
        return False
    if require_path and tuple(a.movement_path) != tuple(b.movement_path):
        return False
    if len(a.shortlist_indices) != len(b.shortlist_indices):
        return False
    return abs(float(a.chosen.final_score) - float(b.chosen.final_score)) <= atol
