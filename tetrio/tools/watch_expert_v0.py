from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import pathlib
from pathlib import Path
import random
import time
from typing import Iterable

import numpy as np
import torch

from tetrio.fast_reachability import enumerate_tetrio_reachable_geometries_fast
from tetrio.network.encoding import (
    PREVIEW_DEPTH,
    dense_candidate_batch,
    dense_state_batch,
    pack_board,
    piece_id,
)
from tetrio.network.model import TetrioExpertV0Network
from tetrio.reachability import (
    enumerate_tetrio_reachable_placements,
    tetrio_spawn_state,
)
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.tetrominoes import occupied_cells
from tetris_ai.core.types import PieceState


PIECES = ("I", "O", "T", "S", "Z", "J", "L")

TETROMINO_COLORS = {
    "I": (0, 240, 240),
    "J": (0, 80, 240),
    "L": (240, 160, 0),
    "O": (240, 240, 0),
    "S": (0, 220, 0),
    "T": (160, 0, 240),
    "Z": (240, 0, 0),
}
VIS_PIECE_ID = {
    "I": 2,
    "O": 3,
    "T": 4,
    "S": 5,
    "Z": 6,
    "J": 7,
    "L": 8,
}
VIS_ID_TO_PIECE = {v: k for k, v in VIS_PIECE_ID.items()}

BG = (18, 20, 24)
PANEL_BG = (27, 30, 36)
PANEL_BORDER = (64, 69, 80)
GRID = (48, 53, 62)
EMPTY = (21, 24, 29)
TEXT = (232, 235, 240)
MUTED = (160, 166, 178)
GOOD = (105, 222, 143)
WARN = (255, 194, 88)
BAD = (255, 112, 112)

SPEED_PRESETS = {
    1: 0.5,
    2: 1.0,
    3: 2.0,
    4: 4.0,
    5: 6.0,
    6: 10.0,
    7: 15.0,
    8: 25.0,
    9: 60.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Autonomous TETR.IO Expert-v0 rollout. The model chooses Hold, "
            "ranks the reachable placement branch, locks its Top-1, and repeats "
            "until top-out or max-pieces."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(r"models\tetrio_expert_v0_full.pt"),
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument(
        "--seeds",
        default="",
        help=(
            "Headless seed list/range, e.g. 9001-9020 or 9001,9003,9005. "
            "If empty, --seed is used."
        ),
    )
    p.add_argument("--max-pieces", type=int, default=5000)
    p.add_argument(
        "--backend",
        choices=("fast", "reference"),
        default="fast",
        help="Fast geometry backend for interactive rollout; reference is slower/path-sensitive.",
    )
    p.add_argument("--fast-max-states", type=int, default=10_000)
    p.add_argument("--reference-max-states", type=int, default=50_000)
    p.add_argument(
        "--reference-audit-every",
        type=int,
        default=250,
        help=(
            "When backend=fast, compare the complete candidate geometry set "
            "against reference every N pieces. 0 disables."
        ),
    )
    p.add_argument("--hold-threshold", type=float, default=0.5)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--speed", type=float, default=6.0)
    p.add_argument(
        "--no-fall-animation",
        action="store_true",
        help="Disable the V3.4-style purely visual falling animation.",
    )
    p.add_argument("--width", type=int, default=1220)
    p.add_argument("--height", type=int, default=900)
    p.add_argument(
        "--save-json",
        type=Path,
        default=Path(r"artifacts\tetrio\expert_v0_autonomous_rollout.json"),
    )
    return p.parse_args()


def parse_seed_spec(spec: str, fallback: int) -> list[int]:
    spec = str(spec or "").strip()
    if not spec:
        return [int(fallback)]
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("No seeds parsed from --seeds")
    return out


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[TetrioExpertV0Network, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    safe_path_globals = [
        pathlib.Path,
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    ]
    with torch.serialization.safe_globals(safe_path_globals):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    if checkpoint.get("format") != "tetrio_expert_v0":
        raise RuntimeError(
            f"Unsupported checkpoint format: {checkpoint.get('format')!r}"
        )

    model = TetrioExpertV0Network().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


class SevenBagQueue:
    """Deterministic 7-bag stream independent from Python global RNG."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.queue: deque[str] = deque()

    def _refill(self) -> None:
        bag = list(PIECES)
        self.rng.shuffle(bag)
        self.queue.extend(bag)

    def ensure(self, n: int) -> None:
        while len(self.queue) < int(n):
            self._refill()

    def pop(self) -> str:
        self.ensure(1)
        return self.queue.popleft()

    def peek(self, n: int) -> tuple[str, ...]:
        self.ensure(n)
        return tuple(list(self.queue)[:n])


@dataclass(frozen=True)
class BranchPlan:
    selected_piece: str
    use_hold: bool
    hold_after: str | None
    next_active: str
    consume_count: int
    mode: str


@dataclass(frozen=True)
class ScoredCandidate:
    state: PieceState
    board_after: np.ndarray
    lines: int
    score: float


@dataclass(frozen=True)
class Decision:
    active: str
    hold_before: str | None
    preview_before: tuple[str, ...]
    hold_probability: float
    use_hold: bool
    branch: BranchPlan
    candidates: tuple[ScoredCandidate, ...]
    chosen_index: int
    audited_reference: bool = False
    holes_before: int = 0
    chosen_holes_after: int = 0
    chosen_hole_delta: int = 0
    min_candidate_holes: int = 0
    safer_candidate_index: int | None = None

    @property
    def chosen(self) -> ScoredCandidate:
        return self.candidates[self.chosen_index]

    @property
    def avoidable_hole(self) -> bool:
        return self.chosen_hole_delta > 0 and self.min_candidate_holes <= self.holes_before

    @property
    def top3(self) -> tuple[tuple[int, ScoredCandidate], ...]:
        order = sorted(
            range(len(self.candidates)),
            key=lambda i: self.candidates[i].score,
            reverse=True,
        )
        return tuple((i, self.candidates[i]) for i in order[:3])


@dataclass
class RolloutStats:
    pieces: int = 0
    line_counts: dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0}
    )
    holds: int = 0
    current_height: int = 0
    max_height: int = 0
    current_holes: int = 0
    max_holes: int = 0
    height_sum: float = 0.0
    candidate_sum: int = 0
    fast_reference_audits: int = 0
    fast_reference_fallbacks: int = 0
    hole_creation_moves: int = 0
    avoidable_hole_moves: int = 0

    @property
    def lines(self) -> int:
        return sum(k * v for k, v in self.line_counts.items())

    @property
    def tetrises(self) -> int:
        return self.line_counts[4]

    @property
    def hold_rate(self) -> float:
        return 0.0 if self.pieces == 0 else self.holds / self.pieces

    @property
    def avg_height(self) -> float:
        return 0.0 if self.pieces == 0 else self.height_sum / self.pieces

    @property
    def avg_candidates(self) -> float:
        return 0.0 if self.pieces == 0 else self.candidate_sum / self.pieces


def board_metrics(board: np.ndarray) -> tuple[int, int]:
    arr = (np.asarray(board).reshape(40, 10) != 0)
    occupied_rows = np.where(np.any(arr, axis=1))[0]
    height = 0 if occupied_rows.size == 0 else int(40 - occupied_rows[0])

    holes = 0
    for x in range(10):
        col = arr[:, x]
        filled = np.where(col)[0]
        if filled.size:
            holes += int(np.count_nonzero(~col[int(filled[0]):]))
    return height, holes


def _reference_landings(
    board: np.ndarray,
    piece: str,
    max_states: int,
) -> list[PieceState]:
    best = {}
    for placement in enumerate_tetrio_reachable_placements(
        board,
        piece,
        max_states=max_states,
    ):
        key = placement.landing_state.geometry_key()
        old = best.get(key)
        if old is None or len(placement.path) < len(old.path):
            best[key] = placement
    return [
        p.landing_state
        for p in sorted(
            best.values(),
            key=lambda p: (
                p.landing_state.rotation % 4,
                p.landing_state.x,
                p.landing_state.y,
                len(p.path),
            ),
        )
    ]


def _geometry_set(states: Iterable[PieceState]) -> set[tuple]:
    return {s.geometry_key() for s in states}


def _clear_visual_rows(
    locked_binary: np.ndarray,
    locked_visual: np.ndarray,
) -> np.ndarray:
    full = np.all(np.asarray(locked_binary) != 0, axis=1)
    count = int(np.count_nonzero(full))
    if count == 0:
        return locked_visual.copy()
    kept = locked_visual[~full]
    zeros = np.zeros((count, locked_visual.shape[1]), dtype=np.uint8)
    return np.vstack((zeros, kept))


class ExpertV0Rollout:
    def __init__(
        self,
        model: TetrioExpertV0Network,
        *,
        device: torch.device,
        seed: int,
        max_pieces: int,
        backend: str,
        fast_max_states: int,
        reference_max_states: int,
        reference_audit_every: int,
        hold_threshold: float,
    ):
        self.model = model
        self.device = device
        self.seed = int(seed)
        self.max_pieces = int(max_pieces)
        self.backend = str(backend)
        self.fast_max_states = int(fast_max_states)
        self.reference_max_states = int(reference_max_states)
        self.reference_audit_every = int(reference_audit_every)
        self.hold_threshold = float(hold_threshold)

        if not 0.0 <= self.hold_threshold <= 1.0:
            raise ValueError("hold_threshold must be in [0,1]")

        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif self.device.type == "cuda":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.stream = SevenBagQueue(self.seed)
        self.board = TETRIO_MOVEMENT.empty_board()
        self.visual_board = np.zeros((40, 10), dtype=np.uint8)
        self.active = self.stream.pop()
        self.hold: str | None = None
        self.stats = RolloutStats()
        self.game_over = False
        self.terminal_reason = ""
        self.last_decision: Decision | None = None
        self.pending_decision: Decision | None = None
        self.risk_events: list[dict] = []
        self._ensure_preview()
        self.refresh_decision()

    def _ensure_preview(self) -> None:
        # Need two queue pieces when Hold is empty: preview[0] becomes the held-
        # branch piece and preview[1] becomes the following active piece.
        self.stream.ensure(PREVIEW_DEPTH + 2)

    def preview(self) -> tuple[str, ...]:
        self._ensure_preview()
        return self.stream.peek(PREVIEW_DEPTH)

    def _plan_branch(self, use_hold: bool) -> BranchPlan:
        self._ensure_preview()
        future = self.stream.peek(PREVIEW_DEPTH + 2)

        if not use_hold:
            return BranchPlan(
                selected_piece=self.active,
                use_hold=False,
                hold_after=self.hold,
                next_active=future[0],
                consume_count=1,
                mode="no_hold",
            )

        if self.hold is None:
            return BranchPlan(
                selected_piece=future[0],
                use_hold=True,
                hold_after=self.active,
                next_active=future[1],
                consume_count=2,
                mode="hold_empty",
            )

        return BranchPlan(
            selected_piece=self.hold,
            use_hold=True,
            hold_after=self.active,
            next_active=future[0],
            consume_count=1,
            mode="hold_swap",
        )

    def _state_tensor(self) -> torch.Tensor:
        preview = self.preview()
        dense = dense_state_batch(
            np.stack([pack_board(self.board)]),
            np.asarray([piece_id(self.active)], dtype=np.uint8),
            np.asarray([piece_id(self.hold)], dtype=np.uint8),
            np.asarray(
                [[piece_id(p) for p in preview]],
                dtype=np.uint8,
            ),
        )
        return torch.from_numpy(dense).to(
            device=self.device,
            non_blocking=self.device.type == "cuda",
        )

    def _hold_probability(
        self,
        state_tensor: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            latent = self.model.scorer.encode_state(state_tensor)
            logit = self.model.hold_head(latent).squeeze(-1)
        return float(torch.sigmoid(logit.float())[0].item()), latent

    def _enumerate_landings(
        self,
        piece: str,
        *,
        audit: bool,
    ) -> tuple[list[PieceState], bool]:
        if self.backend == "reference":
            return (
                _reference_landings(
                    self.board,
                    piece,
                    self.reference_max_states,
                ),
                False,
            )

        fast = enumerate_tetrio_reachable_geometries_fast(
            self.board,
            piece,
            max_states=self.fast_max_states,
        )

        if not fast:
            reference = _reference_landings(
                self.board,
                piece,
                self.reference_max_states,
            )
            if reference:
                self.stats.fast_reference_fallbacks += 1
                return reference, True
            return [], True

        if audit:
            reference = _reference_landings(
                self.board,
                piece,
                self.reference_max_states,
            )
            self.stats.fast_reference_audits += 1
            fast_keys = _geometry_set(fast)
            ref_keys = _geometry_set(reference)
            if fast_keys != ref_keys:
                only_fast = sorted(fast_keys - ref_keys)[:8]
                only_ref = sorted(ref_keys - fast_keys)[:8]
                raise RuntimeError(
                    "FAST/REFERENCE ROLLOUT PARITY FAILURE "
                    f"seed={self.seed} piece_index={self.stats.pieces} "
                    f"piece={piece} fast={len(fast_keys)} ref={len(ref_keys)} "
                    f"only_fast={only_fast} only_ref={only_ref}"
                )
            return fast, True

        return fast, False

    def _build_scored_candidates(
        self,
        branch: BranchPlan,
        state_latent: torch.Tensor,
        *,
        audit: bool,
    ) -> tuple[tuple[ScoredCandidate, ...], bool]:
        landings, audited = self._enumerate_landings(
            branch.selected_piece,
            audit=audit,
        )
        if not landings:
            return (), audited

        boards = []
        lines = []
        for landing in landings:
            locked = lock_piece(self.board, landing, TETRIO_MOVEMENT)
            after, cleared = clear_lines(locked, TETRIO_MOVEMENT)
            boards.append(after)
            lines.append(int(cleared))

        candidate_dense = dense_candidate_batch(
            np.stack([pack_board(b) for b in boards]),
            np.asarray([piece_id(branch.selected_piece)] * len(landings), dtype=np.uint8),
            np.asarray([s.rotation for s in landings], dtype=np.uint8),
            np.asarray([s.x for s in landings], dtype=np.int8),
            np.asarray([s.y for s in landings], dtype=np.int8),
            np.asarray([int(branch.use_hold)] * len(landings), dtype=np.uint8),
            np.asarray(lines, dtype=np.uint8),
        )
        candidate_tensor = torch.from_numpy(candidate_dense).to(
            device=self.device,
            non_blocking=self.device.type == "cuda",
        ).unsqueeze(0)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.device.type == "cuda",
        ):
            scores = self.model.scorer.score_from_state_latent(
                state_latent,
                candidate_tensor,
            )[0]

        scores_np = scores.float().cpu().numpy()
        candidates = tuple(
            ScoredCandidate(
                state=landing,
                board_after=board,
                lines=line_count,
                score=float(score),
            )
            for landing, board, line_count, score in zip(
                landings,
                boards,
                lines,
                scores_np,
            )
        )
        return candidates, audited

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
        hold_probability, state_latent = self._hold_probability(state_tensor)
        use_hold = hold_probability >= self.hold_threshold
        branch = self._plan_branch(use_hold)

        audit = (
            self.backend == "fast"
            and self.reference_audit_every > 0
            and self.stats.pieces % self.reference_audit_every == 0
        )
        candidates, audited = self._build_scored_candidates(
            branch,
            state_latent,
            audit=audit,
        )

        if not candidates:
            self.game_over = True
            self.terminal_reason = (
                f"NO_REACHABLE_PLACEMENTS branch={branch.mode} "
                f"piece={branch.selected_piece}"
            )
            self.pending_decision = None
            return None

        chosen_index = int(
            np.argmax(np.asarray([c.score for c in candidates], dtype=np.float32))
        )

        _, holes_before = board_metrics(self.board)
        candidate_holes = [board_metrics(c.board_after)[1] for c in candidates]
        min_candidate_holes = min(candidate_holes)
        chosen_holes_after = int(candidate_holes[chosen_index])

        safer_index = None
        if min_candidate_holes < chosen_holes_after:
            # Among the structurally safer candidates, show the one the network
            # itself scores highest. This is diagnostic only; policy is unchanged.
            safer_pool = [
                i for i, holes in enumerate(candidate_holes)
                if holes == min_candidate_holes
            ]
            safer_index = max(
                safer_pool,
                key=lambda i: candidates[i].score,
            )

        decision = Decision(
            active=self.active,
            hold_before=self.hold,
            preview_before=self.preview(),
            hold_probability=hold_probability,
            use_hold=use_hold,
            branch=branch,
            candidates=candidates,
            chosen_index=chosen_index,
            audited_reference=audited,
            holes_before=int(holes_before),
            chosen_holes_after=chosen_holes_after,
            chosen_hole_delta=chosen_holes_after - int(holes_before),
            min_candidate_holes=int(min_candidate_holes),
            safer_candidate_index=safer_index,
        )
        self.pending_decision = decision
        return decision

    def _visual_after_choice(self, chosen: ScoredCandidate) -> np.ndarray:
        locked_binary = lock_piece(self.board, chosen.state, TETRIO_MOVEMENT)
        locked_visual = self.visual_board.copy()
        visual_id = VIS_PIECE_ID[chosen.state.piece]
        for x, y in occupied_cells(chosen.state):
            if 0 <= y < 40 and 0 <= x < 10:
                locked_visual[y, x] = visual_id
        return _clear_visual_rows(locked_binary, locked_visual)

    def step(self) -> bool:
        if self.game_over:
            return False
        decision = self.pending_decision or self.refresh_decision()
        if decision is None:
            return False

        chosen = decision.chosen
        self.visual_board = self._visual_after_choice(chosen)
        self.board = chosen.board_after.copy()

        if decision.use_hold:
            self.stats.holds += 1
        self.stats.pieces += 1
        self.stats.candidate_sum += len(decision.candidates)

        if decision.chosen_hole_delta > 0:
            self.stats.hole_creation_moves += 1
        if decision.avoidable_hole:
            self.stats.avoidable_hole_moves += 1
            safer = (
                None
                if decision.safer_candidate_index is None
                else decision.candidates[decision.safer_candidate_index]
            )
            self.risk_events.append(
                {
                    "piece_index": self.stats.pieces,
                    "active": decision.active,
                    "hold_before": decision.hold_before,
                    "preview": list(decision.preview_before),
                    "hold_probability": decision.hold_probability,
                    "use_hold": decision.use_hold,
                    "branch_mode": decision.branch.mode,
                    "chosen": {
                        "piece": chosen.state.piece,
                        "rotation": int(chosen.state.rotation) % 4,
                        "x": int(chosen.state.x),
                        "y": int(chosen.state.y),
                        "score": float(chosen.score),
                        "lines": int(chosen.lines),
                        "holes_after": int(decision.chosen_holes_after),
                    },
                    "holes_before": int(decision.holes_before),
                    "hole_delta": int(decision.chosen_hole_delta),
                    "min_candidate_holes": int(decision.min_candidate_holes),
                    "safer_alternative": None if safer is None else {
                        "piece": safer.state.piece,
                        "rotation": int(safer.state.rotation) % 4,
                        "x": int(safer.state.x),
                        "y": int(safer.state.y),
                        "score": float(safer.score),
                        "lines": int(safer.lines),
                        "holes_after": int(decision.min_candidate_holes),
                        "score_gap_vs_chosen": float(chosen.score - safer.score),
                    },
                }
            )
        if chosen.lines in self.stats.line_counts:
            self.stats.line_counts[chosen.lines] += 1

        # Commit the precomputed Hold/queue transition only after a legal
        # placement is selected and locked.
        consumed = [self.stream.pop() for _ in range(decision.branch.consume_count)]
        if decision.branch.mode == "hold_empty":
            if consumed[0] != decision.branch.selected_piece:
                raise RuntimeError("7-bag hold-empty selected-piece drift")
            if consumed[1] != decision.branch.next_active:
                raise RuntimeError("7-bag hold-empty next-active drift")
        else:
            if consumed[0] != decision.branch.next_active:
                raise RuntimeError("7-bag next-active drift")

        self.hold = decision.branch.hold_after
        self.active = decision.branch.next_active
        self.last_decision = decision

        height, holes = board_metrics(self.board)
        self.stats.current_height = height
        self.stats.max_height = max(self.stats.max_height, height)
        self.stats.current_holes = holes
        self.stats.max_holes = max(self.stats.max_holes, holes)
        self.stats.height_sum += height

        self._ensure_preview()
        self.refresh_decision()
        return True

    def run_to_end(self) -> dict:
        while not self.game_over:
            if not self.step():
                break
        return self.result()

    def result(self) -> dict:
        return {
            "seed": self.seed,
            "pieces": self.stats.pieces,
            "lines": self.stats.lines,
            "singles": self.stats.line_counts[1],
            "doubles": self.stats.line_counts[2],
            "triples": self.stats.line_counts[3],
            "tetrises": self.stats.tetrises,
            "holds": self.stats.holds,
            "hold_rate": self.stats.hold_rate,
            "avg_height": self.stats.avg_height,
            "max_height": self.stats.max_height,
            "holes": self.stats.current_holes,
            "max_holes": self.stats.max_holes,
            "avg_candidates": self.stats.avg_candidates,
            "fast_reference_audits": self.stats.fast_reference_audits,
            "fast_reference_fallbacks": self.stats.fast_reference_fallbacks,
            "hole_creation_moves": self.stats.hole_creation_moves,
            "avoidable_hole_moves": self.stats.avoidable_hole_moves,
            "hole_creation_rate": (
                0.0 if self.stats.pieces == 0
                else self.stats.hole_creation_moves / self.stats.pieces
            ),
            "avoidable_hole_rate": (
                0.0 if self.stats.pieces == 0
                else self.stats.avoidable_hole_moves / self.stats.pieces
            ),
            "risk_events": list(self.risk_events),
            "game_over": self.game_over and self.terminal_reason != "LIMIT",
            "terminal_reason": self.terminal_reason,
        }


def aggregate_results(results: list[dict]) -> dict:
    if not results:
        return {}
    numeric = (
        "pieces",
        "lines",
        "tetrises",
        "hold_rate",
        "avg_height",
        "max_height",
        "holes",
        "max_holes",
        "avg_candidates",
        "hole_creation_rate",
        "avoidable_hole_rate",
    )
    return {
        "games": len(results),
        "game_overs": sum(bool(r["game_over"]) for r in results),
        **{
            f"mean_{key}": float(np.mean([float(r[key]) for r in results]))
            for key in numeric
        },
        "min_pieces": min(int(r["pieces"]) for r in results),
        "max_pieces": max(int(r["pieces"]) for r in results),
    }


def run_headless(
    model: TetrioExpertV0Network,
    checkpoint: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    seeds = parse_seed_spec(args.seeds, args.seed)
    results = []

    print("=" * 108)
    print("TETR.IO EXPERT V0 — AUTONOMOUS ROLLOUT")
    print("=" * 108)
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Best epoch   : {checkpoint.get('epoch')}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"Backend      : {args.backend}")
    print(f"Seeds        : {seeds[0]}..{seeds[-1]} ({len(seeds)} game(s))")
    print(f"Max pieces   : {args.max_pieces}")
    print(f"Hold threshold: {args.hold_threshold:.3f}")
    print()

    for seed in seeds:
        session = ExpertV0Rollout(
            model,
            device=device,
            seed=seed,
            max_pieces=args.max_pieces,
            backend=args.backend,
            fast_max_states=args.fast_max_states,
            reference_max_states=args.reference_max_states,
            reference_audit_every=args.reference_audit_every,
            hold_threshold=args.hold_threshold,
        )
        result = session.run_to_end()
        results.append(result)
        print(
            f"Seed {seed}: pieces={result['pieces']:,} "
            f"lines={result['lines']:,} T={result['tetrises']:,} "
            f"hold={result['hold_rate']:.3f} "
            f"avgH={result['avg_height']:.2f} maxH={result['max_height']} "
            f"holes={result['holes']} maxHoles={result['max_holes']} "
            f"newHole={result['hole_creation_moves']} "
            f"avoidable={result['avoidable_hole_moves']} "
            f"reason={result['terminal_reason']}"
        )

    report = {
        "format": "tetrio_expert_v0_autonomous_rollout",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "backend": args.backend,
        "max_pieces": args.max_pieces,
        "hold_threshold": args.hold_threshold,
        "reference_audit_every": args.reference_audit_every,
        "results": results,
        "aggregate": aggregate_results(results),
        "status": "DEVELOPMENT ROLLOUT (not Champion qualification)",
    }
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Aggregate:")
    for key, value in report["aggregate"].items():
        if isinstance(value, float):
            print(f"  {key:<24} {value:.4f}")
        else:
            print(f"  {key:<24} {value}")
    print(f"Report       : {args.save_json}")



@dataclass(frozen=True)
class VisualDrop:
    """
    Purely visual V3.4-lineage placement animation.

    Important display invariant:
    TETR.IO's real entry position is in hidden rows (around y=17).  The previous
    reconstruction rotated while the piece was still hidden, so the user only
    saw an already-rotated T appear.

    V2 deliberately stages the piece at the top visible row before rotating:

        hidden entry
          -> enter visible field in spawn orientation
          -> visible 90-degree rotation step(s)
          -> visible horizontal move
          -> accelerating vertical fall
          -> exact model-selected landing

    This is a visualization only. It never changes policy/reachability/score.
    """
    piece: str
    target_rotation: int
    target_x: int
    landing_y: int
    display_hold: str | None
    display_preview: tuple[str, ...]
    started_at: float
    duration: float

    def progress(self, now: float | None = None) -> float:
        if now is None:
            now = time.perf_counter()
        if self.duration <= 0.0:
            return 1.0
        return max(
            0.0,
            min(1.0, (float(now) - self.started_at) / self.duration),
        )

    @staticmethod
    def _rotation_steps(target_rotation: int) -> tuple[int, ...]:
        """
        Return a visually readable shortest rotation sequence from spawn r0.

        r1: 0 -> 1
        r2: 0 -> 1 -> 2
        r3: 0 -> 3

        TETR.IO supports 180, but showing r2 as two visible quarter-turns is
        easier to read than an instantaneous 180-degree flip.
        """
        target = int(target_rotation) % 4
        if target == 0:
            return (0,)
        if target == 1:
            return (0, 1)
        if target == 2:
            return (0, 1, 2)
        return (0, 3)

    def pose(self, now: float | None = None) -> PieceState:
        p = self.progress(now)
        spawn = tetrio_spawn_state(self.piece)
        target_rot = int(self.target_rotation) % 4

        # Make rotation visible. TETR.IO entry is above the rendered 20 rows,
        # so bring the piece to a safe visual staging row first.
        visible_stage_y = max(
            20,
            min(
                22,
                int(self.landing_y) - 2
                if int(self.landing_y) >= 22
                else 20,
            ),
        )

        # Four phases:
        #  0.00..0.18 enter visible field, no rotation
        #  0.18..0.48 rotate visibly
        #  0.48..0.66 horizontal adjustment
        #  0.66..1.00 vertical fall
        enter_end = 0.18
        rotate_end = 0.48
        move_end = 0.66

        if p < enter_end:
            a = p / enter_end
            y = int(
                round(
                    spawn.y
                    + (visible_stage_y - spawn.y) * a
                )
            )
            return PieceState(
                piece=self.piece,
                x=int(spawn.x),
                y=y,
                rotation=0,
            )

        if p < rotate_end:
            a = (p - enter_end) / (rotate_end - enter_end)
            steps = self._rotation_steps(target_rot)

            if len(steps) == 1:
                rot = steps[0]
            else:
                # Give each 90-degree orientation a real dwell interval rather
                # than blending/snap-rounding through hidden rows.
                segment_count = len(steps) - 1
                segment = min(
                    segment_count - 1,
                    int(a * segment_count),
                )
                local = (
                    a * segment_count - segment
                    if segment_count > 0
                    else 1.0
                )

                # Change orientation halfway through the segment. This creates
                # visible "before turn" and "after turn" frames at 60 FPS.
                rot = (
                    steps[segment]
                    if local < 0.5
                    else steps[segment + 1]
                )

            return PieceState(
                piece=self.piece,
                x=int(spawn.x),
                y=visible_stage_y,
                rotation=int(rot) % 4,
            )

        if p < move_end:
            a = (p - rotate_end) / (move_end - rotate_end)
            x = int(
                round(
                    spawn.x
                    + (int(self.target_x) - spawn.x) * a
                )
            )
            return PieceState(
                piece=self.piece,
                x=x,
                y=visible_stage_y,
                rotation=target_rot,
            )

        a = (p - move_end) / (1.0 - move_end)
        eased = a * a
        y = int(
            round(
                visible_stage_y
                + (int(self.landing_y) - visible_stage_y) * eased
            )
        )
        return PieceState(
            piece=self.piece,
            x=int(self.target_x),
            y=y,
            rotation=target_rot,
        )


@dataclass(frozen=True)
class ReviewBranch:
    mode: str


@dataclass(frozen=True)
class ReviewCandidate:
    state: PieceState
    lines: int
    score: float


@dataclass(frozen=True)
class ReviewDecision:
    hold_probability: float
    use_hold: bool
    branch: ReviewBranch
    chosen: ReviewCandidate
    top3_items: tuple[ReviewCandidate, ...]
    candidate_count: int
    audited_reference: bool
    holes_before: int
    chosen_holes_after: int
    chosen_hole_delta: int
    min_candidate_holes: int
    safer_candidate: ReviewCandidate | None

    @property
    def avoidable_hole(self) -> bool:
        return (
            self.chosen_hole_delta > 0
            and self.min_candidate_holes <= self.holes_before
        )

    @property
    def top3(self):
        return tuple(
            (i, candidate)
            for i, candidate in enumerate(self.top3_items)
        )


@dataclass(frozen=True)
class ReviewSession:
    visual_board: np.ndarray
    active: str
    hold: str | None
    preview_pieces: tuple[str, ...]
    stats: RolloutStats
    pending_decision: ReviewDecision | None
    game_over: bool
    terminal_reason: str
    backend: str
    seed: int

    def preview(self) -> tuple[str, ...]:
        return self.preview_pieces


class RolloutViewer:
    VIEWER_TITLE = (
        "Tetris Learning AI - TETR.IO Expert v0 Autonomous "
        "V3.4 Visible Rotation + History Replay"
    )
    PANEL_TITLE = "EXPERT v0 AUTONOMOUS"
    RESULT_FORMAT = "tetrio_expert_v0_autonomous_rollout_gui"
    SCREENSHOT_DIR = Path(
        r"artifacts\tetrio\expert_v0_rollout_screenshots"
    )

    def __init__(
        self,
        session: ExpertV0Rollout,
        *,
        width: int,
        height: int,
        speed: float,
        save_json: Path,
        fall_animation: bool = True,
    ):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is required.\n"
                r"Install with: .venv\Scripts\python.exe -m pip install pygame"
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption(self.VIEWER_TITLE)
        self.screen = pygame.display.set_mode(
            (width, height),
            pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_big = pygame.font.Font(None, 31)

        self.session = session
        self.speed = max(0.1, float(speed))
        self.playing = False
        self.running = True
        self.show_detail = True
        self.fall_animation = bool(fall_animation)

        self.visual_drop: VisualDrop | None = None
        self.manual_step_animation = False

        # When browsing history, Right/Next replays the placement animation
        # between history[i] and history[i+1] instead of instantly jumping.
        self.review_animation_source: ReviewSession | None = None
        self.review_animation_target_index: int | None = None

        self.last_step_ms = pygame.time.get_ticks()
        self.save_json = save_json
        self.buttons: list[tuple[object, str]] = []

        # History is display-only. It never rewinds the model/RNG and therefore
        # cannot change rollout semantics.  Prev/Next simply review committed
        # states; autoplay always resumes from the latest live state.
        self.history: list[ReviewSession] = []
        self.history_index = 0
        self._append_history_frame()

    def _brighten(self, color, amount=42):
        return tuple(min(255, int(c) + amount) for c in color)

    def _darken(self, color, factor=0.58):
        return tuple(max(0, int(c * factor)) for c in color)

    def _text(self, text, x, y, *, font=None, color=TEXT):
        font = font or self.font
        self.screen.blit(font.render(str(text), True, color), (x, y))

    def _draw_block(self, rect, color):
        pygame = self.pygame
        pygame.draw.rect(
            self.screen,
            self._darken(color, 0.52),
            rect,
            border_radius=2,
        )
        inner = rect.inflate(-3, -3)
        pygame.draw.rect(
            self.screen,
            color,
            inner,
            border_radius=2,
        )
        if inner.width >= 8 and inner.height >= 8:
            hi = pygame.Rect(
                inner.x + 2,
                inner.y + 2,
                max(2, inner.width - 4),
                max(2, inner.height // 5),
            )
            pygame.draw.rect(
                self.screen,
                self._brighten(color, 38),
                hi,
                border_radius=1,
            )

    @staticmethod
    def _copy_stats(stats: RolloutStats) -> RolloutStats:
        return RolloutStats(
            pieces=int(stats.pieces),
            line_counts=dict(stats.line_counts),
            holds=int(stats.holds),
            current_height=int(stats.current_height),
            max_height=int(stats.max_height),
            current_holes=int(stats.current_holes),
            max_holes=int(stats.max_holes),
            height_sum=float(stats.height_sum),
            candidate_sum=int(stats.candidate_sum),
            fast_reference_audits=int(stats.fast_reference_audits),
            fast_reference_fallbacks=int(stats.fast_reference_fallbacks),
            hole_creation_moves=int(
                getattr(stats, "hole_creation_moves", 0)
            ),
            avoidable_hole_moves=int(
                getattr(stats, "avoidable_hole_moves", 0)
            ),
        )

    @staticmethod
    def _candidate_summary(candidate) -> ReviewCandidate:
        return ReviewCandidate(
            state=PieceState(
                piece=str(candidate.state.piece),
                x=int(candidate.state.x),
                y=int(candidate.state.y),
                rotation=int(candidate.state.rotation) % 4,
            ),
            lines=int(candidate.lines),
            score=float(candidate.score),
        )

    def _decision_summary(
        self,
        decision: Decision | None,
    ) -> ReviewDecision | None:
        if decision is None:
            return None

        chosen = self._candidate_summary(decision.chosen)
        top3 = tuple(
            self._candidate_summary(c)
            for _, c in decision.top3
        )
        safer = None
        if decision.safer_candidate_index is not None:
            safer = self._candidate_summary(
                decision.candidates[decision.safer_candidate_index]
            )

        return ReviewDecision(
            hold_probability=float(decision.hold_probability),
            use_hold=bool(decision.use_hold),
            branch=ReviewBranch(mode=str(decision.branch.mode)),
            chosen=chosen,
            top3_items=top3,
            candidate_count=len(decision.candidates),
            audited_reference=bool(decision.audited_reference),
            holes_before=int(decision.holes_before),
            chosen_holes_after=int(decision.chosen_holes_after),
            chosen_hole_delta=int(decision.chosen_hole_delta),
            min_candidate_holes=int(decision.min_candidate_holes),
            safer_candidate=safer,
        )

    def _snapshot_live(self) -> ReviewSession:
        return ReviewSession(
            visual_board=self.session.visual_board.copy(),
            active=str(self.session.active),
            hold=(
                None
                if self.session.hold is None
                else str(self.session.hold)
            ),
            preview_pieces=tuple(self.session.preview()),
            stats=self._copy_stats(self.session.stats),
            pending_decision=self._decision_summary(
                self.session.pending_decision
            ),
            game_over=bool(self.session.game_over),
            terminal_reason=str(self.session.terminal_reason),
            backend=str(self.session.backend),
            seed=int(self.session.seed),
        )

    def _append_history_frame(self) -> None:
        self.history.append(self._snapshot_live())
        self.history_index = len(self.history) - 1

    def _at_live_edge(self) -> bool:
        return self.history_index == len(self.history) - 1

    def _display_session(self):
        # History transition replay uses the source committed state while the
        # selected piece visibly moves toward the next committed state.
        if (
            self.visual_drop is not None
            and self.review_animation_source is not None
        ):
            source = self.review_animation_source
            return ReviewSession(
                visual_board=source.visual_board,
                active=self.visual_drop.piece,
                hold=self.visual_drop.display_hold,
                preview_pieces=self.visual_drop.display_preview,
                stats=source.stats,
                pending_decision=source.pending_decision,
                game_over=False,
                terminal_reason="",
                backend=source.backend,
                seed=source.seed,
            )

        if not self._at_live_edge():
            return self.history[self.history_index]

        if self.visual_drop is None:
            return self.session

        # Live Hold animation: show post-Hold / pre-lock HOLD and NEXT.
        return ReviewSession(
            visual_board=self.session.visual_board,
            active=self.visual_drop.piece,
            hold=self.visual_drop.display_hold,
            preview_pieces=self.visual_drop.display_preview,
            stats=self.session.stats,
            pending_decision=self._decision_summary(
                self.session.pending_decision
            ),
            game_over=False,
            terminal_reason="",
            backend=self.session.backend,
            seed=self.session.seed,
        )

    def _decision_candidate_count(self, decision) -> int:
        if hasattr(decision, "candidate_count"):
            return int(decision.candidate_count)
        return len(decision.candidates)

    def _decision_safer_candidate(self, decision):
        if hasattr(decision, "safer_candidate"):
            return decision.safer_candidate
        if decision.safer_candidate_index is None:
            return None
        return decision.candidates[decision.safer_candidate_index]

    def _ghost_visual(self):
        s = self._display_session()
        ids = np.asarray(s.visual_board, dtype=np.uint8).copy()
        decision = s.pending_decision
        ghost = None if decision is None else decision.chosen.state

        falling = None
        if self.visual_drop is not None:
            falling = self.visual_drop.pose()

        return ids, ghost, falling

    def _draw_board(self, rect):
        pygame = self.pygame
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            rect,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            PANEL_BORDER,
            rect,
            2,
            border_radius=6,
        )
        self._text(
            "PLAYFIELD",
            rect.x + 10,
            rect.y + 8,
            font=self.font_small,
        )

        inner = pygame.Rect(
            rect.x + 18,
            rect.y + 34,
            rect.width - 36,
            rect.height - 50,
        )
        cell = min(inner.width // 10, inner.height // 20)
        bw, bh = cell * 10, cell * 20
        ox = inner.x + (inner.width - bw) // 2
        oy = inner.y + (inner.height - bh) // 2

        board_ids, ghost, falling = self._ghost_visual()
        visible = board_ids[20:40]

        ghost_cells = set()
        if ghost is not None:
            for x, y in occupied_cells(ghost):
                if 20 <= y < 40 and 0 <= x < 10:
                    ghost_cells.add((x, y - 20))

        falling_cells = set()
        if falling is not None:
            for x, y in occupied_cells(falling):
                if 20 <= y < 40 and 0 <= x < 10:
                    falling_cells.add((x, y - 20))

        for y in range(20):
            for x in range(10):
                r = pygame.Rect(
                    ox + x * cell,
                    oy + y * cell,
                    cell,
                    cell,
                )
                pid = int(visible[y, x])
                piece = VIS_ID_TO_PIECE.get(pid)

                if piece is not None:
                    self._draw_block(r, TETROMINO_COLORS[piece])
                else:
                    pygame.draw.rect(self.screen, EMPTY, r)
                    pygame.draw.rect(self.screen, GRID, r, 1)

                if (
                    (x, y) in ghost_cells
                    and pid == 0
                    and (x, y) not in falling_cells
                ):
                    c = TETROMINO_COLORS[ghost.piece]
                    ghost_rect = r.inflate(
                        -max(2, cell // 4),
                        -max(2, cell // 4),
                    )
                    pygame.draw.rect(
                        self.screen,
                        self._darken(c, 0.45),
                        ghost_rect,
                        2,
                    )

                if (x, y) in falling_cells:
                    self._draw_block(
                        r,
                        TETROMINO_COLORS[falling.piece],
                    )

    def _button(self, x, y, label, action):
        pygame = self.pygame
        surf = self.font_small.render(label, True, TEXT)
        rect = pygame.Rect(
            x,
            y,
            surf.get_width() + 18,
            27,
        )
        pygame.draw.rect(
            self.screen,
            (42, 46, 55),
            rect,
            border_radius=5,
        )
        pygame.draw.rect(
            self.screen,
            PANEL_BORDER,
            rect,
            1,
            border_radius=5,
        )
        self.screen.blit(
            surf,
            (rect.x + 9, rect.y + 5),
        )
        self.buttons.append((rect, action))
        return rect.right + 6

    def _mini_piece_cells(
        self,
        piece: str,
    ) -> tuple[tuple[int, int], ...]:
        state = PieceState(
            piece=piece,
            x=0,
            y=0,
            rotation=0,
        )
        cells = list(occupied_cells(state))
        min_x = min(x for x, _ in cells)
        min_y = min(y for _, y in cells)
        return tuple(
            (x - min_x, y - min_y)
            for x, y in cells
        )

    def _draw_mini_piece(
        self,
        rect,
        piece: str | None,
        *,
        label: str | None = None,
    ):
        pygame = self.pygame
        pygame.draw.rect(
            self.screen,
            (23, 26, 32),
            rect,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            PANEL_BORDER,
            rect,
            1,
            border_radius=6,
        )

        top_pad = 5
        if label is not None:
            self._text(
                label,
                rect.x + 7,
                rect.y + 5,
                font=self.font_small,
                color=MUTED,
            )
            top_pad = 22

        if piece is None:
            self._text(
                "-",
                rect.centerx - 3,
                rect.centery - 7,
                font=self.font,
                color=MUTED,
            )
            return

        cells = self._mini_piece_cells(piece)
        max_x = max(x for x, _ in cells)
        max_y = max(y for _, y in cells)
        shape_w = max_x + 1
        shape_h = max_y + 1

        usable_h = rect.height - top_pad - 7
        mini_cell = max(
            5,
            min(
                18,
                (rect.width - 14) // max(1, shape_w),
                usable_h // max(1, shape_h),
            ),
        )
        ox = (
            rect.x
            + (rect.width - shape_w * mini_cell) // 2
        )
        oy = (
            rect.y
            + top_pad
            + max(
                0,
                (usable_h - shape_h * mini_cell) // 2,
            )
        )

        color = TETROMINO_COLORS[piece]
        for cx, cy in cells:
            block = pygame.Rect(
                ox + cx * mini_cell,
                oy + cy * mini_cell,
                mini_cell,
                mini_cell,
            )
            self._draw_block(block, color)

    def _draw_hold_next(
        self,
        session_view,
        x: int,
        y: int,
        width: int,
        height: int,
    ):
        pygame = self.pygame
        preview_w = max(76, min(108, width))
        hold_h = 78
        gap = 7

        hold_rect = pygame.Rect(
            x,
            y,
            preview_w,
            hold_h,
        )
        self._draw_mini_piece(
            hold_rect,
            session_view.hold,
            label="HOLD",
        )

        next_y = hold_rect.bottom + gap
        self._text(
            "NEXT",
            x + 4,
            next_y,
            font=self.font_small,
            color=MUTED,
        )
        next_y += 19

        preview = session_view.preview()
        count = min(5, len(preview))
        if count == 0:
            return

        available = max(
            0,
            height - (next_y - y),
        )
        box_h = max(
            44,
            min(
                66,
                (
                    available
                    - gap * (count - 1)
                ) // count,
            ),
        )
        for i, piece in enumerate(preview[:count]):
            box = pygame.Rect(
                x,
                next_y + i * (box_h + gap),
                preview_w,
                box_h,
            )
            self._draw_mini_piece(box, piece)

    def _draw_side(self, rect):
        pygame = self.pygame
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            rect,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            PANEL_BORDER,
            rect,
            2,
            border_radius=6,
        )

        s = self._display_session()
        d = s.pending_decision
        y = rect.y + 14

        self._text(
            self.PANEL_TITLE,
            rect.x + 14,
            y,
            font=self.font_big,
            color=WARN,
        )
        y += 42

        preview_x = rect.x + 14
        preview_y = y
        preview_w = 104
        preview_h = min(
            470,
            rect.height - 100,
        )
        self._draw_hold_next(
            s,
            preview_x,
            preview_y,
            preview_w,
            preview_h,
        )

        stats_x = preview_x + preview_w + 18
        stats_w = rect.right - stats_x - 12

        active_rect = pygame.Rect(
            stats_x,
            y,
            min(150, stats_w),
            58,
        )
        self._draw_mini_piece(
            active_rect,
            s.active,
            label="ACTIVE",
        )
        y = active_rect.bottom + 14

        st = s.stats
        self._text(
            f"Seed        {s.seed}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            f"Pieces      {st.pieces:,}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            f"Lines       {st.lines:,}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            f"Tetrises    {st.tetrises:,}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            "Singles/Doubles/Triples  "
            f"{st.line_counts[1]}/"
            f"{st.line_counts[2]}/"
            f"{st.line_counts[3]}",
            stats_x,
            y,
            font=self.font_small,
        )
        y += 25
        self._text(
            f"Height      {st.current_height} "
            f" max {st.max_height}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            f"Holes       {st.current_holes} "
            f" max {st.max_holes}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            "Hole-making  "
            f"{getattr(st, 'hole_creation_moves', 0)}"
            "   avoidable "
            f"{getattr(st, 'avoidable_hole_moves', 0)}",
            stats_x,
            y,
            font=self.font_small,
            color=(
                BAD
                if getattr(st, "avoidable_hole_moves", 0)
                else MUTED
            ),
        )
        y += 25
        self._text(
            f"Avg height  {st.avg_height:.2f}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            f"Hold rate   {st.hold_rate:.3f}",
            stats_x,
            y,
        )
        y += 25
        self._text(
            f"Avg cand.   {st.avg_candidates:.2f}",
            stats_x,
            y,
        )
        y += 33

        if d is not None:
            hold_color = GOOD if d.use_hold else MUTED
            self._text(
                f"Hold P={d.hold_probability:.3f}  "
                f"decision={'HOLD' if d.use_hold else 'NO HOLD'}  "
                f"mode={d.branch.mode}",
                stats_x,
                y,
                font=self.font_small,
                color=hold_color,
            )
            y += 28

            c = d.chosen
            self._text(
                f"Top-1: {c.state.piece} "
                f"r{c.state.rotation % 4} "
                f"x={c.state.x} y={c.state.y} "
                f"lines={c.lines} score={c.score:.3f}",
                stats_x,
                y,
                font=self.font_small,
                color=GOOD,
            )
            y += 25

            candidate_count = self._decision_candidate_count(d)
            self._text(
                f"Candidates={candidate_count} "
                f"{'REFERENCE AUDIT' if d.audited_reference else ''}",
                stats_x,
                y,
                font=self.font_small,
                color=MUTED,
            )
            y += 25

            if d.chosen_hole_delta > 0:
                risk_text = (
                    f"HOLE +{d.chosen_hole_delta}"
                    + (
                        "  AVOIDABLE"
                        if d.avoidable_hole
                        else ""
                    )
                )
                self._text(
                    risk_text,
                    stats_x,
                    y,
                    font=self.font_small,
                    color=BAD,
                )

                safer = self._decision_safer_candidate(d)
                if safer is not None:
                    self._text(
                        f"safer: r{safer.state.rotation % 4} "
                        f"x={safer.state.x} y={safer.state.y} "
                        f"holes={d.min_candidate_holes} "
                        f"score gap={d.chosen.score - safer.score:.3f}",
                        stats_x + 105,
                        y,
                        font=self.font_small,
                        color=WARN,
                    )
                y += 25

            if self.show_detail:
                for rank, (_, cand) in enumerate(
                    d.top3,
                    1,
                ):
                    self._text(
                        f"#{rank} {cand.state.piece} "
                        f"r{cand.state.rotation % 4} "
                        f"x={cand.state.x} y={cand.state.y} "
                        f"L{cand.lines} score={cand.score:.3f}",
                        stats_x + 10,
                        y,
                        font=self.font_small,
                        color=(
                            TEXT
                            if rank > 1
                            else GOOD
                        ),
                    )
                    y += 23
        else:
            self._text(
                f"TERMINAL: {s.terminal_reason}",
                stats_x,
                y,
                color=(
                    BAD
                    if s.terminal_reason != "LIMIT"
                    else WARN
                ),
            )

        if self.show_detail:
            y = min(
                rect.bottom - 78,
                y + 18,
            )
            self._text(
                f"Backend={s.backend}  "
                f"ref audits={st.fast_reference_audits} "
                f"fallbacks={st.fast_reference_fallbacks}",
                stats_x,
                y,
                font=self.font_small,
                color=MUTED,
            )

    def _save_screenshot(self):
        path = self.SCREENSHOT_DIR
        path.mkdir(parents=True, exist_ok=True)
        display = self._display_session()
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        out = (
            path
            / (
                f"seed_{display.seed}_"
                f"p{display.stats.pieces}_{stamp}.png"
            )
        )
        self.pygame.image.save(
            self.screen,
            str(out),
        )
        print(f"Screenshot: {out}")

    def _save_result(self):
        result = {
            "format": self.RESULT_FORMAT,
            "result": self.session.result(),
        }
        self.save_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.save_json.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _reset(self, seed=None):
        self.playing = False
        self.visual_drop = None
        self.manual_step_animation = False
        self.review_animation_source = None
        self.review_animation_target_index = None
        self.session.reset(
            self.session.seed
            if seed is None
            else seed
        )
        self.history = []
        self.history_index = 0
        self._append_history_frame()
        self.last_step_ms = self.pygame.time.get_ticks()

    def _animation_duration(self) -> float:
        # Exact V3.4 lineage scaling:
        # max(0.12, min(1.15, 2.20 / speed))
        return max(
            0.12,
            min(
                1.15,
                2.20 / max(self.speed, 0.05),
            ),
        )

    def _visual_hold_preview_from(
        self,
        source,
        decision,
    ) -> tuple[str | None, tuple[str, ...]]:
        preview = tuple(source.preview())

        if not decision.use_hold:
            return source.hold, preview

        # Current Active goes into Hold before the selected piece locks.
        display_hold = source.active

        if decision.branch.mode == "hold_empty":
            # For a historical frame we already captured Next5, so shift what
            # is available. For the live session, pull one extra bag item so
            # the visual queue still shows five pieces.
            if source is self.session:
                future = self.session.stream.peek(PREVIEW_DEPTH + 1)
                return (
                    display_hold,
                    tuple(future[1:1 + PREVIEW_DEPTH]),
                )

            shifted = tuple(preview[1:])
            return display_hold, shifted

        # hold_swap consumes no queue before lock.
        return display_hold, preview

    def _visual_hold_preview(
        self,
        decision: Decision,
    ) -> tuple[str | None, tuple[str, ...]]:
        return self._visual_hold_preview_from(
            self.session,
            decision,
        )

    def _begin_step_animation(
        self,
        *,
        manual: bool,
    ) -> None:
        if self.session.game_over:
            return
        if self.visual_drop is not None:
            return
        if not self._at_live_edge():
            return

        decision = self.session.pending_decision
        if decision is None:
            return

        if not self.fall_animation:
            self.session.step()
            self._append_history_frame()
            self.last_step_ms = self.pygame.time.get_ticks()
            return

        hold, preview = self._visual_hold_preview(decision)
        chosen = decision.chosen.state

        self.visual_drop = VisualDrop(
            piece=str(chosen.piece),
            target_rotation=int(chosen.rotation) % 4,
            target_x=int(chosen.x),
            landing_y=int(chosen.y),
            display_hold=hold,
            display_preview=preview,
            started_at=time.perf_counter(),
            duration=self._animation_duration(),
        )
        self.manual_step_animation = bool(manual)

    def _commit_visual_drop(self) -> None:
        if self.visual_drop is None:
            return

        # History replay: only advance the display cursor. Never touch model,
        # RNG, queue, or rollout statistics.
        if self.review_animation_target_index is not None:
            self.history_index = self.review_animation_target_index
            self.review_animation_target_index = None
            self.review_animation_source = None
            self.visual_drop = None
            self.manual_step_animation = False
            return

        # Live edge: now commit the already-selected model action.
        self.session.step()
        self.visual_drop = None
        self.manual_step_animation = False
        self._append_history_frame()
        self.last_step_ms = self.pygame.time.get_ticks()

        if self.session.game_over:
            self.playing = False

    def _finish_visual_drop(self) -> None:
        if self.visual_drop is None:
            return
        self._commit_visual_drop()
        self.playing = False

    def _update_animation(self) -> None:
        if self.visual_drop is None:
            return

        # Manual Right-step runs exactly one animation while the viewer remains
        # paused. Autoplay animations run while self.playing is true.
        if not (self.playing or self.manual_step_animation):
            return

        if self.visual_drop.progress() >= 1.0:
            self._commit_visual_drop()

    def _begin_review_animation(self) -> None:
        if self.visual_drop is not None:
            return
        if self.history_index >= len(self.history) - 1:
            return

        source = self.history[self.history_index]
        decision = source.pending_decision
        if decision is None:
            # No stored decision: fall back to direct history advance.
            self.history_index += 1
            return

        hold, preview = self._visual_hold_preview_from(
            source,
            decision,
        )
        chosen = decision.chosen.state

        self.review_animation_source = source
        self.review_animation_target_index = self.history_index + 1
        self.visual_drop = VisualDrop(
            piece=str(chosen.piece),
            target_rotation=int(chosen.rotation) % 4,
            target_x=int(chosen.x),
            landing_y=int(chosen.y),
            display_hold=hold,
            display_preview=preview,
            started_at=time.perf_counter(),
            duration=self._animation_duration(),
        )
        self.manual_step_animation = True

    def _review_prev(self) -> None:
        self.playing = False
        self.manual_step_animation = False

        # Cancel an uncommitted display-only animation.
        self.visual_drop = None
        self.review_animation_source = None
        self.review_animation_target_index = None

        if self.history_index > 0:
            self.history_index -= 1

    def _review_next_or_step(self) -> None:
        self.playing = False

        if self.visual_drop is not None:
            # Same responsive behavior as old viewer: pressing Next again
            # immediately reveals the target committed state.
            self._finish_visual_drop()
            return

        if self.history_index < len(self.history) - 1:
            self._begin_review_animation()
            return

        # Live edge: animate one new model-selected placement.
        self._begin_step_animation(manual=True)

    def _handle_action(self, action):
        if action == "play":
            if not self._at_live_edge():
                self.history_index = len(self.history) - 1
            self.manual_step_animation = False
            self.playing = not self.playing
            if self.playing:
                self.last_step_ms = self.pygame.time.get_ticks()

        elif action == "prev":
            self._review_prev()

        elif action == "next":
            self._review_next_or_step()

        elif action == "reset":
            self._reset()

        elif action == "next_seed":
            self._reset(self.session.seed + 1)

        elif action == "slower":
            self.speed = max(
                0.1,
                self.speed / 1.4,
            )

        elif action == "faster":
            self.speed = min(
                240.0,
                self.speed * 1.4,
            )

        elif action == "detail":
            self.show_detail = not self.show_detail

        elif action == "screenshot":
            self._save_screenshot()

        elif action == "quit":
            self.running = False

    def _events(self):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h),
                    pygame.RESIZABLE,
                )

            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                for rect, action in self.buttons:
                    if rect.collidepoint(event.pos):
                        self._handle_action(action)
                        break

            elif event.type == pygame.KEYDOWN:
                if event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q,
                ):
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self._handle_action("play")

                elif event.key == pygame.K_LEFT:
                    self._handle_action("prev")

                elif event.key == pygame.K_RIGHT:
                    self._handle_action("next")

                elif event.key == pygame.K_r:
                    self._handle_action("reset")

                elif event.key == pygame.K_n:
                    self._handle_action("next_seed")

                elif event.key in (
                    pygame.K_MINUS,
                    pygame.K_KP_MINUS,
                ):
                    self._handle_action("slower")

                elif event.key in (
                    pygame.K_EQUALS,
                    pygame.K_PLUS,
                    pygame.K_KP_PLUS,
                ):
                    self._handle_action("faster")

                elif event.key == pygame.K_d:
                    self._handle_action("detail")

                elif event.key == pygame.K_s:
                    self._handle_action("screenshot")

                elif pygame.K_1 <= event.key <= pygame.K_9:
                    preset = event.key - pygame.K_0
                    self.speed = SPEED_PRESETS[preset]

    def _draw_controls(self, rect):
        pygame = self.pygame
        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            rect,
            border_radius=6,
        )
        pygame.draw.rect(
            self.screen,
            PANEL_BORDER,
            rect,
            1,
            border_radius=6,
        )
        self.buttons = []

        x = rect.x + 8
        y = rect.y + 8
        x = self._button(
            x,
            y,
            "Space Play/Pause",
            "play",
        )
        x = self._button(
            x,
            y,
            "← Prev",
            "prev",
        )
        x = self._button(
            x,
            y,
            "→ Next/Step",
            "next",
        )
        x = self._button(
            x,
            y,
            "R Reset",
            "reset",
        )
        x = self._button(
            x,
            y,
            "N Next Seed",
            "next_seed",
        )
        x = self._button(
            x,
            y,
            "- Slower",
            "slower",
        )
        x = self._button(
            x,
            y,
            "+ Faster",
            "faster",
        )
        x = self._button(
            x,
            y,
            "D Detail",
            "detail",
        )
        x = self._button(
            x,
            y,
            "S Screenshot",
            "screenshot",
        )
        self._button(
            x,
            y,
            "Esc Quit",
            "quit",
        )

        display = self._display_session()

        if not self._at_live_edge():
            state = (
                f"REVIEW {self.history_index + 1}/"
                f"{len(self.history)}"
            )
            state_color = WARN
        elif self.visual_drop is not None:
            if self.review_animation_source is not None:
                state = "REPLAY FALLING/ROTATING"
            else:
                state = (
                    "FALLING/ROTATING"
                    if self.playing
                    else "STEP FALLING/ROTATING"
                )
            state_color = GOOD
        elif self.session.game_over:
            state = "GAME OVER"
            state_color = BAD
        elif self.playing:
            state = "PLAYING"
            state_color = GOOD
        else:
            state = "PAUSED"
            state_color = WARN

        self._text(
            f"State: {state}   "
            f"Speed: {self.speed:.1f} pieces/s   "
            f"Seed: {display.seed}   "
            f"Piece: {display.stats.pieces:,}   "
            f"History: {self.history_index + 1}/{len(self.history)}   "
            "1-9 speed presets",
            rect.x + 10,
            rect.bottom - 25,
            font=self.font_small,
            color=state_color,
        )

    def _draw(self):
        self.screen.fill(BG)
        w, h = self.screen.get_size()
        top = 12
        control_h = 78
        gap = 12
        board_w = min(
            520,
            max(380, int(w * 0.43)),
        )

        board_rect = self.pygame.Rect(
            12,
            top,
            board_w,
            h - control_h - top - gap,
        )
        side_rect = self.pygame.Rect(
            board_rect.right + gap,
            top,
            w - board_rect.right - gap - 12,
            board_rect.height,
        )
        control_rect = self.pygame.Rect(
            12,
            h - control_h,
            w - 24,
            control_h - 8,
        )

        self._draw_board(board_rect)
        self._draw_side(side_rect)
        self._draw_controls(control_rect)
        self.pygame.display.flip()

    def run(self):
        try:
            while self.running:
                self._events()

                self._update_animation()

                if (
                    self.playing
                    and self.visual_drop is None
                    and not self.session.game_over
                    and self._at_live_edge()
                ):
                    now_ms = self.pygame.time.get_ticks()
                    interval = (
                        1000.0
                        / max(0.1, self.speed)
                    )
                    if now_ms - self.last_step_ms >= interval:
                        self._begin_step_animation(
                            manual=False
                        )
                        self.last_step_ms = now_ms

                if self.session.game_over:
                    self.playing = False

                self._draw()
                self.clock.tick(60)

        finally:
            self._save_result()
            self.pygame.quit()


def main() -> None:
    args = parse_args()

    if args.max_pieces < 0:
        raise SystemExit("--max-pieces must be >= 0")
    if args.reference_audit_every < 0:
        raise SystemExit("--reference-audit-every must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    model, checkpoint = load_checkpoint(args.checkpoint, device)

    if args.headless:
        run_headless(model, checkpoint, args, device)
        return

    session = ExpertV0Rollout(
        model,
        device=device,
        seed=args.seed,
        max_pieces=args.max_pieces,
        backend=args.backend,
        fast_max_states=args.fast_max_states,
        reference_max_states=args.reference_max_states,
        reference_audit_every=args.reference_audit_every,
        hold_threshold=args.hold_threshold,
    )
    viewer = RolloutViewer(
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
