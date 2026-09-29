from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from tetrio.vision.board import MINO, TRANSIENT, UNKNOWN, VisualBoard
from tetrio.vision.piece_reader import PieceObservation, PreviewObservation


_BASE_SHAPES: dict[str, np.ndarray] = {
    "I": np.asarray([[1, 1, 1, 1]], dtype=np.uint8),
    "O": np.asarray([[1, 1], [1, 1]], dtype=np.uint8),
    "T": np.asarray([[0, 1, 0], [1, 1, 1]], dtype=np.uint8),
    "S": np.asarray([[0, 1, 1], [1, 1, 0]], dtype=np.uint8),
    "Z": np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
    "J": np.asarray([[1, 0, 0], [1, 1, 1]], dtype=np.uint8),
    "L": np.asarray([[0, 0, 1], [1, 1, 1]], dtype=np.uint8),
}


def _trim(matrix: np.ndarray) -> np.ndarray:
    ys, xs = np.where(matrix > 0)
    if len(xs) == 0:
        return matrix[:0, :0]
    return matrix[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _shape_rotations() -> dict[str, tuple[np.ndarray, ...]]:
    out: dict[str, tuple[np.ndarray, ...]] = {}
    for piece, base in _BASE_SHAPES.items():
        rotations: list[np.ndarray] = []
        current = base.copy()
        for _ in range(4):
            current = _trim(current)
            if not any(np.array_equal(current, old) for old in rotations):
                rotations.append(current.copy())
            # Project rotation numbering is clockwise. np.rot90(..., -1) is CW.
            current = np.rot90(current, -1)
        out[piece] = tuple(rotations)
    return out


_SHAPES = _shape_rotations()
_UNSET = object()


class TrackerPhase(str, Enum):
    WARMUP = "WARMUP"
    TRACKING = "TRACKING"
    TRANSITION = "TRANSITION"
    READY = "READY"


@dataclass(frozen=True)
class TrackerConfig:
    history_frames: int = 5
    min_history_frames: int = 3
    panel_stability_frames: int = 2
    active_stability_frames: int = 2
    locked_stability_frames: int = 2
    min_cell_confidence: float = 0.45
    min_panel_confidence: float = 0.70
    min_active_score: float = 0.72
    min_active_margin: float = 0.025
    cold_locked_vote_ratio: float = 0.80
    max_missing_active_cells: int = 1
    transition_cooldown_frames: int = 1
    layout_shift_tolerance_cells: float = 0.35
    layout_scale_tolerance: float = 0.03
    preview_max_shift: int = 3
    event_window_frames: int = 3
    generation_upward_jump_rows: int = 3
    generation_spawn_y_max: int = 6

    def __post_init__(self) -> None:
        if self.history_frames < 3:
            raise ValueError("history_frames must be >= 3")
        if not (2 <= self.min_history_frames <= self.history_frames):
            raise ValueError("min_history_frames must be in 2..history_frames")
        if self.panel_stability_frames < 1:
            raise ValueError("panel_stability_frames must be >= 1")
        if self.active_stability_frames < 1:
            raise ValueError("active_stability_frames must be >= 1")
        if self.locked_stability_frames < 1:
            raise ValueError("locked_stability_frames must be >= 1")
        if not (0.5 <= self.cold_locked_vote_ratio <= 1.0):
            raise ValueError("cold_locked_vote_ratio must be in [0.5, 1.0]")
        if self.max_missing_active_cells not in (0, 1):
            raise ValueError("max_missing_active_cells must be 0 or 1")
        if not (1 <= self.preview_max_shift <= 3):
            raise ValueError("preview_max_shift must be in 1..3")
        if self.event_window_frames < 1:
            raise ValueError("event_window_frames must be >= 1")
        if self.generation_upward_jump_rows < 2:
            raise ValueError("generation_upward_jump_rows must be >= 2")


@dataclass(frozen=True)
class ActivePieceObservation:
    piece: str
    confidence: float
    rotation: int
    x: int
    y: int
    visible_cells: tuple[tuple[int, int], ...]
    observed_cells: tuple[tuple[int, int], ...]
    recovered_cells: tuple[tuple[int, int], ...]

    @property
    def recovered(self) -> bool:
        return bool(self.recovered_cells)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalObservation:
    frame_index: int
    phase: str
    locked_board: tuple[tuple[int, ...], ...] | None
    active: ActivePieceObservation | None
    hold_piece: str | None
    hold_confidence: float
    preview_queue: tuple[str, ...] | None
    preview_confidence: float
    board_confidence: float
    stable_pre_action: bool
    spawn_event: bool
    lock_event: bool
    preview_shift_event: bool
    hold_changed: bool
    layout_reset: bool
    transient_count: int
    unknown_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["active"] = None if self.active is None else self.active.to_dict()
        if self.locked_board is not None:
            out["locked_board"] = [list(row) for row in self.locked_board]
        if self.preview_queue is not None:
            out["preview_queue"] = list(self.preview_queue)
        return out


@dataclass(frozen=True)
class _FrameEvidence:
    mino: np.ndarray
    transient: np.ndarray
    confidence: np.ndarray
    transient_count: int
    unknown_count: int


@dataclass(frozen=True)
class _ActiveCandidate:
    piece: str
    rotation: int
    x: int
    y: int
    mask: np.ndarray
    visible_cells: tuple[tuple[int, int], ...]
    observed_cells: tuple[tuple[int, int], ...]
    recovered_cells: tuple[tuple[int, int], ...]
    score: float


class TemporalStateTracker:
    """Temporal semantics for the static TETR.IO visual readers.

    Contract:
    - input remains 100% visual (VisualBoard + PreviewObservation);
    - the static MINO label is never treated as a locked cell by itself;
    - active identity is recovered from a legal tetromino silhouette over time;
    - HOLD/NEXT must be temporally stable before they can make a model-ready
      observation;
    - uncertain frames fail closed with stable_pre_action=False.

    Active x/y/rotation are *visible trimmed-shape debug coordinates*. They are
    not TETR.IO/SRS PieceState coordinates and must not be used as a movement
    path start state. Closed-loop movement still starts from tetrio_spawn_state.
    """

    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        self.config = config
        self._frame_index = 0
        self._layout_bbox: tuple[int, int, int, int] | None = None
        self._reset_runtime(keep_frame_index=True)

    def _reset_runtime(self, *, keep_frame_index: bool) -> None:
        if not keep_frame_index:
            self._frame_index = 0
        self._history: deque[_FrameEvidence] = deque(
            maxlen=self.config.history_frames
        )
        self._locked_board: np.ndarray | None = None
        self._locked_candidate: np.ndarray | None = None
        self._locked_candidate_count = 0
        self._last_board_commit_frame = -10**9

        self._raw_active_piece: str | None = None
        self._raw_active_count = 0
        self._last_active_candidate: _ActiveCandidate | None = None

        self._stable_hold: object | str | None = _UNSET
        self._hold_candidate: object | str | None = _UNSET
        self._hold_candidate_count = 0
        self._hold_current_ok = False
        self._hold_confidence = 0.0

        self._stable_preview: tuple[str, ...] | None = None
        self._preview_candidate: tuple[str, ...] | None = None
        self._preview_candidate_count = 0
        self._preview_current_ok = False
        self._preview_confidence = 0.0

        self._cooldown_until = -1
        self._last_stable_active: _ActiveCandidate | None = None
        self._pending_board_change_frame: int | None = None
        self._pending_generation_frame: int | None = None
        self._board_commit_count = 0

    def reset(self) -> None:
        self._layout_bbox = None
        self._reset_runtime(keep_frame_index=True)

    @property
    def locked_board(self) -> np.ndarray | None:
        if self._locked_board is None:
            return None
        return self._locked_board.copy()

    @staticmethod
    def _board_tuple(board: np.ndarray | None) -> tuple[tuple[int, ...], ...] | None:
        if board is None:
            return None
        return tuple(tuple(int(v) for v in row) for row in board.astype(np.uint8))

    def _layout_changed(self, bbox: tuple[int, int, int, int] | None) -> bool:
        if bbox is None:
            return False
        if self._layout_bbox is None:
            self._layout_bbox = tuple(int(v) for v in bbox)
            return False

        old = self._layout_bbox
        new = tuple(int(v) for v in bbox)
        old_cell = max(float(old[2]) / 10.0, 1.0)
        shift = max(abs(new[0] - old[0]), abs(new[1] - old[1])) / old_cell
        scale = max(
            abs(float(new[2]) / max(old[2], 1) - 1.0),
            abs(float(new[3]) / max(old[3], 1) - 1.0),
        )
        if (
            shift > self.config.layout_shift_tolerance_cells
            or scale > self.config.layout_scale_tolerance
        ):
            self._layout_bbox = new
            return True
        self._layout_bbox = new
        return False

    def _frame_evidence(self, board: VisualBoard) -> _FrameEvidence:
        rows = np.asarray(board.rows, dtype=object)
        conf = np.asarray(board.confidences, dtype=np.float32)
        if rows.shape != (20, 10) or conf.shape != (20, 10):
            raise ValueError(
                f"VisualBoard must be 20x10, got rows={rows.shape} conf={conf.shape}"
            )
        mino = (rows == MINO) & (conf >= self.config.min_cell_confidence)
        transient = rows == TRANSIENT
        transient_count = int(np.count_nonzero(transient))
        unknown_count = int(np.count_nonzero(rows == UNKNOWN))
        return _FrameEvidence(
            mino=np.asarray(mino, dtype=bool),
            transient=np.asarray(transient, dtype=bool),
            confidence=conf,
            transient_count=transient_count,
            unknown_count=unknown_count,
        )

    def _cold_locked_guess(self) -> np.ndarray | None:
        if len(self._history) < self.config.min_history_frames:
            return None
        stack = np.stack([f.mino for f in self._history], axis=0)
        votes = stack.mean(axis=0)
        return votes >= self.config.cold_locked_vote_ratio

    @staticmethod
    def _mask_for_shape(
        matrix: np.ndarray,
        *,
        x: int,
        y: int,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        mask = np.zeros((20, 10), dtype=bool)
        cells: list[tuple[int, int]] = []
        ys, xs = np.where(matrix > 0)
        for dy, dx in zip(ys.tolist(), xs.tolist()):
            row = y + int(dy)
            col = x + int(dx)
            if 0 <= row < 20 and 0 <= col < 10:
                mask[row, col] = True
                cells.append((row, col))
        return mask, tuple(cells)

    def _active_candidates(
        self,
        frame: _FrameEvidence,
    ) -> list[_ActiveCandidate]:
        reference_locked = (
            self._locked_board
            if self._locked_board is not None
            else self._cold_locked_guess()
        )
        if reference_locked is None:
            return []

        extras = frame.mino & ~reference_locked
        extra_count = int(extras.sum())
        if extra_count < 2:
            return []

        # While a piece locks, the old committed board and the new board can
        # coexist in ``extras`` until the next active piece is recovered.
        # Separate persistent unexplained cells (candidate new locked stack)
        # from dynamic unexplained cells (usually a wrong active hypothesis).
        # This breaks the chicken-and-egg cycle without promoting the new stack
        # until a legal active piece has independently been found.
        if len(self._history) >= 2:
            recent = np.stack([f.mino for f in self._history], axis=0)
            persistence = recent.mean(axis=0)
        else:
            persistence = frame.mino.astype(np.float32)
        pending_locked = (persistence >= 0.60) & ~reference_locked

        previous_piece = (
            None if self._last_active_candidate is None
            else self._last_active_candidate.piece
        )
        candidates: list[_ActiveCandidate] = []

        for piece, rotations in _SHAPES.items():
            for rotation, matrix in enumerate(rotations):
                mh, mw = matrix.shape
                # y may be slightly negative while a spawning piece is entering
                # the 20-row visible field. Require >=3 visible cells below.
                for y in range(-(mh - 1), 20):
                    for x in range(0, 10 - mw + 1):
                        shape_mask, visible_cells = self._mask_for_shape(
                            matrix,
                            x=x,
                            y=y,
                        )
                        visible_count = len(visible_cells)
                        if visible_count < 3:
                            continue
                        if np.any(shape_mask & reference_locked):
                            continue

                        observed_mask = shape_mask & extras
                        hits = int(observed_mask.sum())
                        missing = visible_count - hits
                        if hits < 3 or missing > self.config.max_missing_active_cells:
                            continue

                        unexplained_mask = extras & ~shape_mask
                        dynamic_unexplained_mask = unexplained_mask & ~pending_locked
                        dynamic_unexplained = int(
                            np.count_nonzero(dynamic_unexplained_mask)
                        )
                        if dynamic_unexplained > 1:
                            continue

                        coverage = hits / float(visible_count)
                        effective_extra_count = hits + dynamic_unexplained
                        precision = hits / float(max(effective_extra_count, 1))
                        hit_conf = float(
                            frame.confidence[observed_mask].mean()
                        ) if hits else 0.0
                        prior = 1.0 if previous_piece == piece else 0.0
                        score = (
                            0.52 * coverage
                            + 0.26 * min(1.0, precision)
                            + 0.14 * min(1.0, hit_conf)
                            + 0.08 * prior
                            - 0.05 * dynamic_unexplained
                        )

                        observed_cells = tuple(
                            (r, c)
                            for r, c in visible_cells
                            if extras[r, c]
                        )
                        recovered_cells = tuple(
                            (r, c)
                            for r, c in visible_cells
                            if not extras[r, c]
                        )
                        candidates.append(
                            _ActiveCandidate(
                                piece=piece,
                                rotation=rotation,
                                x=x,
                                y=y,
                                mask=shape_mask,
                                visible_cells=visible_cells,
                                observed_cells=observed_cells,
                                recovered_cells=recovered_cells,
                                score=float(score),
                            )
                        )

        return candidates

    def _find_active(self, frame: _FrameEvidence) -> _ActiveCandidate | None:
        candidates = self._active_candidates(frame)
        if not candidates:
            self._raw_active_piece = None
            self._raw_active_count = 0
            self._last_active_candidate = None
            return None

        candidates.sort(
            key=lambda c: (
                c.score,
                len(c.observed_cells),
                -len(c.recovered_cells),
                c.piece,
            ),
            reverse=True,
        )
        best = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else -1.0

        if best.score < self.config.min_active_score:
            return None

        # Exact four-cell matches are usually unique. For a 3/4 recovery, do
        # not accept a geometry tie unless temporal identity breaks it.
        margin = best.score - second_score
        if (
            margin < self.config.min_active_margin
            and best.recovered_cells
            and (
                self._last_active_candidate is None
                or self._last_active_candidate.piece != best.piece
            )
        ):
            return None

        if self._raw_active_piece == best.piece:
            self._raw_active_count += 1
        else:
            self._raw_active_piece = best.piece
            self._raw_active_count = 1
        self._last_active_candidate = best
        return best

    def _active_is_stable(self, active: _ActiveCandidate | None) -> bool:
        return (
            active is not None
            and self._raw_active_piece == active.piece
            and self._raw_active_count >= self.config.active_stability_frames
        )

    def _update_hold(self, obs: PieceObservation) -> tuple[bool, bool]:
        raw_valid = (
            obs.confidence >= self.config.min_panel_confidence
            and (obs.piece is None or obs.piece in _BASE_SHAPES)
        )
        if not raw_valid:
            self._hold_current_ok = False
            self._hold_candidate = _UNSET
            self._hold_candidate_count = 0
            return False, False

        raw = obs.piece

        # Once a populated HOLD has been established, a visual ``None`` is
        # ambiguous: the gray piece may simply be missed for one or several
        # frames. Never clear a populated HOLD from vision alone. New-game
        # boundaries must call reset()/reacquire layout, which restores _UNSET.
        # This is intentionally fail-closed: while the panel is unreadable we
        # report hold_ok=False rather than silently feeding an empty HOLD.
        if raw is None and self._stable_hold not in (_UNSET, None):
            self._hold_current_ok = False
            self._hold_candidate = _UNSET
            self._hold_candidate_count = 0
            self._hold_confidence = 0.0
            return False, False

        if self._hold_candidate is not _UNSET and self._hold_candidate == raw:
            self._hold_candidate_count += 1
        else:
            self._hold_candidate = raw
            self._hold_candidate_count = 1

        changed = False
        if self._hold_candidate_count >= self.config.panel_stability_frames:
            if self._stable_hold is _UNSET:
                # Initial stabilization is not a gameplay HOLD transition.
                self._stable_hold = raw
            elif self._stable_hold != raw:
                self._stable_hold = raw
                changed = True

        self._hold_current_ok = (
            self._stable_hold is not _UNSET
            and self._stable_hold == raw
        )
        self._hold_confidence = float(obs.confidence) if self._hold_current_ok else 0.0
        return self._hold_current_ok, changed

    def _update_preview(
        self,
        obs: PreviewObservation,
    ) -> tuple[bool, bool, bool]:
        queue: tuple[str, ...] | None = None
        confidence = 0.0
        if obs.next_complete and len(obs.next_queue) == 5:
            pieces = tuple(item.piece for item in obs.next_queue)
            if (
                all(piece in _BASE_SHAPES for piece in pieces)
                and all(
                    item.confidence >= self.config.min_panel_confidence
                    for item in obs.next_queue
                )
            ):
                queue = tuple(str(piece) for piece in pieces)
                confidence = float(min(item.confidence for item in obs.next_queue))

        if queue is None:
            self._preview_current_ok = False
            self._preview_candidate = None
            self._preview_candidate_count = 0
            return False, False, False

        if self._preview_candidate == queue:
            self._preview_candidate_count += 1
        else:
            self._preview_candidate = queue
            self._preview_candidate_count = 1

        changed = False
        shift_event = False
        previous = self._stable_preview
        if self._preview_candidate_count >= self.config.panel_stability_frames:
            if previous != queue:
                changed = True
                if previous is not None:
                    # Live manual play can advance more than one piece between
                    # two confirmed NEXT observations. Accept a deterministic
                    # suffix/prefix shift by 1..preview_max_shift, but require
                    # at least two overlapping pieces (max shift is capped at 3).
                    for shift in range(1, self.config.preview_max_shift + 1):
                        overlap = len(previous) - shift
                        if overlap >= 2 and previous[shift:] == queue[:overlap]:
                            shift_event = True
                            break
                self._stable_preview = queue

        self._preview_current_ok = self._stable_preview == queue
        self._preview_confidence = confidence if self._preview_current_ok else 0.0
        return self._preview_current_ok, changed, shift_event

    def _update_locked_board(
        self,
        frame: _FrameEvidence,
        active: _ActiveCandidate | None,
        active_stable: bool,
    ) -> tuple[bool, bool]:
        """Return (board_currently_stable, board_committed_this_frame)."""
        if active is None or not active_stable:
            self._locked_candidate = None
            self._locked_candidate_count = 0
            return False, False

        candidate = frame.mino & ~active.mask
        if (
            self._locked_candidate is not None
            and np.array_equal(candidate, self._locked_candidate)
        ):
            self._locked_candidate_count += 1
        else:
            self._locked_candidate = candidate.copy()
            self._locked_candidate_count = 1

        committed = False
        if self._locked_candidate_count >= self.config.locked_stability_frames:
            if self._locked_board is None or not np.array_equal(
                candidate,
                self._locked_board,
            ):
                self._locked_board = candidate.copy()
                self._last_board_commit_frame = self._frame_index
                committed = True

        current_stable = (
            self._locked_board is not None
            and np.array_equal(candidate, self._locked_board)
            and self._locked_candidate_count >= self.config.locked_stability_frames
        )
        return current_stable, committed

    @staticmethod
    def _dilate_mask(mask: np.ndarray) -> np.ndarray:
        out = mask.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                src_r1 = max(0, -dr)
                src_r2 = min(mask.shape[0], mask.shape[0] - dr)
                src_c1 = max(0, -dc)
                src_c2 = min(mask.shape[1], mask.shape[1] - dc)
                dst_r1 = src_r1 + dr
                dst_r2 = src_r2 + dr
                dst_c1 = src_c1 + dc
                dst_c2 = src_c2 + dc
                out[dst_r1:dst_r2, dst_c1:dst_c2] |= mask[
                    src_r1:src_r2, src_c1:src_c2
                ]
        return out

    def _transient_blocks(
        self,
        frame: _FrameEvidence,
        active: _ActiveCandidate | None,
        active_stable: bool,
    ) -> bool:
        """Return True only when transient cells can corrupt game semantics.

        The static reader labels any strongly colored *partial-cell* coverage as
        TRANSIENT to reject countdown/UI glyphs. During live gravity, however, a
        legitimate falling mino spends time between grid rows and creates the
        same partial-cell signature. Once a locked board and a stable active
        tetromino exist, transient cells inside a one-cell corridor around that
        active shape are benign motion evidence. Anything touching locked stack
        or outside that corridor remains fail-closed.
        """
        if frame.transient_count == 0:
            return False
        if self._locked_board is None or active is None or not active_stable:
            return True
        if np.any(frame.transient & self._locked_board):
            return True
        active_corridor = self._dilate_mask(active.mask)
        outside = frame.transient & ~active_corridor
        return bool(np.any(outside))

    def _active_generation_event(
        self,
        active: _ActiveCandidate | None,
        active_stable: bool,
        preview_shift: bool,
    ) -> bool:
        if active is None or not active_stable:
            return False
        previous = self._last_stable_active
        event = False
        if previous is not None:
            upward_jump = (
                previous.y - active.y
                >= self.config.generation_upward_jump_rows
            )
            piece_changed_near_spawn = (
                previous.piece != active.piece
                and active.y <= self.config.generation_spawn_y_max
            )
            event = bool(
                upward_jump
                or piece_changed_near_spawn
                or preview_shift
            )
        self._last_stable_active = active
        return event

    def _paired_lock_spawn_events(
        self,
        *,
        board_committed: bool,
        generation_event: bool,
    ) -> tuple[bool, bool]:
        if board_committed:
            self._board_commit_count += 1
            if self._board_commit_count > 1:
                self._pending_board_change_frame = self._frame_index

        if generation_event:
            self._pending_generation_frame = self._frame_index

        window = self.config.event_window_frames
        for attr in (
            "_pending_board_change_frame",
            "_pending_generation_frame",
        ):
            value = getattr(self, attr)
            if value is not None and self._frame_index - value > window:
                setattr(self, attr, None)

        paired = (
            self._pending_board_change_frame is not None
            and self._pending_generation_frame is not None
            and abs(
                self._pending_board_change_frame
                - self._pending_generation_frame
            ) <= window
        )
        if not paired:
            return False, False

        self._pending_board_change_frame = None
        self._pending_generation_frame = None
        return True, True

    def update(
        self,
        board: VisualBoard,
        previews: PreviewObservation,
        *,
        layout_bbox: tuple[int, int, int, int] | None = None,
    ) -> TemporalObservation:
        self._frame_index += 1
        layout_reset = self._layout_changed(layout_bbox)
        if layout_reset:
            self._reset_runtime(keep_frame_index=True)

        frame = self._frame_evidence(board)
        self._history.append(frame)

        hold_ok, hold_changed = self._update_hold(previews.hold)
        preview_ok, _preview_changed, preview_shift = self._update_preview(previews)

        active = self._find_active(frame)
        active_stable = self._active_is_stable(active)

        board_stable, board_committed = self._update_locked_board(
            frame,
            active,
            active_stable,
        )

        generation_event = self._active_generation_event(
            active,
            active_stable,
            preview_shift,
        )
        lock_event, spawn_event = self._paired_lock_spawn_events(
            board_committed=board_committed,
            generation_event=generation_event,
        )
        if lock_event:
            self._cooldown_until = max(
                self._cooldown_until,
                self._frame_index + self.config.transition_cooldown_frames,
            )

        transient_blocking = self._transient_blocks(
            frame,
            active,
            active_stable,
        )
        transient_free = not transient_blocking
        unknown_free = frame.unknown_count == 0
        history_ready = len(self._history) >= self.config.min_history_frames
        cooldown_clear = self._frame_index > self._cooldown_until

        stable_pre_action = bool(
            history_ready
            and self._locked_board is not None
            and board_stable
            and active_stable
            and hold_ok
            and preview_ok
            and transient_free
            and unknown_free
            and cooldown_clear
        )

        if stable_pre_action:
            phase = TrackerPhase.READY
            reason = "stable_pre_action"
        elif layout_reset:
            phase = TrackerPhase.WARMUP
            reason = "layout_reset"
        elif not history_ready:
            phase = TrackerPhase.WARMUP
            reason = "history_warmup"
        elif frame.unknown_count:
            phase = TrackerPhase.TRANSITION
            reason = "unknown_board_cells"
        elif not active_stable:
            phase = TrackerPhase.TRACKING
            reason = "active_unresolved"
        elif not hold_ok:
            phase = TrackerPhase.TRACKING
            reason = "hold_unstable"
        elif not preview_ok:
            phase = TrackerPhase.TRACKING
            reason = "preview_unstable"
        elif not board_stable:
            phase = TrackerPhase.TRANSITION
            reason = "locked_board_unstable"
        elif transient_blocking:
            phase = TrackerPhase.TRANSITION
            reason = "transient_overlay"
        elif not cooldown_clear:
            phase = TrackerPhase.TRANSITION
            reason = "post_lock_cooldown"
        else:
            phase = TrackerPhase.TRACKING
            reason = "not_ready"

        if active is None:
            active_out = None
            active_conf = 0.0
        else:
            active_conf = float(active.score)
            active_out = ActivePieceObservation(
                piece=active.piece,
                confidence=active_conf,
                rotation=int(active.rotation),
                x=int(active.x),
                y=int(active.y),
                visible_cells=active.visible_cells,
                observed_cells=active.observed_cells,
                recovered_cells=active.recovered_cells,
            )

        mean_cell_conf = float(np.mean(frame.confidence))
        board_confidence = min(
            1.0,
            max(0.0, mean_cell_conf)
            * (1.0 - frame.unknown_count / 200.0)
            * (1.0 - frame.transient_count / 200.0),
        )

        stable_hold = None if self._stable_hold is _UNSET else self._stable_hold
        return TemporalObservation(
            frame_index=self._frame_index,
            phase=phase.value,
            locked_board=self._board_tuple(self._locked_board),
            active=active_out,
            hold_piece=stable_hold,
            hold_confidence=float(self._hold_confidence),
            preview_queue=self._stable_preview,
            preview_confidence=float(self._preview_confidence),
            board_confidence=board_confidence,
            stable_pre_action=stable_pre_action,
            spawn_event=spawn_event,
            lock_event=lock_event,
            preview_shift_event=preview_shift,
            hold_changed=hold_changed,
            layout_reset=layout_reset,
            transient_count=frame.transient_count,
            unknown_count=frame.unknown_count,
            reason=reason,
        )
