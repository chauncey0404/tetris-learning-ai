from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np


PIECES = frozenset("IOTSZJL")


class ObservationNotReady(RuntimeError):
    """Raised when a temporal frame is not safe to feed to the policy."""


@dataclass(frozen=True)
class ModelObservation:
    """Minimal causal state consumed by the frozen TETR.IO policy.

    The active piece's visual x/y/rotation are deliberately excluded.  They are
    debug coordinates from the 20-row image and are not TETR.IO SRS spawn
    coordinates.  Reachability always starts from ``tetrio_spawn_state``.
    """

    frame_index: int
    board: tuple[tuple[int, ...], ...]  # canonical project 40x10 board
    active_piece: str
    hold_piece: str | None
    preview_queue: tuple[str, ...]
    board_confidence: float
    active_confidence: float
    hold_confidence: float
    preview_confidence: float
    fingerprint: str

    def board_array(self) -> np.ndarray:
        return np.asarray(self.board, dtype=np.uint8)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["board"] = [list(row) for row in self.board]
        out["preview_queue"] = list(self.preview_queue)
        return out


def _fingerprint(
    board: np.ndarray,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(board, dtype=np.uint8).tobytes(order="C"))
    h.update(b"\0")
    h.update(active.encode("ascii"))
    h.update(b"\0")
    h.update((hold or "-").encode("ascii"))
    h.update(b"\0")
    h.update("".join(preview).encode("ascii"))
    return h.hexdigest()[:20]


def build_model_observation(temporal) -> ModelObservation:
    """Convert one fail-closed temporal snapshot into the V1.1 state contract.

    Runtime checks intentionally duplicate the tracker's READY gate.  This is
    the final boundary before neural inference, so a malformed/partial state is
    rejected instead of guessed.
    """

    if not bool(getattr(temporal, "stable_pre_action", False)):
        raise ObservationNotReady(
            f"temporal snapshot is not stable_pre_action: "
            f"{getattr(temporal, 'reason', 'unknown')}"
        )

    active = getattr(temporal, "active", None)
    if active is None or getattr(active, "piece", None) not in PIECES:
        raise ObservationNotReady("active piece is unresolved")
    active_piece = str(active.piece)

    hold_piece = getattr(temporal, "hold_piece", None)
    if hold_piece is not None and hold_piece not in PIECES:
        raise ObservationNotReady(f"invalid hold piece: {hold_piece!r}")

    preview_raw = getattr(temporal, "preview_queue", None)
    if preview_raw is None:
        raise ObservationNotReady("preview queue is unresolved")
    preview = tuple(str(x) for x in preview_raw)
    if len(preview) != 5 or any(piece not in PIECES for piece in preview):
        raise ObservationNotReady(
            f"preview queue must contain exactly five canonical pieces: {preview!r}"
        )

    locked_raw = getattr(temporal, "locked_board", None)
    if locked_raw is None:
        raise ObservationNotReady("locked board is unresolved")
    visible = (np.asarray(locked_raw) != 0).astype(np.uint8)
    if visible.shape != (20, 10):
        raise ObservationNotReady(
            f"locked board must be 20x10, got {visible.shape}"
        )

    if int(getattr(temporal, "unknown_count", 0)) != 0:
        raise ObservationNotReady("READY snapshot still contains UNKNOWN cells")

    # Use the project's authoritative board lifting rule rather than duplicating
    # hidden-row assumptions in the vision layer.
    from tetrio.ruleset import TETRIO_MOVEMENT

    board40 = TETRIO_MOVEMENT.lift_visible_board(visible)
    board40 = (np.asarray(board40) != 0).astype(np.uint8)
    if board40.shape != (40, 10):
        raise RuntimeError(
            f"TETR.IO movement rules returned invalid board shape: {board40.shape}"
        )

    return ModelObservation(
        frame_index=int(getattr(temporal, "frame_index", -1)),
        board=tuple(tuple(int(v) for v in row) for row in board40),
        active_piece=active_piece,
        hold_piece=None if hold_piece is None else str(hold_piece),
        preview_queue=preview,
        board_confidence=float(getattr(temporal, "board_confidence", 0.0)),
        active_confidence=float(getattr(active, "confidence", 0.0)),
        hold_confidence=float(getattr(temporal, "hold_confidence", 0.0)),
        preview_confidence=float(getattr(temporal, "preview_confidence", 0.0)),
        fingerprint=_fingerprint(board40, active_piece, hold_piece, preview),
    )
