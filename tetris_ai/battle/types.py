from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional


class Rotation(IntEnum):
    """Canonical project rotation numbering.

    0 = spawn, 1 = geometric clockwise 90°, 2 = 180°, 3 = clockwise 270°.
    """

    SPAWN = 0
    RIGHT = 1
    REVERSE = 2
    LEFT = 3


class MoveAction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    CW = "cw"
    CCW = "ccw"
    ROTATE_180 = "180"
    HARD_DROP = "hard_drop"

    @property
    def is_rotation(self) -> bool:
        return self in {self.CW, self.CCW, self.ROTATE_180}


@dataclass(frozen=True)
class RotationTrace:
    action: MoveAction
    from_rotation: int
    to_rotation: int
    kick_index: int
    kick_dx: int
    kick_dy: int


@dataclass(frozen=True)
class PieceState:
    """Falling-piece state in V9 board coordinates.

    Coordinates use a top-left matrix anchor:
    - x grows rightward
    - y grows downward
    - y < 0 is permitted for pieces above the board

    The piece matrix itself preserves the SRS native box size:
    I=4x4, O=2x2, JLSTZ=3x3.
    """

    piece: str
    x: int
    y: int
    rotation: int = 0
    last_action: Optional[MoveAction] = None
    last_rotation: Optional[RotationTrace] = None

    def geometry_key(self) -> tuple[str, int, int, int]:
        return (self.piece, int(self.x), int(self.y), int(self.rotation) % 4)

    def search_key(self) -> tuple:
        rotation_meta = None
        if self.last_action is not None and self.last_action.is_rotation and self.last_rotation is not None:
            rotation_meta = (
                self.last_rotation.action.value,
                self.last_rotation.from_rotation,
                self.last_rotation.to_rotation,
                self.last_rotation.kick_index,
                self.last_rotation.kick_dx,
                self.last_rotation.kick_dy,
            )
        return self.geometry_key() + (
            None if self.last_action is None else self.last_action.value,
            rotation_meta,
        )


@dataclass(frozen=True)
class RotationAttempt:
    success: bool
    state: PieceState
    kick_index: Optional[int] = None
    kick_dx: int = 0
    kick_dy: int = 0


@dataclass(frozen=True)
class ReachablePlacement:
    """A hard-drop lock candidate reachable through an exact input path."""

    pre_drop_state: PieceState
    landing_state: PieceState
    path: tuple[MoveAction, ...]
    drop_distance: int

    @property
    def spin_signature(self) -> tuple:
        """Metadata needed by the later V9.1 spin classifier.

        Do not classify spins here. V9.0 only preserves the path-sensitive facts.
        """

        lr = self.pre_drop_state.last_rotation
        if lr is None:
            return (self.pre_drop_state.last_action, None)
        return (
            self.pre_drop_state.last_action,
            lr.action,
            lr.from_rotation,
            lr.to_rotation,
            lr.kick_index,
            lr.kick_dx,
            lr.kick_dy,
        )
