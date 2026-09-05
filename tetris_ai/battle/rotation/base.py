from __future__ import annotations

from abc import ABC, abstractmethod

from tetris_ai.battle.types import MoveAction


Kick = tuple[int, int]
KickSequence = tuple[Kick, ...]


class RotationSystem(ABC):
    """Rotation/kick policy independent from board simulation.

    Kick deltas are expressed in V9 board coordinates:
    +x = right, +y = down.
    """

    name: str

    @property
    @abstractmethod
    def supports_180(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def kick_tests(
        self,
        piece: str,
        from_rotation: int,
        action: MoveAction,
    ) -> tuple[int, KickSequence]:
        """Return (target_rotation, ordered kick tests)."""
        raise NotImplementedError


def target_rotation(from_rotation: int, action: MoveAction) -> int:
    r = int(from_rotation) % 4
    if action == MoveAction.CW:
        return (r + 1) % 4
    if action == MoveAction.CCW:
        return (r - 1) % 4
    if action == MoveAction.ROTATE_180:
        return (r + 2) % 4
    raise ValueError(f"Not a rotation action: {action}")


def y_up_to_board(kicks: tuple[tuple[int, int], ...]) -> KickSequence:
    """Convert conventional SRS (+y upward) offsets to board (+y downward)."""

    return tuple((int(dx), -int(dy)) for dx, dy in kicks)
