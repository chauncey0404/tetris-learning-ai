from __future__ import annotations

from tetris_ai.battle.rotation.base import RotationSystem, target_rotation, y_up_to_board
from tetris_ai.battle.types import MoveAction


# Source convention: +x right, +y upward (standard SRS notation).
_JLSTZ_Y_UP = {
    (0, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    (1, 0): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    (1, 2): ((0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)),
    (2, 1): ((0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)),
    (2, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
    (3, 2): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    (3, 0): ((0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)),
    (0, 3): ((0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)),
}

_I_Y_UP = {
    (0, 1): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    (1, 0): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (1, 2): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
    (2, 1): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (2, 3): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (3, 2): ((0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)),
    (3, 0): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (0, 3): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
}


class SRSRotationSystem(RotationSystem):
    """Standard SRS 90° wall kicks.

    The standard SRS tables do not define a native 180° kick operation, so this
    implementation deliberately rejects ROTATE_180. Games that add 180° use a
    different RotationSystem.
    """

    name = "SRS"

    @property
    def supports_180(self) -> bool:
        return False

    def kick_tests(self, piece: str, from_rotation: int, action: MoveAction):
        if action == MoveAction.ROTATE_180:
            raise ValueError("Standard SRS has no native 180-degree kick table")
        if action not in (MoveAction.CW, MoveAction.CCW):
            raise ValueError(f"Not a rotation action: {action}")

        to_rotation = target_rotation(from_rotation, action)
        key = (int(from_rotation) % 4, to_rotation)

        if piece == "O":
            # Geometry is invariant in this project's 2x2 native O box.
            return to_rotation, ((0, 0),)
        if piece == "I":
            return to_rotation, y_up_to_board(_I_Y_UP[key])
        if piece in {"J", "L", "S", "T", "Z"}:
            return to_rotation, y_up_to_board(_JLSTZ_Y_UP[key])
        raise ValueError(f"Unknown piece: {piece!r}")
