from __future__ import annotations

from tetris_ai.core.rotation.base import target_rotation, y_up_to_board
from tetris_ai.core.rotation.srs import SRSRotationSystem, _JLSTZ_Y_UP
from tetris_ai.core.types import MoveAction


_I_SRS_PLUS_90_Y_UP = {
    (0, 1): ((0, 0), (+1, 0), (-2, 0), (-2, -1), (+1, +2)),
    (1, 2): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
    (2, 3): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (3, 0): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    (0, 3): ((0, 0), (-1, 0), (+2, 0), (+2, -1), (-1, +2)),
    (1, 0): ((0, 0), (-1, 0), (+2, 0), (-1, -2), (+2, +1)),
    (2, 1): ((0, 0), (-2, 0), (+1, 0), (-2, +1), (+1, -2)),
    (3, 2): ((0, 0), (+1, 0), (-2, 0), (+1, +2), (-2, -1)),
}

_OTHER_180_Y_UP = {
    0: ((0, 0), (0, +1), (+1, +1), (-1, +1), (+1, 0), (-1, 0)),
    1: ((0, 0), (+1, 0), (+1, +2), (+1, +1), (0, +2), (0, +1)),
    2: ((0, 0), (0, -1), (-1, -1), (+1, -1), (-1, 0), (+1, 0)),
    3: ((0, 0), (-1, 0), (-1, +2), (-1, +1), (0, +2), (0, +1)),
}

_I_180_Y_UP = {
    0: ((0, 0), (0, +1)),
    1: ((0, 0), (+1, 0)),
    2: ((0, 0), (0, -1)),
    3: ((0, 0), (-1, 0)),
}


class TetrioSRSPlusRotationSystem(SRSRotationSystem):
    """TETR.IO SRS+ movement rotation system used by current V9 profiles."""

    name = "TETR.IO SRS+"

    @property
    def supports_180(self) -> bool:
        return True

    def kick_tests(self, piece: str, from_rotation: int, action: MoveAction):
        r = int(from_rotation) % 4

        if action == MoveAction.ROTATE_180:
            to_rotation = target_rotation(r, action)
            if piece == "O":
                return to_rotation, ((0, 0),)
            if piece == "I":
                return to_rotation, y_up_to_board(_I_180_Y_UP[r])
            if piece in {"J", "L", "S", "T", "Z"}:
                return to_rotation, y_up_to_board(_OTHER_180_Y_UP[r])
            raise ValueError(f"Unknown piece: {piece!r}")

        if action not in (MoveAction.CW, MoveAction.CCW):
            raise ValueError(f"Not a rotation action: {action}")

        to_rotation = target_rotation(r, action)
        key = (r, to_rotation)
        if piece == "I":
            return to_rotation, y_up_to_board(_I_SRS_PLUS_90_Y_UP[key])
        if piece in {"J", "L", "S", "T", "Z"}:
            return to_rotation, y_up_to_board(_JLSTZ_Y_UP[key])
        if piece == "O":
            return to_rotation, ((0, 0),)
        raise ValueError(f"Unknown piece: {piece!r}")
