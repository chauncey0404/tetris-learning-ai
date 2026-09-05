from __future__ import annotations

from tetris_ai.battle.rotation.base import target_rotation, y_up_to_board
from tetris_ai.battle.rotation.srs import SRSRotationSystem, _JLSTZ_Y_UP
from tetris_ai.battle.types import MoveAction


# TETR.IO SRS+ keeps the JLSTZ 90-degree SRS kicks and changes the I-piece
# 90-degree ordering/side symmetry. These are written in the conventional
# +y-up coordinate system, then converted at the boundary to the project's
# row-major +y-down board coordinates.
#
# Values are the "basic rotation coordinate system" sequences documented in
# the SRS+ implementation notes linked from the TETR.IO rotation references.
_I_SRS_PLUS_90_Y_UP = {
    # CW
    (0, 1): ((0, 0), (+1, 0), (-2, 0), (-2, -1), (+1, +2)),
    (1, 2): ((0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)),
    (2, 3): ((0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)),
    (3, 0): ((0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)),
    # CCW
    (0, 3): ((0, 0), (-1, 0), (+2, 0), (+2, -1), (-1, +2)),
    (1, 0): ((0, 0), (-1, 0), (+2, 0), (-1, -2), (+2, +1)),
    (2, 1): ((0, 0), (-2, 0), (+1, 0), (-2, +1), (+1, -2)),
    (3, 2): ((0, 0), (+1, 0), (-2, 0), (+1, +2), (-2, -1)),
}


# TETR.IO's native 180° table, conventional +y-up notation.
# JLSTZ use six ordered tests per source orientation.
_OTHER_180_Y_UP = {
    0: ((0, 0), (0, +1), (+1, +1), (-1, +1), (+1, 0), (-1, 0)),
    1: ((0, 0), (+1, 0), (+1, +2), (+1, +1), (0, +2), (0, +1)),
    2: ((0, 0), (0, -1), (-1, -1), (+1, -1), (-1, 0), (+1, 0)),
    3: ((0, 0), (-1, 0), (-1, +2), (-1, +1), (0, +2), (0, +1)),
}

# The referenced solver stores I-piece positions with orientation-dependent
# anchor offsets. Converting those raw offsets back into this project's fixed
# 4x4 SRS matrix anchor gives the following basic-rotation kick tests.
_I_180_Y_UP = {
    0: ((0, 0), (0, +1)),
    1: ((0, 0), (+1, 0)),
    2: ((0, 0), (0, -1)),
    3: ((0, 0), (-1, 0)),
}


class TetrioSRSPlusRotationSystem(SRSRotationSystem):
    """TETR.IO default SRS+ movement rotation system.

    This class models movement only. Spin scoring/classification is intentionally
    deferred to V9.1 because TETR.IO permits independently configurable kick and
    spin rules.
    """

    name = "TETR.IO SRS+"

    @property
    def supports_180(self) -> bool:
        return True

    def kick_tests(self, piece: str, from_rotation: int, action: MoveAction):
        r = int(from_rotation) % 4

        if action == MoveAction.ROTATE_180:
            to_rotation = target_rotation(r, action)
            if piece == "O":
                # The O geometry is unchanged. Keep the input legal as a no-kick
                # orientation transition without inventing movement.
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
