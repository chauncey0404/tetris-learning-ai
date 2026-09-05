from __future__ import annotations

import math


def advance_combo(previous_combo: int, lines_cleared: int) -> int:
    """Return TETR.IO combo x for this placement.

    First consecutive line clear is combo 0, second is combo 1, etc.
    A no-clear placement resets the combo to -1.
    """

    if int(lines_cleared) <= 0:
        return -1
    return int(previous_combo) + 1


def multiplier_attack(effective_base: int, combo: int) -> int:
    """TETR.IO Multiplier attack with DOWN rounding.

    Tetra League/default non-Quick-Play multiplayer uses floor rounding.
    """

    effective_base = int(effective_base)
    combo = int(combo)
    if combo < 0:
        return 0
    if effective_base > 0:
        return int(math.floor(effective_base * (1.0 + 0.25 * combo)))
    if combo < 2:
        return 0
    return int(math.floor(math.log(1.0 + 1.25 * combo)))
