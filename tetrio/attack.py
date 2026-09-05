from __future__ import annotations

from dataclasses import dataclass

from tetrio.clear import ClearEvent
from tetrio.combo import multiplier_attack
from tetris_ai.core.spins.base import SpinKind


def base_attack(event: ClearEvent) -> int:
    """Season-2 TETR.IO base attack before B2B/combo/AC/special bonuses."""

    lines = int(event.lines)
    if lines <= 0:
        return 0

    if event.spin is SpinKind.FULL:
        # Normal T-spin table. Extended clears are only relevant to custom
        # mechanics; preserve TETR.IO's established 2x+2 continuation.
        if lines <= 3:
            return 2 * lines
        return 2 * lines + 2

    if event.spin is SpinKind.MINI:
        if event.piece == "T":
            if lines == 1:
                return 0
            if lines == 2:
                return 1
            return 0
        # Season-2 All-Mini: non-T All-Spins have zero base attack. They may
        # still sustain B2B and a combo can create attack through Multiplier.
        return 0

    return {1: 0, 2: 1, 3: 2, 4: 4}.get(lines, 0)


@dataclass(frozen=True, slots=True)
class AttackBreakdown:
    base: int
    b2b_bonus: int
    combo: int
    multiplied: int
    all_clear_bonus: int
    garbage_special_bonus: int
    total: int


def calculate_attack(event: ClearEvent, *, combo: int, b2b_bonus: int) -> AttackBreakdown:
    base = base_attack(event)

    # All-Mini non-T spins are documented as not sending on their own. Their
    # B2B value is strategic/charging rather than an immediate +1 attack.
    applied_b2b = int(b2b_bonus)
    if event.spin is SpinKind.MINI and event.piece != "T":
        applied_b2b = 0

    multiplied = multiplier_attack(base + applied_b2b, combo)
    all_clear_bonus = 5 if event.all_clear else 0
    garbage_special_bonus = (
        1
        if event.garbage_rows_cleared > 0 and (event.lines == 4 or event.is_spin)
        else 0
    )
    total = multiplied + all_clear_bonus + garbage_special_bonus
    return AttackBreakdown(
        base=base,
        b2b_bonus=applied_b2b,
        combo=int(combo),
        multiplied=multiplied,
        all_clear_bonus=all_clear_bonus,
        garbage_special_bonus=garbage_special_bonus,
        total=total,
    )
