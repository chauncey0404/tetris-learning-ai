from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tetris_ai.core.spins.base import SpinKind


class ClearKind(str, Enum):
    NONE = "none"
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    QUAD = "quad"
    EXTENDED = "extended"


def clear_kind(lines: int) -> ClearKind:
    lines = int(lines)
    if lines <= 0:
        return ClearKind.NONE
    if lines == 1:
        return ClearKind.SINGLE
    if lines == 2:
        return ClearKind.DOUBLE
    if lines == 3:
        return ClearKind.TRIPLE
    if lines == 4:
        return ClearKind.QUAD
    return ClearKind.EXTENDED


@dataclass(frozen=True, slots=True)
class ClearEvent:
    """Game-agnostic description of one locked piece after clear detection.

    TETR.IO attack logic consumes this compact event instead of rescanning the
    board.  The simulator is responsible for supplying all_clear and the number
    of garbage rows among the cleared rows.
    """

    piece: str
    lines: int
    spin: SpinKind = SpinKind.NONE
    all_clear: bool = False
    garbage_rows_cleared: int = 0

    def __post_init__(self) -> None:
        if self.lines < 0:
            raise ValueError("lines must be >= 0")
        if self.garbage_rows_cleared < 0:
            raise ValueError("garbage_rows_cleared must be >= 0")
        if self.garbage_rows_cleared > self.lines:
            raise ValueError("garbage_rows_cleared cannot exceed lines")

    @property
    def kind(self) -> ClearKind:
        return clear_kind(self.lines)

    @property
    def has_clear(self) -> bool:
        return self.lines > 0

    @property
    def is_spin(self) -> bool:
        return self.spin is not SpinKind.NONE

    @property
    def is_mini(self) -> bool:
        return self.spin is SpinKind.MINI

    @property
    def is_full_spin(self) -> bool:
        return self.spin is SpinKind.FULL

    @property
    def is_quad(self) -> bool:
        return self.lines == 4 and not self.is_spin

    @property
    def is_difficult_clear(self) -> bool:
        # Season 2 treats Quads, spin line clears, and All Clears as B2B clears.
        return self.has_clear and (self.lines == 4 or self.is_spin or self.all_clear)

    @property
    def preserves_b2b_without_increment(self) -> bool:
        # A placement with no line clear does not break B2B.  We intentionally
        # do not increment the chain for Spin-Zero here; special modes may
        # override that behavior later (e.g. Quick Play Warlock).
        return not self.has_clear
