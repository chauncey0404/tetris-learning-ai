from __future__ import annotations

from dataclasses import dataclass

from tetrio.clear import ClearEvent


@dataclass(frozen=True, slots=True)
class B2BState:
    """TETR.IO displayed B2B count plus stored Surge.

    count=-1 means no difficult clear has started the chain.
    The first difficult clear advances to count=0; the next to count=1.
    This matches the long-standing TETR.IO B2B numbering convention and makes
    Surge begin when the displayed count reaches x4.
    """

    count: int = -1
    surge: int = 0

    @property
    def active_bonus(self) -> bool:
        return self.count >= 0


def split_surge(lines: int) -> tuple[int, ...]:
    """Split stored Surge into three near-even packets.

    Remainders are placed in the first and then second packet, matching the
    documented TETR.IO ordering.
    """

    lines = int(lines)
    if lines <= 0:
        return ()
    q, r = divmod(lines, 3)
    packets = (q + (1 if r >= 1 else 0), q + (1 if r >= 2 else 0), q)
    return tuple(v for v in packets if v > 0)


@dataclass(frozen=True, slots=True)
class B2BTransition:
    before: B2BState
    after: B2BState
    bonus_for_current_attack: int
    released_surge: tuple[int, ...]
    broke_chain: bool


def advance_b2b(state: B2BState, event: ClearEvent, *, surge_start: int = 4) -> B2BTransition:
    """Advance Season-2 B2B Charging for one placement."""

    if event.is_difficult_clear:
        bonus = 1 if state.active_bonus else 0
        new_count = state.count + 1
        new_surge = state.surge
        if new_count >= int(surge_start):
            # In Tetra League/other standard multiplayer, Surge equals the
            # displayed B2B count from x4 onward.
            new_surge = new_count
        return B2BTransition(
            before=state,
            after=B2BState(count=new_count, surge=new_surge),
            bonus_for_current_attack=bonus,
            released_surge=(),
            broke_chain=False,
        )

    if event.has_clear:
        released = split_surge(state.surge)
        return B2BTransition(
            before=state,
            after=B2BState(),
            bonus_for_current_attack=0,
            released_surge=released,
            broke_chain=state.count >= 0,
        )

    # No clear: preserve the chain and charge, but do not increment it.
    return B2BTransition(
        before=state,
        after=state,
        bonus_for_current_attack=0,
        released_surge=(),
        broke_chain=False,
    )
