from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


EMPTY_HOLD = "N"


class ActionMode(str, Enum):
    NO_HOLD = "no_hold"
    HOLD_SWAP = "hold_swap"
    HOLD_EMPTY = "hold_empty"


@dataclass(frozen=True)
class ActionInference:
    modes: tuple[ActionMode, ...]
    active_piece: str | None
    consumed_preview: int | None
    queue_prefix_match: bool
    reason: str

    @property
    def uniquely_classified(self) -> bool:
        return len(self.modes) == 1


def _queue_remainder_matches(prev_next: str, cur_next: str, consumed: int) -> bool:
    if consumed < 0 or len(prev_next) < consumed:
        return False
    remainder = prev_next[consumed:]
    # The corpus keeps a rolling preview and appends future bag pieces.
    # Therefore every still-visible piece from the previous preview must be
    # the prefix of the next row's preview.
    return cur_next.startswith(remainder)


def infer_action_from_previous_post_state(
    *,
    prev_hold: str,
    prev_next: str,
    placed: str,
    cur_hold: str,
    cur_next: str,
    empty_hold: str = EMPTY_HOLD,
) -> ActionInference:
    """Infer the next placement's hold usage from consecutive corpus rows.

    Hypothesis under test:
    - each row's `hold` and `next` describe state *after* that row's placement;
    - the next active piece is therefore `prev_next[0]`;
    - the following row's `placed` is the piece that ultimately locks.

    Three standard cases are considered:
    1. no hold: active locks; one preview piece is consumed.
    2. swap hold: previous hold locks; active moves to hold; one preview piece
       is consumed.
    3. empty hold: active moves to empty hold; next preview piece becomes
       active and locks; two preview pieces are consumed.
    """

    prev_hold = str(prev_hold)
    prev_next = str(prev_next)
    placed = str(placed)
    cur_hold = str(cur_hold)
    cur_next = str(cur_next)

    if not prev_next:
        return ActionInference((), None, None, False, "previous preview is empty")

    active = prev_next[0]
    modes: list[ActionMode] = []

    no_hold_queue = _queue_remainder_matches(prev_next, cur_next, 1)
    if placed == active and cur_hold == prev_hold and no_hold_queue:
        modes.append(ActionMode.NO_HOLD)

    if prev_hold != empty_hold:
        swap_queue = _queue_remainder_matches(prev_next, cur_next, 1)
        if placed == prev_hold and cur_hold == active and swap_queue:
            modes.append(ActionMode.HOLD_SWAP)

    if prev_hold == empty_hold and len(prev_next) >= 2:
        empty_queue = _queue_remainder_matches(prev_next, cur_next, 2)
        if placed == prev_next[1] and cur_hold == active and empty_queue:
            modes.append(ActionMode.HOLD_EMPTY)

    if len(modes) == 1:
        mode = modes[0]
        consumed = 2 if mode is ActionMode.HOLD_EMPTY else 1
        return ActionInference(
            tuple(modes),
            active_piece=active,
            consumed_preview=consumed,
            queue_prefix_match=True,
            reason="unique standard hold/preview transition",
        )

    if len(modes) > 1:
        return ActionInference(
            tuple(modes),
            active_piece=active,
            consumed_preview=None,
            queue_prefix_match=True,
            reason="multiple modes fit because piece identities are ambiguous",
        )

    # Report whether the queue alone looks like a 1- or 2-piece consumption,
    # which helps diagnose extractor timing even when piece/hold fields disagree.
    q1 = _queue_remainder_matches(prev_next, cur_next, 1)
    q2 = _queue_remainder_matches(prev_next, cur_next, 2)
    queue_ok = q1 or q2
    return ActionInference(
        (),
        active_piece=active,
        consumed_preview=None,
        queue_prefix_match=queue_ok,
        reason=f"no standard mode matched (queue consume1={q1}, consume2={q2})",
    )
