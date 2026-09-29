from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnownFutureState:
    """Known post-lock queue state without inventing unseen bag pieces.

    Expert-v0/v1 state exposes Active + Hold + Next5.  After one placement we
    can advance this queue exactly for the next decision, although the preview
    becomes shorter because the unseen sixth preview piece is intentionally not
    fabricated.
    """

    active: str
    hold: str | None
    preview: tuple[str, ...]


def selected_piece_for_branch(
    *,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
    use_hold: bool,
) -> str:
    if not use_hold:
        return active

    if hold is not None:
        return hold

    if not preview:
        raise ValueError("hold-empty branch requires preview[0]")
    return preview[0]


def advance_after_lock(
    *,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
    use_hold: bool,
    placed_piece: str | None = None,
) -> KnownFutureState:
    """Advance Active/Hold/known-preview after the current placement.

    This mirrors the autonomous rollout semantics already validated in
    ``watch_expert_v0.py``.

    no_hold:
        place Active
        next Active = preview[0]
        Hold unchanged

    hold_swap:
        place old Hold
        current Active becomes Hold
        next Active = preview[0]

    hold_empty:
        current Active becomes Hold
        place preview[0]
        next Active = preview[1]
    """
    preview = tuple(preview)
    expected = selected_piece_for_branch(
        active=active,
        hold=hold,
        preview=preview,
        use_hold=use_hold,
    )
    if placed_piece is not None and str(placed_piece) != expected:
        raise ValueError(
            "candidate piece/hold transition mismatch: "
            f"expected {expected}, got {placed_piece}"
        )

    if not use_hold:
        if len(preview) < 1:
            raise ValueError("no-hold transition requires preview[0]")
        return KnownFutureState(
            active=preview[0],
            hold=hold,
            preview=preview[1:],
        )

    if hold is None:
        if len(preview) < 2:
            raise ValueError("hold-empty transition requires preview[0:2]")
        return KnownFutureState(
            active=preview[1],
            hold=active,
            preview=preview[2:],
        )

    if len(preview) < 1:
        raise ValueError("hold-swap transition requires preview[0]")
    return KnownFutureState(
        active=preview[0],
        hold=active,
        preview=preview[1:],
    )


def available_next_branch_pieces(
    state: KnownFutureState,
) -> tuple[tuple[bool, str], ...]:
    """Return exact pieces available on the next decision.

    The first tuple is always no-hold.  The second is the hold branch when its
    selected piece is known from the currently exposed queue.
    """
    out: list[tuple[bool, str]] = [(False, state.active)]
    if state.hold is not None:
        out.append((True, state.hold))
    elif state.preview:
        out.append((True, state.preview[0]))
    return tuple(out)


def t_distance_flags(
    state: KnownFutureState,
) -> tuple[int, int, int, int]:
    """T availability categories for the post-candidate state."""
    active = int(state.active == "T")
    hold = int(state.hold == "T")
    p1 = int(len(state.preview) >= 1 and state.preview[0] == "T")
    p2 = int(len(state.preview) >= 2 and state.preview[1] == "T")
    return active, hold, p1, p2


def t_is_near(
    *,
    active: str,
    hold: str | None,
    preview: tuple[str, ...],
    depth: int = 2,
) -> bool:
    if active == "T" or hold == "T":
        return True
    return "T" in tuple(preview)[: max(0, int(depth))]
