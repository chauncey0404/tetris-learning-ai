from __future__ import annotations

from dataclasses import dataclass


_FALSE_SPIN_LABELS = frozenset(
    {
        "",
        "0",
        "false",
        "n",
        "no",
        "none",
        "null",
        "normal",
    }
)


def normalize_spin_label(value: object) -> str:
    if value is None:
        return "NONE"
    text = str(value).strip()
    if text.lower() in _FALSE_SPIN_LABELS:
        return "NONE"
    return text.upper()


def is_spin_clear(*, lines_cleared: int, t_spin: object) -> bool:
    return int(lines_cleared) > 0 and normalize_spin_label(t_spin) != "NONE"


def is_difficult_clear(*, lines_cleared: int, t_spin: object) -> bool:
    """Historical-corpus structural predicate, not a Season-2 rules claim.

    This is deliberately conservative:
    - a four-line clear is difficult;
    - a line-clearing row labelled as a T-spin is difficult;
    - no-clear placements preserve the reconstructed difficult-clear chain;
    - ordinary line clears break it.

    The battle-state audit compares this reconstruction against the corpus'
    raw `btb` field before that raw field is allowed into model state.
    """

    lines = int(lines_cleared)
    return lines == 4 or is_spin_clear(lines_cleared=lines, t_spin=t_spin)


@dataclass(frozen=True, slots=True)
class PlacementOutcome:
    cleared: int = 0
    t_spin: str = "NONE"
    attack: float = 0.0
    garbage_cleared: int = 0


@dataclass(frozen=True, slots=True)
class CausalBattleHistory:
    """State that is knowable immediately before the next placement.

    `combo_chain` is normalized as number of consecutive prior placements that
    cleared at least one line. Thus 0 means no active combo; 1 means the
    immediately preceding placement started a combo.

    `difficult_chain` counts difficult clears since the last ordinary line
    clear. No-clear placements preserve this count. It is a reconstruction
    diagnostic until raw historical B2B timing is independently validated.
    """

    combo_chain: int = 0
    difficult_chain: int = 0
    previous: PlacementOutcome = PlacementOutcome()

    @property
    def combo_index(self) -> int:
        # TETR.IO-style index convention: no combo=-1, first clear=0.
        return self.combo_chain - 1 if self.combo_chain > 0 else -1

    @property
    def difficult_index(self) -> int:
        return self.difficult_chain - 1 if self.difficult_chain > 0 else -1

    @property
    def difficult_active(self) -> bool:
        return self.difficult_chain > 0


def advance_causal_history(
    state: CausalBattleHistory,
    outcome: PlacementOutcome,
) -> CausalBattleHistory:
    lines = int(outcome.cleared)

    combo_chain = state.combo_chain + 1 if lines > 0 else 0

    if is_difficult_clear(lines_cleared=lines, t_spin=outcome.t_spin):
        difficult_chain = state.difficult_chain + 1
    elif lines > 0:
        difficult_chain = 0
    else:
        difficult_chain = state.difficult_chain

    return CausalBattleHistory(
        combo_chain=int(combo_chain),
        difficult_chain=int(difficult_chain),
        previous=PlacementOutcome(
            cleared=lines,
            t_spin=normalize_spin_label(outcome.t_spin),
            attack=float(outcome.attack),
            garbage_cleared=int(outcome.garbage_cleared),
        ),
    )
