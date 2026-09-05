from __future__ import annotations

from dataclasses import dataclass

from tetrio.attack import AttackBreakdown, calculate_attack
from tetrio.b2b import B2BState, B2BTransition, advance_b2b
from tetrio.clear import ClearEvent
from tetrio.combo import advance_combo
from tetrio.garbage import CancelResult, cancel_packets


@dataclass(frozen=True, slots=True)
class BattleState:
    pieces_placed: int = 0
    combo: int = -1
    b2b: B2BState = B2BState()
    incoming_pending: int = 0
    attack_sent_round: int = 0
    attack_generated_round: int = 0
    garbage_cancelled_round: int = 0

    def with_incoming(self, lines: int) -> "BattleState":
        lines = int(lines)
        if lines < 0:
            raise ValueError("lines must be >= 0")
        return BattleState(
            pieces_placed=self.pieces_placed,
            combo=self.combo,
            b2b=self.b2b,
            incoming_pending=self.incoming_pending + lines,
            attack_sent_round=self.attack_sent_round,
            attack_generated_round=self.attack_generated_round,
            garbage_cancelled_round=self.garbage_cancelled_round,
        )


@dataclass(frozen=True, slots=True)
class BattleStepResult:
    before: BattleState
    after: BattleState
    event: ClearEvent
    combo: int
    b2b: B2BTransition
    attack: AttackBreakdown
    outgoing_packets_before_cancel: tuple[int, ...]
    cancellation: CancelResult

    @property
    def attack_generated(self) -> int:
        return sum(self.outgoing_packets_before_cancel)

    @property
    def garbage_cancelled(self) -> int:
        return self.cancellation.cancelled

    @property
    def garbage_sent(self) -> int:
        return self.cancellation.sent


def resolve_battle_step(
    state: BattleState,
    event: ClearEvent,
    *,
    opener_phase_pieces: int = 14,
    surge_start: int = 4,
) -> BattleStepResult:
    """Resolve attack/B2B/combo/cancellation for one Tetra-League placement.

    This is deliberately board-independent and allocation-light. The future
    fast/self-play backend can batch these scalar transitions without invoking
    the reference reachability engine.
    """

    combo = advance_combo(state.combo, event.lines)
    b2b = advance_b2b(state.b2b, event, surge_start=surge_start)
    attack = calculate_attack(event, combo=combo, b2b_bonus=b2b.bonus_for_current_attack)

    packets: list[int] = []
    if attack.total > 0:
        packets.append(attack.total)
    packets.extend(b2b.released_surge)
    packet_tuple = tuple(packets)

    # Current documented opener rule: during the first 14 pieces, double
    # cancellation is active while pending garbage exceeds lines actually sent
    # so far in the round. Keeping this condition isolated makes future parity
    # corrections local if replay validation reveals a finer engine detail.
    opener_double = (
        state.pieces_placed < int(opener_phase_pieces)
        and state.incoming_pending > state.attack_sent_round
    )
    cancellation = cancel_packets(
        state.incoming_pending,
        packet_tuple,
        opener_double_cancel=opener_double,
    )

    generated = sum(packet_tuple)
    sent = cancellation.sent
    after = BattleState(
        pieces_placed=state.pieces_placed + 1,
        combo=combo,
        b2b=b2b.after,
        incoming_pending=cancellation.pending_after,
        attack_sent_round=state.attack_sent_round + sent,
        attack_generated_round=state.attack_generated_round + generated,
        garbage_cancelled_round=state.garbage_cancelled_round + cancellation.cancelled,
    )
    return BattleStepResult(
        before=state,
        after=after,
        event=event,
        combo=combo,
        b2b=b2b,
        attack=attack,
        outgoing_packets_before_cancel=packet_tuple,
        cancellation=cancellation,
    )
