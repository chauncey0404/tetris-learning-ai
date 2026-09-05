from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tetrio.battle_state import BattleState, BattleStepResult, resolve_battle_step
from tetrio.clear import ClearEvent
from tetrio.garbage_board import GarbageInsertResult, insert_garbage_rows
from tetrio.garbage_queue import GarbageQueue


@dataclass(frozen=True, slots=True)
class VersusTransportConfig:
    """Transport facts separated from still-unverified ranked tuning values."""

    travel_frames: int = 20
    passthrough: bool = False
    garbage_cap_per_piece: int | None = None

    def __post_init__(self) -> None:
        if self.travel_frames < 0:
            raise ValueError("travel_frames must be >= 0")
        if self.garbage_cap_per_piece is not None and self.garbage_cap_per_piece < 0:
            raise ValueError("garbage_cap_per_piece must be >= 0")


TETRA_LEAGUE_TRANSPORT_REFERENCE = VersusTransportConfig(
    travel_frames=20,
    passthrough=False,
    # Intentionally not guessed.  Validate exact ranked cap/activation behavior
    # against current TETR.IO replays before freezing V9.3B.
    garbage_cap_per_piece=None,
)


@dataclass(frozen=True, slots=True)
class VersusPlayerState:
    board: np.ndarray
    battle: BattleState = BattleState()
    incoming: GarbageQueue = GarbageQueue()
    frame: int = 0
    topped_out: bool = False

    @property
    def pending_garbage(self) -> int:
        return self.incoming.pending_lines


@dataclass(frozen=True, slots=True)
class VersusExchangeResult:
    actor_before: VersusPlayerState
    actor_after: VersusPlayerState
    opponent_before: VersusPlayerState
    opponent_after: VersusPlayerState
    battle_step: BattleStepResult
    packets_delivered: tuple[int, ...]


def _battle_synced_to_queue(player: VersusPlayerState) -> BattleState:
    return replace(player.battle, incoming_pending=player.incoming.pending_lines)


def resolve_clear_exchange(
    actor: VersusPlayerState,
    opponent: VersusPlayerState,
    event: ClearEvent,
    *,
    transport: VersusTransportConfig = TETRA_LEAGUE_TRANSPORT_REFERENCE,
    opener_phase_pieces: int = 14,
    surge_start: int = 4,
) -> VersusExchangeResult:
    """Resolve one actor clear and deliver uncancelled attack to the opponent.

    Default TETR.IO passthrough is disabled, so new attack is represented in
    the opponent's queue immediately and becomes tankable after travel_frames.
    """

    if transport.passthrough:
        raise NotImplementedError(
            "V9.3A intentionally targets default Tetra League passthrough=off."
        )

    synced = _battle_synced_to_queue(actor)
    step = resolve_battle_step(
        synced,
        event,
        opener_phase_pieces=opener_phase_pieces,
        surge_start=surge_start,
    )

    queue_cancel = actor.incoming.cancel_lines(step.garbage_cancelled)
    if queue_cancel.remaining_request != 0:
        raise AssertionError("BattleState cancellation exceeded detailed queue")

    actor_queue_after = GarbageQueue(queue_cancel.packets_after)
    actor_battle_after = replace(step.after, incoming_pending=actor_queue_after.pending_lines)
    actor_after = replace(actor, battle=actor_battle_after, incoming=actor_queue_after)

    delivered = tuple(step.cancellation.outgoing_after)
    opponent_queue_after = opponent.incoming.enqueue_packets(
        delivered,
        sent_frame=actor.frame,
        travel_frames=transport.travel_frames,
        seq_start=len(opponent.incoming.packets),
    )
    opponent_battle_after = replace(
        opponent.battle,
        incoming_pending=opponent_queue_after.pending_lines,
    )
    opponent_after = replace(
        opponent,
        battle=opponent_battle_after,
        incoming=opponent_queue_after,
    )

    return VersusExchangeResult(
        actor_before=actor,
        actor_after=actor_after,
        opponent_before=opponent,
        opponent_after=opponent_after,
        battle_step=step,
        packets_delivered=delivered,
    )


def tank_active_garbage(
    player: VersusPlayerState,
    *,
    holes: tuple[int, ...] | None = None,
    transport: VersusTransportConfig = TETRA_LEAGUE_TRANSPORT_REFERENCE,
) -> tuple[VersusPlayerState, GarbageInsertResult]:
    """Insert currently-active garbage into the actor board.

    If queued packets do not yet have replay-validated holes, the caller may
    supply one hole per active line.  This keeps unknown garbage messiness out
    of the engine instead of silently inventing a ranked default.
    """

    queue = player.incoming
    if holes is not None:
        # Materialize a queue copy whose active packets receive explicit holes
        # in FIFO order.  One packet must use one hole in this V9.3A reference
        # implementation (change-on-attack structure).
        hole_iter = iter(holes)
        rebuilt = []
        for p in queue.packets:
            if p.is_active(player.frame) and p.hole is None:
                try:
                    h = next(hole_iter)
                except StopIteration as exc:
                    raise ValueError("not enough holes supplied") from exc
                rebuilt.append(replace(p, hole=int(h)))
            else:
                rebuilt.append(p)
        queue = GarbageQueue(tuple(rebuilt))

    popped = queue.pop_active(
        player.frame,
        cap=transport.garbage_cap_per_piece,
        require_holes=True,
    )
    inserted = insert_garbage_rows(player.board, popped.holes)
    after_queue = GarbageQueue(popped.packets_after)
    after_battle = replace(player.battle, incoming_pending=after_queue.pending_lines)
    after = replace(
        player,
        board=inserted.board,
        incoming=after_queue,
        battle=after_battle,
        topped_out=player.topped_out or inserted.top_out,
    )
    return after, inserted
