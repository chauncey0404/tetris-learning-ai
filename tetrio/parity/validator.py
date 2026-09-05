from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from tetrio.battle_state import BattleState
from tetrio.garbage_board import insert_garbage_rows
from tetrio.garbage_queue import GarbageQueue
from tetrio.parity.trace import GarbageParityEvent
from tetrio.versus import VersusPlayerState


@dataclass(frozen=True, slots=True)
class GarbageParityMismatch:
    event_index: int
    kind: str
    player: str
    field: str
    expected: Any
    actual: Any


@dataclass(frozen=True, slots=True)
class GarbageParityReport:
    event_count: int
    mismatches: tuple[GarbageParityMismatch, ...]
    players: dict[str, VersusPlayerState]

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _empty_player() -> VersusPlayerState:
    return VersusPlayerState(board=np.zeros((40, 10), dtype=np.uint8))


def _with_queue(player: VersusPlayerState, queue: GarbageQueue) -> VersusPlayerState:
    return replace(
        player,
        incoming=queue,
        battle=replace(player.battle, incoming_pending=queue.pending_lines),
    )


def _snapshot(player: VersusPlayerState) -> dict[str, Any]:
    bottom_holes: list[int | None] = []
    for row in player.board:
        occupied = np.flatnonzero(row == 0)
        if int(row.sum()) == 9 and len(occupied) == 1:
            bottom_holes.append(int(occupied[0]))
    return {
        "frame": int(player.frame),
        "pending": int(player.incoming.pending_lines),
        "active": int(player.incoming.active_lines(player.frame)),
        "packet_lines": [int(p.lines) for p in player.incoming.packets],
        "packet_holes": [None if p.hole is None else int(p.hole) for p in player.incoming.packets],
        "active_frames": [int(p.active_frame) for p in player.incoming.packets],
        "topped_out": bool(player.topped_out),
        "bottom_garbage_holes": bottom_holes,
    }


def _record_expected(
    mismatches: list[GarbageParityMismatch],
    index: int,
    event: GarbageParityEvent,
    player: VersusPlayerState,
) -> None:
    if not event.expected:
        return
    snapshot = _snapshot(player)
    for field, expected in event.expected.items():
        if field not in snapshot:
            mismatches.append(
                GarbageParityMismatch(index, event.kind, event.player, field, expected, "<unknown field>")
            )
            continue
        actual = snapshot[field]
        if actual != expected:
            mismatches.append(
                GarbageParityMismatch(index, event.kind, event.player, field, expected, actual)
            )


def validate_garbage_trace(
    events: Iterable[GarbageParityEvent],
    *,
    initial_players: dict[str, VersusPlayerState] | None = None,
    default_travel_frames: int = 20,
) -> GarbageParityReport:
    """Replay a normalized oracle trace against the V9.3A transport engine.

    This intentionally validates only transport/queue/board insertion facts.
    It does not infer raw TETR.IO replay event semantics.  A raw `.ttrm` must
    first be inspected and normalized from observed engine/replay evidence.
    """

    players = dict(initial_players or {})
    mismatches: list[GarbageParityMismatch] = []
    event_list = tuple(events)

    def get_player(name: str) -> VersusPlayerState:
        if name not in players:
            players[name] = _empty_player()
        return players[name]

    for index, event in enumerate(event_list):
        player = get_player(event.player)
        if event.frame is not None:
            player = replace(player, frame=int(event.frame))
            players[event.player] = player

        if event.kind == "send":
            if not event.target:
                raise ValueError(f"event {index}: send requires target")
            target = get_player(event.target)
            sent_frame = player.frame if event.frame is None else int(event.frame)
            travel = default_travel_frames if event.travel_frames is None else int(event.travel_frames)
            holes = event.holes
            if holes is not None and len(holes) != len(tuple(x for x in event.packets if x > 0)):
                raise ValueError(f"event {index}: send holes must match positive packet count")
            queue = target.incoming.enqueue_packets(
                event.packets,
                sent_frame=sent_frame,
                travel_frames=travel,
                holes=holes,
                seq_start=len(target.incoming.packets),
            )
            target = _with_queue(target, queue)
            players[event.target] = target
            _record_expected(mismatches, index, event, target)
            continue

        if event.kind == "cancel":
            if event.lines is None:
                raise ValueError(f"event {index}: cancel requires lines")
            result = player.incoming.cancel_lines(event.lines)
            player = _with_queue(player, GarbageQueue(result.packets_after))
            players[event.player] = player
            if event.expected is not None and "cancelled" in event.expected:
                expected_cancelled = event.expected["cancelled"]
                if result.cancelled != expected_cancelled:
                    mismatches.append(
                        GarbageParityMismatch(
                            index,
                            event.kind,
                            event.player,
                            "cancelled",
                            expected_cancelled,
                            result.cancelled,
                        )
                    )
                # prevent generic snapshot from flagging a non-snapshot field
                expected = dict(event.expected)
                expected.pop("cancelled", None)
                event = replace(event, expected=expected)
            _record_expected(mismatches, index, event, player)
            continue

        if event.kind == "advance":
            if event.frame is None:
                raise ValueError(f"event {index}: advance requires frame")
            players[event.player] = player
            _record_expected(mismatches, index, event, player)
            continue

        if event.kind == "assert_queue":
            players[event.player] = player
            _record_expected(mismatches, index, event, player)
            continue

        if event.kind == "tank":
            cap = event.cap
            queue = player.incoming
            if event.holes is not None:
                hole_iter = iter(event.holes)
                rebuilt = []
                for packet in queue.packets:
                    if packet.is_active(player.frame) and packet.hole is None:
                        try:
                            hole = next(hole_iter)
                        except StopIteration as exc:
                            raise ValueError(f"event {index}: not enough holes for active packets") from exc
                        rebuilt.append(replace(packet, hole=int(hole)))
                    else:
                        rebuilt.append(packet)
                queue = GarbageQueue(tuple(rebuilt))
            popped = queue.pop_active(player.frame, cap=cap, require_holes=True)
            inserted = insert_garbage_rows(player.board, popped.holes)
            queue_after = GarbageQueue(popped.packets_after)
            player = replace(
                _with_queue(player, queue_after),
                board=inserted.board,
                topped_out=player.topped_out or inserted.top_out,
            )
            players[event.player] = player
            if event.expected is not None and "inserted" in event.expected:
                expected_inserted = event.expected["inserted"]
                if inserted.lines_inserted != expected_inserted:
                    mismatches.append(
                        GarbageParityMismatch(
                            index,
                            event.kind,
                            event.player,
                            "inserted",
                            expected_inserted,
                            inserted.lines_inserted,
                        )
                    )
                expected = dict(event.expected)
                expected.pop("inserted", None)
                event = replace(event, expected=expected)
            _record_expected(mismatches, index, event, player)
            continue

        raise AssertionError(f"unhandled event kind {event.kind}")

    return GarbageParityReport(len(event_list), tuple(mismatches), players)
