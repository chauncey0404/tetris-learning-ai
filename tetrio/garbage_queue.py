from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional


@dataclass(frozen=True, slots=True)
class GarbagePacket:
    """One incoming TETR.IO garbage attack packet.

    Packets are cancelable immediately when passthrough is disabled.  They
    become tankable only after ``active_frame``.  Hole layout is intentionally
    optional: ranked TL messiness defaults are not hard-coded until replay
    parity confirms the exact current engine parameters.
    """

    lines: int
    sent_frame: int
    active_frame: int
    hole: Optional[int] = None
    source_seq: int = 0

    def __post_init__(self) -> None:
        if self.lines < 0:
            raise ValueError("lines must be >= 0")
        if self.sent_frame < 0 or self.active_frame < 0:
            raise ValueError("frames must be >= 0")
        if self.active_frame < self.sent_frame:
            raise ValueError("active_frame cannot precede sent_frame")
        if self.hole is not None and not 0 <= int(self.hole) < 10:
            raise ValueError("hole must be in [0, 9]")

    def with_lines(self, lines: int) -> "GarbagePacket":
        return replace(self, lines=max(0, int(lines)))

    def is_active(self, frame: int) -> bool:
        return int(frame) >= self.active_frame


@dataclass(frozen=True, slots=True)
class QueueCancelResult:
    requested: int
    cancelled: int
    remaining_request: int
    packets_after: tuple[GarbagePacket, ...]


@dataclass(frozen=True, slots=True)
class ActivePopResult:
    lines: int
    holes: tuple[int, ...]
    packets_after: tuple[GarbagePacket, ...]


@dataclass(frozen=True, slots=True)
class GarbageQueue:
    packets: tuple[GarbagePacket, ...] = ()

    @property
    def pending_lines(self) -> int:
        return sum(p.lines for p in self.packets)

    def active_lines(self, frame: int) -> int:
        return sum(p.lines for p in self.packets if p.is_active(frame))

    def enqueue_packets(
        self,
        packets: Iterable[int],
        *,
        sent_frame: int,
        travel_frames: int = 20,
        holes: Iterable[Optional[int]] | None = None,
        seq_start: int = 0,
    ) -> "GarbageQueue":
        packet_values = [int(x) for x in packets if int(x) > 0]
        if holes is None:
            hole_values: list[Optional[int]] = [None] * len(packet_values)
        else:
            hole_values = list(holes)
            if len(hole_values) != len(packet_values):
                raise ValueError("holes length must match positive packet count")

        appended = list(self.packets)
        for i, (lines, hole) in enumerate(zip(packet_values, hole_values)):
            appended.append(
                GarbagePacket(
                    lines=lines,
                    sent_frame=int(sent_frame),
                    active_frame=int(sent_frame) + int(travel_frames),
                    hole=hole,
                    source_seq=int(seq_start) + i,
                )
            )
        return GarbageQueue(tuple(appended))

    def cancel_lines(self, lines: int) -> QueueCancelResult:
        """FIFO-cancel incoming lines, including packets not yet active."""

        requested = max(0, int(lines))
        remaining = requested
        after: list[GarbagePacket] = []
        cancelled = 0

        for packet in self.packets:
            if remaining <= 0:
                after.append(packet)
                continue
            take = min(packet.lines, remaining)
            cancelled += take
            remaining -= take
            left = packet.lines - take
            if left > 0:
                after.append(packet.with_lines(left))

        return QueueCancelResult(
            requested=requested,
            cancelled=cancelled,
            remaining_request=remaining,
            packets_after=tuple(after),
        )

    def pop_active(
        self,
        frame: int,
        *,
        cap: int | None = None,
        require_holes: bool = True,
    ) -> ActivePopResult:
        """Pop active garbage in FIFO order for board insertion.

        ``cap`` is deliberately profile-driven.  Public sources document that
        TETR.IO exposes a garbage cap, but do not currently give us a reliable
        ranked Season-2 value to hard-code.  Passing ``None`` means unlimited.
        """

        remaining_cap = None if cap is None else max(0, int(cap))
        after: list[GarbagePacket] = []
        holes_out: list[int] = []
        lines_out = 0

        for packet in self.packets:
            if not packet.is_active(frame) or remaining_cap == 0:
                after.append(packet)
                continue

            take = packet.lines if remaining_cap is None else min(packet.lines, remaining_cap)
            if take <= 0:
                after.append(packet)
                continue
            if require_holes and packet.hole is None:
                raise ValueError(
                    "Garbage hole layout is unresolved for an active packet; "
                    "supply replay-validated holes before tanking it."
                )

            hole = 0 if packet.hole is None else int(packet.hole)
            holes_out.extend([hole] * take)
            lines_out += take
            if remaining_cap is not None:
                remaining_cap -= take

            left = packet.lines - take
            if left > 0:
                after.append(packet.with_lines(left))

        return ActivePopResult(
            lines=lines_out,
            holes=tuple(holes_out),
            packets_after=tuple(after),
        )
