from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CancelResult:
    pending_before: int
    outgoing_before: tuple[int, ...]
    opener_double_cancel: bool
    cancelled: int
    pending_after: int
    outgoing_after: tuple[int, ...]

    @property
    def sent(self) -> int:
        return sum(self.outgoing_after)


def _cancel_packet(packet: int, pending: int, *, multiplier: int) -> tuple[int, int, int]:
    packet = max(0, int(packet))
    pending = max(0, int(pending))
    if packet == 0 or pending == 0:
        return packet, pending, 0

    capacity = packet * multiplier
    cancelled = min(pending, capacity)
    # One outgoing line is consumed for each `multiplier` incoming lines (or
    # partial final group) it cancels.
    used = (cancelled + multiplier - 1) // multiplier
    return max(0, packet - used), pending - cancelled, cancelled


def cancel_packets(
    pending: int,
    outgoing_packets: tuple[int, ...],
    *,
    opener_double_cancel: bool = False,
) -> CancelResult:
    """Cancel outgoing attack against pending garbage, preserving packet order."""

    pending_before = max(0, int(pending))
    pending_now = pending_before
    multiplier = 2 if opener_double_cancel else 1
    remaining: list[int] = []
    cancelled = 0

    for packet in outgoing_packets:
        packet_after, pending_now, c = _cancel_packet(packet, pending_now, multiplier=multiplier)
        cancelled += c
        if packet_after > 0:
            remaining.append(packet_after)

    return CancelResult(
        pending_before=pending_before,
        outgoing_before=tuple(int(x) for x in outgoing_packets if int(x) > 0),
        opener_double_cancel=bool(opener_double_cancel),
        cancelled=cancelled,
        pending_after=pending_now,
        outgoing_after=tuple(remaining),
    )
