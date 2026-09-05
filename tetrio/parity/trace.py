from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SUPPORTED_KINDS = frozenset({"send", "cancel", "advance", "assert_queue", "tank"})


@dataclass(frozen=True, slots=True)
class GarbageParityEvent:
    kind: str
    player: str
    frame: int | None = None
    target: str | None = None
    packets: tuple[int, ...] = ()
    lines: int | None = None
    holes: tuple[int, ...] | None = None
    travel_frames: int | None = None
    cap: int | None = None
    expected: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GarbageParityEvent":
        kind = str(raw.get("kind", "")).strip().lower()
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported parity event kind: {kind!r}")
        player = str(raw.get("player", "")).strip()
        if not player:
            raise ValueError("parity event requires player")
        packets_raw = raw.get("packets", ())
        packets = tuple(int(x) for x in packets_raw) if packets_raw is not None else ()
        holes_raw = raw.get("holes")
        holes = None if holes_raw is None else tuple(int(x) for x in holes_raw)
        expected = raw.get("expected")
        if expected is not None and not isinstance(expected, dict):
            raise ValueError("expected must be an object")
        return cls(
            kind=kind,
            player=player,
            frame=None if raw.get("frame") is None else int(raw["frame"]),
            target=None if raw.get("target") is None else str(raw["target"]),
            packets=packets,
            lines=None if raw.get("lines") is None else int(raw["lines"]),
            holes=holes,
            travel_frames=None
            if raw.get("travel_frames") is None
            else int(raw["travel_frames"]),
            cap=None if raw.get("cap") is None else int(raw["cap"]),
            expected=dict(expected) if expected is not None else None,
        )


def load_parity_trace(path: str | Path) -> tuple[GarbageParityEvent, ...]:
    trace_path = Path(path)
    with trace_path.open("r", encoding="utf-8-sig") as handle:
        obj = json.load(handle)
    if isinstance(obj, dict):
        events = obj.get("events")
    else:
        events = obj
    if not isinstance(events, list):
        raise ValueError("parity trace must be a list or an object with an events list")
    return tuple(GarbageParityEvent.from_dict(item) for item in events)
