from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable


INTERESTING_TYPES = frozenset(
    {
        "garbage",
        "interaction",
        "interaction_confirm",
        "target",
        "allow_targeting",
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract raw current-.ttrm multiplayer events that mention "
            "garbage/interaction semantics. Diagnostic only: no semantic "
            "normalization is performed."
        )
    )
    p.add_argument("replay", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"artifacts\tetrio\parity\garbage_event_probe.json"),
    )
    p.add_argument("--max-print", type=int, default=40)
    return p.parse_args()


def _all_type_tags(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    stack = [value]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            t = cur.get("type")
            if isinstance(t, str):
                out.append(t)
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return tuple(out)


def _iter_current_player_replays(
    obj: Any,
) -> Iterable[tuple[int, int, dict[str, Any], dict[str, Any]]]:
    if not isinstance(obj, dict):
        return
    replay_root = obj.get("replay")
    if not isinstance(replay_root, dict):
        return
    rounds = replay_root.get("rounds")
    if not isinstance(rounds, list):
        return

    for round_index, round_obj in enumerate(rounds):
        if not isinstance(round_obj, list):
            continue
        for player_index, player_obj in enumerate(round_obj):
            if not isinstance(player_obj, dict):
                continue
            replay = player_obj.get("replay")
            if isinstance(replay, dict):
                yield round_index, player_index, player_obj, replay


def _user_summary(obj: Any) -> list[dict[str, Any]]:
    users = obj.get("users") if isinstance(obj, dict) else None
    if not isinstance(users, list):
        return []
    out = []
    for i, user in enumerate(users):
        if not isinstance(user, dict):
            out.append({"index": i, "raw": user})
            continue
        compact = {"index": i}
        for key in (
            "_id",
            "id",
            "username",
            "name",
            "role",
        ):
            if key in user:
                compact[key] = user[key]
        out.append(compact)
    return out


def _compact_event(
    *,
    round_index: int,
    player_index: int,
    event_index: int,
    event: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "player": player_index,
        "event_index": event_index,
        "frame": event.get("frame"),
        "event_type": event.get("type"),
        "type_tags": list(_all_type_tags(event)),
        "event": event,
    }


def main() -> None:
    args = parse_args()
    if not args.replay.is_file():
        raise SystemExit(f"Replay not found: {args.replay}")

    with args.replay.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    streams = []
    extracted = []
    tag_counts: Counter[str] = Counter()

    for round_index, player_index, player_obj, replay in _iter_current_player_replays(obj):
        events = replay.get("events")
        if not isinstance(events, list):
            events = []

        stream_counter: Counter[str] = Counter()
        selected_count = 0

        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            tags = _all_type_tags(event)
            for tag in tags:
                tag_counts[tag] += 1
                stream_counter[tag] += 1

            if event.get("type") != "ige":
                continue
            if not (INTERESTING_TYPES & set(tags)):
                continue

            extracted.append(
                _compact_event(
                    round_index=round_index,
                    player_index=player_index,
                    event_index=event_index,
                    event=event,
                )
            )
            selected_count += 1

        streams.append(
            {
                "round": round_index,
                "player": player_index,
                "frames": replay.get("frames"),
                "event_count": len(events),
                "selected_event_count": selected_count,
                "type_counts": dict(stream_counter),
                "player_object_keys": sorted(player_obj.keys()),
                "replay_keys": sorted(replay.keys()),
            }
        )

    report = {
        "format": "tetrio_raw_garbage_event_probe",
        "replay": str(args.replay),
        "top_level_keys": (
            sorted(obj.keys()) if isinstance(obj, dict) else []
        ),
        "users": _user_summary(obj),
        "stream_count": len(streams),
        "streams": streams,
        "tag_counts": dict(tag_counts),
        "interesting_types": sorted(INTERESTING_TYPES),
        "selected_events": extracted,
        "status": "RAW_EVIDENCE_ONLY",
        "warning": (
            "No event-name semantics are assumed. Review exact payloads before "
            "normalizing them into send/cancel/advance/tank oracle events."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 104)
    print("TETR.IO RAW GARBAGE EVENT PROBE")
    print("=" * 104)
    print("Replay   :", args.replay)
    print("Streams  :", len(streams))
    print("Selected :", len(extracted))
    print("Users    :", report["users"])
    print()
    for s in streams:
        print(
            f"round={s['round']} player={s['player']} "
            f"frames={s['frames']} events={s['event_count']} "
            f"selected={s['selected_event_count']}"
        )
        interesting = {
            k: v
            for k, v in s["type_counts"].items()
            if k in INTERESTING_TYPES or k == "ige"
        }
        print("  types:", interesting)

    print()
    print("FIRST RAW EVENTS")
    for item in extracted[: max(0, args.max_print)]:
        ev = item["event"]
        data = ev.get("data")
        print(
            f"R{item['round']} P{item['player']} "
            f"idx={item['event_index']} frame={item['frame']} "
            f"tags={item['type_tags']}"
        )
        print(" ", json.dumps(data, ensure_ascii=False, separators=(",", ":")))

    print()
    print("Status :", report["status"])
    print("Saved  :", args.output)


if __name__ == "__main__":
    main()
