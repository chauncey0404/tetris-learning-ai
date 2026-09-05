from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GARBAGE_TERMS = (
    "garbage",
    "attack",
    "ige",
    "hole",
    "mess",
    "queue",
    "target",
    "cancel",
    "frame",
)


def _json_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    if parent == "$":
        return f"$.{key}"
    return f"{parent}.{key}"


def _small_value(value: Any, *, max_chars: int = 240) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = repr(value)
        return value if len(text) <= max_chars else text[:max_chars] + "..."
    if isinstance(value, list):
        return {"type": "list", "len": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": list(value)[:20], "len": len(value)}
    return repr(value)[:max_chars]


@dataclass(frozen=True, slots=True)
class ReplayPlayerSummary:
    round_index: int
    replay_index: int
    frames: int | None
    keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplayInventory:
    path: str
    top_level_keys: tuple[str, ...]
    round_count: int | None
    player_replays: tuple[ReplayPlayerSummary, ...]
    key_counts: dict[str, int]
    tagged_value_counts: dict[str, dict[str, int]]
    matching_paths: dict[str, tuple[dict[str, Any], ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "top_level_keys": list(self.top_level_keys),
            "round_count": self.round_count,
            "player_replays": [
                {
                    "round_index": p.round_index,
                    "replay_index": p.replay_index,
                    "frames": p.frames,
                    "keys": list(p.keys),
                }
                for p in self.player_replays
            ],
            "key_counts": dict(sorted(self.key_counts.items())),
            "tagged_value_counts": {
                k: dict(sorted(v.items(), key=lambda item: (-item[1], item[0])))
                for k, v in sorted(self.tagged_value_counts.items())
            },
            "matching_paths": {k: list(v) for k, v in self.matching_paths.items()},
        }


def _extract_round_replays(obj: Any) -> tuple[int | None, tuple[ReplayPlayerSummary, ...]]:
    # Community TETR.IO tooling represents multiplayer .ttrm data as
    # root["data"][round]["replays"].  Do not assume any deeper event schema.
    if not isinstance(obj, dict):
        return None, ()
    data = obj.get("data")
    if not isinstance(data, list):
        return None, ()

    summaries: list[ReplayPlayerSummary] = []
    for round_index, round_obj in enumerate(data):
        if not isinstance(round_obj, dict):
            continue
        replays = round_obj.get("replays")
        if not isinstance(replays, list):
            continue
        for replay_index, replay in enumerate(replays):
            if not isinstance(replay, dict):
                continue
            frames = replay.get("frames")
            summaries.append(
                ReplayPlayerSummary(
                    round_index=round_index,
                    replay_index=replay_index,
                    frames=int(frames) if isinstance(frames, int) else None,
                    keys=tuple(sorted(str(k) for k in replay)),
                )
            )
    return len(data), tuple(summaries)


def inspect_ttrm_object(
    obj: Any,
    *,
    path: str = "<memory>",
    terms: Iterable[str] = DEFAULT_GARBAGE_TERMS,
    samples_per_term: int = 40,
) -> ReplayInventory:
    normalized_terms = tuple(dict.fromkeys(str(t).lower() for t in terms if str(t).strip()))
    key_counts: Counter[str] = Counter()
    tagged: dict[str, Counter[str]] = defaultdict(Counter)
    matches: dict[str, list[dict[str, Any]]] = {term: [] for term in normalized_terms}

    stack: list[tuple[str, Any]] = [("$", obj)]
    while stack:
        current_path, value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                key_s = str(key)
                key_counts[key_s] += 1
                child_path = _json_path(current_path, key_s)
                key_lower = key_s.lower()
                child_scalar_text = (
                    str(child).lower()
                    if isinstance(child, (str, int, float, bool)) or child is None
                    else ""
                )
                for term in normalized_terms:
                    if (term in key_lower or term in child_scalar_text) and len(matches[term]) < samples_per_term:
                        matches[term].append({"path": child_path, "value": _small_value(child)})
                if key_lower in {"type", "event", "name", "action"} and isinstance(
                    child, (str, int, float, bool)
                ):
                    tagged[key_lower][str(child)] += 1
                stack.append((child_path, child))
        elif isinstance(value, list):
            for index in range(len(value) - 1, -1, -1):
                stack.append((_json_path(current_path, index), value[index]))

    round_count, player_replays = _extract_round_replays(obj)
    top_keys = tuple(sorted(str(k) for k in obj)) if isinstance(obj, dict) else ()
    return ReplayInventory(
        path=path,
        top_level_keys=top_keys,
        round_count=round_count,
        player_replays=player_replays,
        key_counts=dict(key_counts),
        tagged_value_counts={k: dict(v) for k, v in tagged.items()},
        matching_paths={k: tuple(v) for k, v in matches.items()},
    )


def inspect_ttrm_file(
    path: str | Path,
    *,
    terms: Iterable[str] = DEFAULT_GARBAGE_TERMS,
    samples_per_term: int = 40,
) -> ReplayInventory:
    replay_path = Path(path)
    with replay_path.open("r", encoding="utf-8-sig") as handle:
        obj = json.load(handle)
    return inspect_ttrm_object(
        obj,
        path=str(replay_path),
        terms=terms,
        samples_per_term=samples_per_term,
    )
