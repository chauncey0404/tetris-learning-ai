from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-player packet parity audit for current multiplayer .ttrm "
            "garbage interaction / interaction_confirm events. Diagnostic only."
        )
    )
    p.add_argument("replay", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(r"artifacts\tetrio\parity\garbage_packet_parity.json"),
    )
    return p.parse_args()


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _stats(xs: list[float]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def _rounds(obj: Any) -> list:
    if not isinstance(obj, dict):
        return []
    replay = obj.get("replay")
    if not isinstance(replay, dict):
        return []
    rounds = replay.get("rounds")
    return rounds if isinstance(rounds, list) else []


def _payload(event: Any) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(event, dict) or event.get("type") != "ige":
        return None, None
    data = event.get("data")
    if not isinstance(data, dict):
        return None, None
    typ = data.get("type")
    inner = data.get("data")
    if typ not in ("interaction", "interaction_confirm"):
        return None, None
    if not isinstance(inner, dict) or inner.get("type") != "garbage":
        return None, None
    return str(typ), inner


def _packet_key(inner: dict[str, Any]) -> tuple:
    return (
        inner.get("cid"),
        inner.get("iid"),
        inner.get("ackiid"),
        inner.get("gameid"),
        inner.get("amt"),
        inner.get("frame"),
        inner.get("x"),
        inner.get("y"),
        inner.get("size"),
    )


def _garbage_stats(player_obj: Any) -> dict[str, Any]:
    if not isinstance(player_obj, dict):
        return {}
    replay = player_obj.get("replay")
    if not isinstance(replay, dict):
        return {}
    results = replay.get("results")
    if not isinstance(results, dict):
        return {}
    stats = results.get("stats")
    if not isinstance(stats, dict):
        return {}
    garbage = stats.get("garbage")
    return dict(garbage) if isinstance(garbage, dict) else {}


def _username_map(obj: Any) -> list[str | None]:
    users = obj.get("users") if isinstance(obj, dict) else None
    out: list[str | None] = []
    if isinstance(users, list):
        for u in users:
            if isinstance(u, dict):
                out.append(
                    str(u.get("username"))
                    if u.get("username") is not None
                    else None
                )
            else:
                out.append(None)
    return out


def main() -> None:
    args = parse_args()
    if not args.replay.is_file():
        raise SystemExit(f"Replay not found: {args.replay}")

    with args.replay.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    usernames = _username_map(obj)
    rounds = _rounds(obj)
    report_rounds = []

    all_confirm_outer_delays: list[float] = []
    all_confirm_envelope_delays: list[float] = []
    all_source_to_outer: list[float] = []
    total_interactions = 0
    total_confirms = 0
    total_matched = 0
    exact_opponent_attack_rows = 0
    comparable_opponent_attack_rows = 0

    for round_index, round_obj in enumerate(rounds):
        if not isinstance(round_obj, list):
            continue

        player_reports = []
        inbound_amounts: list[float] = []
        attacks: list[float | None] = []

        parsed_by_player: list[dict[str, Any]] = []
        for player_index, player_obj in enumerate(round_obj):
            replay = (
                player_obj.get("replay")
                if isinstance(player_obj, dict)
                else None
            )
            events = replay.get("events") if isinstance(replay, dict) else None
            if not isinstance(events, list):
                events = []

            interactions: list[dict[str, Any]] = []
            confirms: list[dict[str, Any]] = []
            for event_index, event in enumerate(events):
                typ, inner = _payload(event)
                if typ is None or inner is None:
                    continue

                data = event.get("data") if isinstance(event, dict) else {}
                item = {
                    "event_index": event_index,
                    "outer_frame": event.get("frame"),
                    "envelope_frame": (
                        data.get("frame") if isinstance(data, dict) else None
                    ),
                    "packet": dict(inner),
                    "key": _packet_key(inner),
                }
                if typ == "interaction":
                    interactions.append(item)
                else:
                    confirms.append(item)

            confirm_buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
            for c in confirms:
                confirm_buckets[c["key"]].append(c)

            matched = []
            unmatched_interactions = []
            for inter in interactions:
                bucket = confirm_buckets.get(inter["key"], [])
                if bucket:
                    conf = bucket.pop(0)
                    outer_delay = None
                    env_delay = None
                    source_to_outer = None

                    if isinstance(inter["outer_frame"], int) and isinstance(
                        conf["outer_frame"], int
                    ):
                        outer_delay = conf["outer_frame"] - inter["outer_frame"]
                        all_confirm_outer_delays.append(float(outer_delay))

                    if isinstance(inter["envelope_frame"], int) and isinstance(
                        conf["envelope_frame"], int
                    ):
                        env_delay = (
                            conf["envelope_frame"]
                            - inter["envelope_frame"]
                        )
                        all_confirm_envelope_delays.append(float(env_delay))

                    packet_frame = inter["packet"].get("frame")
                    if isinstance(inter["outer_frame"], int) and isinstance(
                        packet_frame, int
                    ):
                        source_to_outer = inter["outer_frame"] - packet_frame
                        all_source_to_outer.append(float(source_to_outer))

                    matched.append(
                        {
                            "interaction_event_index": inter["event_index"],
                            "confirm_event_index": conf["event_index"],
                            "packet": inter["packet"],
                            "interaction_outer_frame": inter["outer_frame"],
                            "interaction_envelope_frame": inter["envelope_frame"],
                            "confirm_outer_frame": conf["outer_frame"],
                            "confirm_envelope_frame": conf["envelope_frame"],
                            "confirm_outer_delay": outer_delay,
                            "confirm_envelope_delay": env_delay,
                            "packet_frame_to_interaction_outer": source_to_outer,
                        }
                    )
                else:
                    unmatched_interactions.append(inter)

            unmatched_confirms = []
            for bucket in confirm_buckets.values():
                unmatched_confirms.extend(bucket)

            inbound = sum(
                float(x["packet"].get("amt", 0) or 0)
                for x in interactions
            )
            garbage_stats = _garbage_stats(player_obj)
            attack = _num(garbage_stats.get("attack"))

            inbound_amounts.append(inbound)
            attacks.append(attack)
            total_interactions += len(interactions)
            total_confirms += len(confirms)
            total_matched += len(matched)

            parsed_by_player.append(
                {
                    "player": player_index,
                    "username": (
                        usernames[player_index]
                        if player_index < len(usernames)
                        else None
                    ),
                    "frames": replay.get("frames") if isinstance(replay, dict) else None,
                    "interaction_count": len(interactions),
                    "confirm_count": len(confirms),
                    "matched_pairs": len(matched),
                    "unmatched_interactions": len(unmatched_interactions),
                    "unmatched_confirms": len(unmatched_confirms),
                    "interaction_amount_sum": inbound,
                    "results_garbage": garbage_stats,
                    "pairs": matched,
                }
            )

        # In a 2-player round, a stream's interaction packets are hypothesized
        # to be inbound, so compare their amount sum to the *other* player's
        # final attack statistic. This is evidence only, not a semantic axiom.
        if len(parsed_by_player) == 2:
            for p in (0, 1):
                other = 1 - p
                opponent_attack = attacks[other]
                parsed_by_player[p]["opponent_attack_candidate"] = opponent_attack
                if opponent_attack is None:
                    parsed_by_player[p]["inbound_equals_opponent_attack"] = None
                    parsed_by_player[p]["inbound_minus_opponent_attack"] = None
                else:
                    comparable_opponent_attack_rows += 1
                    delta = inbound_amounts[p] - opponent_attack
                    exact = abs(delta) < 1e-9
                    exact_opponent_attack_rows += int(exact)
                    parsed_by_player[p]["inbound_equals_opponent_attack"] = exact
                    parsed_by_player[p]["inbound_minus_opponent_attack"] = delta

        report_rounds.append(
            {
                "round": round_index,
                "players": parsed_by_player,
            }
        )

    all_pairs_exact = (
        total_interactions > 0
        and total_interactions == total_confirms == total_matched
    )
    cross_attack_rate = (
        exact_opponent_attack_rows / comparable_opponent_attack_rows
        if comparable_opponent_attack_rows
        else None
    )

    if all_pairs_exact and cross_attack_rate == 1.0:
        packet_status = "STRONG_INBOUND_PACKET_ORACLE_EVIDENCE"
    elif all_pairs_exact:
        packet_status = "PAIRED_PACKET_ORACLE_WITH_CROSS_STAT_MISMATCH"
    else:
        packet_status = "PACKET_PAIRING_UNRESOLVED"

    report = {
        "format": "tetrio_ttrm_garbage_packet_parity",
        "replay": str(args.replay),
        "round_count": len(report_rounds),
        "interaction_events": total_interactions,
        "interaction_confirm_events": total_confirms,
        "matched_interaction_confirm_pairs": total_matched,
        "all_interactions_exactly_confirmed": all_pairs_exact,
        "confirm_outer_delay_frames": _stats(all_confirm_outer_delays),
        "confirm_envelope_delay_frames": _stats(all_confirm_envelope_delays),
        "packet_frame_to_interaction_outer_frames": _stats(all_source_to_outer),
        "inbound_vs_opponent_attack": {
            "comparable_streams": comparable_opponent_attack_rows,
            "exact_streams": exact_opponent_attack_rows,
            "exact_rate": cross_attack_rate,
            "interpretation": (
                "If exact, this strongly supports that a player's interaction "
                "garbage events are inbound packets generated by the opponent. "
                "A mismatch does not by itself falsify this because historical/"
                "current stats may differ in gross-vs-net attack semantics."
            ),
        },
        "rounds": report_rounds,
        "packet_oracle_status": packet_status,
        "model_input_status": {
            "incoming_garbage": "BLOCKED_PENDING_QUEUE_RECONSTRUCTION",
            "immediate_garbage": "BLOCKED",
        },
        "next_gate": (
            "Reconstruct the receiver-side pending queue from packet arrivals, "
            "then identify own cancellation and tank/insertion timing. Do not "
            "approve historical incoming_garbage from packet receipt alone."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 108)
    print("TETR.IO TTRM GARBAGE PACKET PARITY")
    print("=" * 108)
    print("Replay              :", args.replay)
    print("Rounds              :", len(report_rounds))
    print(
        "interaction/confirm:",
        f"{total_interactions}/{total_confirms}",
        f"matched={total_matched}",
    )
    print("Exact pair parity   :", all_pairs_exact)
    print(
        "Confirm outer delay:",
        report["confirm_outer_delay_frames"],
    )
    print(
        "Source->outer delay:",
        report["packet_frame_to_interaction_outer_frames"],
    )
    print()

    for rr in report_rounds:
        print(f"ROUND {rr['round']}")
        for p in rr["players"]:
            print(
                f"  P{p['player']} {p['username'] or ''}: "
                f"inbound_candidate={p['interaction_amount_sum']:.0f} "
                f"attack={p['results_garbage'].get('attack')} "
                f"opponent_attack={p.get('opponent_attack_candidate')} "
                f"cross_exact={p.get('inbound_equals_opponent_attack')} "
                f"pairs={p['matched_pairs']}/{p['interaction_count']}"
            )
        print()

    print(
        "Cross-stream exact  :",
        f"{exact_opponent_attack_rows}/{comparable_opponent_attack_rows}",
        f"rate={cross_attack_rate}",
    )
    print("Packet oracle status:", packet_status)
    print("incoming_garbage    : BLOCKED_PENDING_QUEUE_RECONSTRUCTION")
    print("immediate_garbage   : BLOCKED")
    print("Saved               :", args.output)


if __name__ == "__main__":
    main()
