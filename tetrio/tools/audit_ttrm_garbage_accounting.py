from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit end-of-round garbage accounting in a current multiplayer "
            ".ttrm. Compares raw inbound interaction packets against final "
            "garbage stats without assuming undocumented semantics."
        )
    )
    p.add_argument("replay", type=Path)
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"artifacts\tetrio\parity\garbage_accounting.json"
        ),
    )
    return p.parse_args()


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        if math.isfinite(x):
            return x
    return None


def _rounds(obj: Any) -> list:
    if not isinstance(obj, dict):
        return []
    replay = obj.get("replay")
    if not isinstance(replay, dict):
        return []
    rounds = replay.get("rounds")
    return rounds if isinstance(rounds, list) else []


def _users(obj: Any) -> list[str | None]:
    raw = obj.get("users") if isinstance(obj, dict) else None
    out: list[str | None] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, dict) and item.get("username") is not None:
            out.append(str(item["username"]))
        else:
            out.append(None)
    return out


def _replay(player_obj: Any) -> dict[str, Any]:
    if not isinstance(player_obj, dict):
        return {}
    rep = player_obj.get("replay")
    return rep if isinstance(rep, dict) else {}


def _garbage_stats(player_obj: Any) -> dict[str, Any]:
    rep = _replay(player_obj)
    results = rep.get("results")
    if not isinstance(results, dict):
        return {}
    stats = results.get("stats")
    if not isinstance(stats, dict):
        return {}
    garbage = stats.get("garbage")
    return dict(garbage) if isinstance(garbage, dict) else {}


def _inbound_packets(player_obj: Any) -> list[dict[str, Any]]:
    rep = _replay(player_obj)
    events = rep.get("events")
    if not isinstance(events, list):
        return []

    out = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("type") != "ige":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or data.get("type") != "interaction":
            continue
        inner = data.get("data")
        if not isinstance(inner, dict) or inner.get("type") != "garbage":
            continue

        amt = _num(inner.get("amt"))
        if amt is None:
            continue
        out.append(
            {
                "event_index": event_index,
                "outer_frame": event.get("frame"),
                "envelope_frame": data.get("frame"),
                "packet_frame": inner.get("frame"),
                "amt": amt,
                "cid": inner.get("cid"),
                "iid": inner.get("iid"),
                "ackiid": inner.get("ackiid"),
                "gameid": inner.get("gameid"),
                "x": inner.get("x"),
                "y": inner.get("y"),
                "size": inner.get("size"),
            }
        )
    return out


def _eq(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None:
        return None
    return abs(a - b) < 1e-9


def main() -> None:
    args = parse_args()
    if not args.replay.is_file():
        raise SystemExit(f"Replay not found: {args.replay}")

    with args.replay.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)

    usernames = _users(obj)
    round_reports = []

    sent_exact = 0
    sent_comparable = 0
    conservation_exact = 0
    conservation_comparable = 0
    all_nonnegative_cancel = True

    for round_index, round_obj in enumerate(_rounds(obj)):
        if not isinstance(round_obj, list):
            continue

        streams = []
        for player_index, player_obj in enumerate(round_obj):
            packets = _inbound_packets(player_obj)
            inbound = sum(p["amt"] for p in packets)
            stats = _garbage_stats(player_obj)

            streams.append(
                {
                    "player": player_index,
                    "username": (
                        usernames[player_index]
                        if player_index < len(usernames)
                        else None
                    ),
                    "inbound_packets": packets,
                    "inbound_total": inbound,
                    "stats": stats,
                }
            )

        # Current replay is 1v1; compare each player's net-outbound candidate
        # with the other player's inbound interaction total.
        if len(streams) == 2:
            for p in (0, 1):
                other = 1 - p
                stats = streams[p]["stats"]

                attack = _num(stats.get("attack"))
                sent = _num(stats.get("sent"))
                sent_nomult = _num(stats.get("sent_nomult"))
                received = _num(stats.get("received"))
                cleared = _num(stats.get("cleared"))

                observed_outbound = streams[other]["inbound_total"]
                sent_match = _eq(sent, observed_outbound)

                if sent_match is not None:
                    sent_comparable += 1
                    sent_exact += int(sent_match)

                inferred_cancel = (
                    None
                    if attack is None or sent is None
                    else attack - sent
                )
                if inferred_cancel is not None and inferred_cancel < -1e-9:
                    all_nonnegative_cancel = False

                # Candidate end-of-round conservation:
                # inbound = cancellation + actually received/tanked + residual queue
                # where attack-sent is only a *candidate* cancellation total.
                residual = None
                conservation_ok = None
                if (
                    received is not None
                    and inferred_cancel is not None
                ):
                    residual = (
                        streams[p]["inbound_total"]
                        - inferred_cancel
                        - received
                    )
                    # We cannot require residual==0 because a round can terminate
                    # with pending queue. We can only require non-negative
                    # conservation under this candidate interpretation.
                    conservation_ok = residual >= -1e-9
                    conservation_comparable += 1
                    conservation_exact += int(conservation_ok)

                streams[p].update(
                    {
                        "observed_outbound_from_opponent_stream": observed_outbound,
                        "sent_equals_observed_outbound": sent_match,
                        "attack_minus_sent_candidate_cancelled": inferred_cancel,
                        "candidate_end_pending": residual,
                        "candidate_conservation_nonnegative": conservation_ok,
                        "sent_nomult": sent_nomult,
                        "received": received,
                        "cleared": cleared,
                    }
                )

        round_reports.append(
            {
                "round": round_index,
                "players": streams,
            }
        )

    sent_rate = (
        sent_exact / sent_comparable
        if sent_comparable
        else None
    )
    conservation_rate = (
        conservation_exact / conservation_comparable
        if conservation_comparable
        else None
    )

    if sent_rate == 1.0 and all_nonnegative_cancel:
        status = "STRONG_NET_SENT_ACCOUNTING_EVIDENCE"
    elif sent_rate is not None and sent_rate >= 0.8:
        status = "PARTIAL_NET_SENT_ACCOUNTING_EVIDENCE"
    else:
        status = "ACCOUNTING_UNRESOLVED"

    report = {
        "format": "tetrio_ttrm_garbage_accounting",
        "replay": str(args.replay),
        "rounds": round_reports,
        "sent_vs_observed_outbound": {
            "exact": sent_exact,
            "comparable": sent_comparable,
            "exact_rate": sent_rate,
        },
        "candidate_conservation": {
            "nonnegative": conservation_exact,
            "comparable": conservation_comparable,
            "rate": conservation_rate,
            "all_attack_minus_sent_nonnegative": all_nonnegative_cancel,
        },
        "status": status,
        "interpretation_limits": [
            "stats.garbage.attack is treated only as a gross-attack candidate",
            "stats.garbage.sent is treated only as a net-sent candidate",
            "attack-sent is not yet approved as exact cancellation timing",
            "received is not yet approved as exact tank timing",
            "candidate_end_pending may be non-zero at round termination",
        ],
        "model_input_status": {
            "incoming_garbage": "BLOCKED_PENDING_DECISION_TIME_QUEUE_PARITY",
            "immediate_garbage": "BLOCKED",
        },
        "next_gate": (
            "Use packet chronology plus a controlled replay or engine replay "
            "simulation to locate cancellation and tank events at exact frames."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 112)
    print("TETR.IO TTRM GARBAGE END-OF-ROUND ACCOUNTING")
    print("=" * 112)
    print("Replay :", args.replay)
    print()

    for rr in round_reports:
        print(f"ROUND {rr['round']}")
        for p in rr["players"]:
            s = p["stats"]
            print(
                f"  P{p['player']} {p['username'] or ''}: "
                f"in={p['inbound_total']:.0f} "
                f"attack={s.get('attack')} "
                f"sent={s.get('sent')} "
                f"sent_nomult={s.get('sent_nomult')} "
                f"received={s.get('received')} "
                f"cleared={s.get('cleared')} "
                f"| observed_out={p.get('observed_outbound_from_opponent_stream')} "
                f"sent_match={p.get('sent_equals_observed_outbound')} "
                f"attack-sent={p.get('attack_minus_sent_candidate_cancelled')} "
                f"end_pending?={p.get('candidate_end_pending')}"
            )
        print()

    print(
        "sent == opponent inbound : "
        f"{sent_exact}/{sent_comparable} rate={sent_rate}"
    )
    print(
        "candidate conservation   : "
        f"{conservation_exact}/{conservation_comparable} "
        f"nonnegative rate={conservation_rate}"
    )
    print("Status                  :", status)
    print("incoming_garbage        : BLOCKED_PENDING_DECISION_TIME_QUEUE_PARITY")
    print("immediate_garbage       : BLOCKED")
    print("Saved                   :", args.output)


if __name__ == "__main__":
    main()
