from __future__ import annotations

import argparse
import json
from pathlib import Path

from tetrio.parity.replay_inventory import DEFAULT_GARBAGE_TERMS, inspect_ttrm_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a TETR.IO .ttrm JSON replay without assuming undocumented gameplay-event schema."
        )
    )
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--terms",
        default=",".join(DEFAULT_GARBAGE_TERMS),
        help="Comma-separated key substrings to inventory.",
    )
    parser.add_argument("--samples-per-term", type=int, default=40)
    args = parser.parse_args()

    terms = tuple(x.strip() for x in args.terms.split(",") if x.strip())
    inventory = inspect_ttrm_file(
        args.replay,
        terms=terms,
        samples_per_term=max(1, args.samples_per_term),
    )
    payload = inventory.to_dict()

    print("=" * 88)
    print("TETR.IO REPLAY INVENTORY")
    print("=" * 88)
    print("Replay       :", args.replay)
    print("Top keys     :", ", ".join(inventory.top_level_keys) or "<non-object root>")
    print("Rounds       :", inventory.round_count)
    print("Player replays:", len(inventory.player_replays))
    for p in inventory.player_replays[:30]:
        print(
            f"  round={p.round_index} replay={p.replay_index} "
            f"frames={p.frames} keys={','.join(p.keys)}"
        )
    print()
    for term, samples in inventory.matching_paths.items():
        print(f"[{term}] {len(samples)} sample path(s)")
        for sample in samples[:10]:
            print(" ", sample["path"], "=", sample["value"])
    if inventory.tagged_value_counts:
        print()
        print("Tagged values:")
        for field, counts in inventory.tagged_value_counts.items():
            print(" ", field, counts)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print("Saved inventory:", args.output)


if __name__ == "__main__":
    main()
