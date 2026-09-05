from __future__ import annotations

import argparse
from pathlib import Path

from tetrio.parity.trace import load_parity_trace
from tetrio.parity.validator import validate_garbage_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a normalized TETR.IO garbage oracle trace against the V9 transport engine."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--travel-frames", type=int, default=20)
    args = parser.parse_args()

    events = load_parity_trace(args.trace)
    report = validate_garbage_trace(events, default_travel_frames=args.travel_frames)
    print("=" * 88)
    print("TETR.IO GARBAGE PARITY")
    print("=" * 88)
    print("Events:", report.event_count)
    print("Result:", "PASS" if report.passed else "FAIL")
    if report.mismatches:
        for mismatch in report.mismatches:
            print(
                f"event#{mismatch.event_index} {mismatch.kind} player={mismatch.player} "
                f"field={mismatch.field}: expected={mismatch.expected!r} actual={mismatch.actual!r}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
