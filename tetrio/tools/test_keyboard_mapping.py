from __future__ import annotations
import argparse
import time
from tetrio.control.input_controller import InputTiming, WindowsInputController
from tetrio.control.keymap import Action, describe_keymap

TEST_SEQUENCE = (
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.SOFT_DROP,
    Action.HARD_DROP,
    Action.ROTATE_CCW,
    Action.ROTATE_CW,
    Action.ROTATE_180,
    Action.HOLD,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--tap-ms", type=float, default=25.0)
    p.add_argument("--gap-ms", type=float, default=180.0)
    return p.parse_args()

def main():
    args = parse_args()
    print("=" * 72)
    print("TETR.IO SIMULATOR KEYBOARD MAPPING")
    print("=" * 72)
    print(describe_keymap())
    print()

    if args.live:
        print("LIVE MODE: focus your simulator window now.")
        for remaining in range(max(0, args.countdown), 0, -1):
            print(f"Starting in {remaining}...")
            time.sleep(1.0)
    else:
        print("DRY RUN: no real keyboard input will be sent.")

    timing = InputTiming(
        tap_hold_seconds=max(0.0, args.tap_ms / 1000.0),
        inter_key_seconds=max(0.0, args.gap_ms / 1000.0),
    )
    controller = WindowsInputController(
        timing=timing,
        dry_run=not args.live,
    )

    labels = {
        Action.MOVE_LEFT: "LEFT / A",
        Action.MOVE_RIGHT: "RIGHT / D",
        Action.SOFT_DROP: "SOFT DROP / W",
        Action.HARD_DROP: "HARD DROP / S",
        Action.ROTATE_CCW: "ROTATE CCW / LEFT ARROW",
        Action.ROTATE_CW: "ROTATE CW / RIGHT ARROW",
        Action.ROTATE_180: "ROTATE 180 / UP ARROW",
        Action.HOLD: "HOLD / LEFT SHIFT",
    }

    try:
        with controller:
            for i, action in enumerate(TEST_SEQUENCE, 1):
                print(f"[{i}/8] {labels[action]}")
                controller.tap(action)
                if i != len(TEST_SEQUENCE):
                    time.sleep(timing.inter_key_seconds)
    except KeyboardInterrupt:
        controller.release_all()
        raise

    print("Mapping test complete.")

if __name__ == "__main__":
    main()
