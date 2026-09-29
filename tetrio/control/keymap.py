from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

class Action(str, Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    SOFT_DROP = "soft_drop"
    HARD_DROP = "hard_drop"
    ROTATE_CCW = "rotate_ccw"
    ROTATE_CW = "rotate_cw"
    ROTATE_180 = "rotate_180"
    HOLD = "hold"

@dataclass(frozen=True)
class KeyBinding:
    name: str
    scan_code: int
    extended: bool = False

A = KeyBinding("A", 0x1E)
D = KeyBinding("D", 0x20)
W = KeyBinding("W", 0x11)
S = KeyBinding("S", 0x1F)
LEFT_ARROW = KeyBinding("LEFT", 0x4B, extended=True)
RIGHT_ARROW = KeyBinding("RIGHT", 0x4D, extended=True)
UP_ARROW = KeyBinding("UP", 0x48, extended=True)
LEFT_SHIFT = KeyBinding("LSHIFT", 0x2A)

DEFAULT_KEYMAP: Mapping[Action, KeyBinding] = {
    Action.MOVE_LEFT: A,
    Action.MOVE_RIGHT: D,
    Action.SOFT_DROP: W,
    Action.HARD_DROP: S,
    Action.ROTATE_CCW: LEFT_ARROW,
    Action.ROTATE_CW: RIGHT_ARROW,
    Action.ROTATE_180: UP_ARROW,
    Action.HOLD: LEFT_SHIFT,
}

def describe_keymap(
    keymap: Mapping[Action, KeyBinding] = DEFAULT_KEYMAP,
) -> str:
    order = (
        Action.MOVE_LEFT,
        Action.MOVE_RIGHT,
        Action.SOFT_DROP,
        Action.HARD_DROP,
        Action.ROTATE_CCW,
        Action.ROTATE_CW,
        Action.ROTATE_180,
        Action.HOLD,
    )
    return "\n".join(
        f"{action.value:12s} -> {keymap[action].name}"
        for action in order
    )
