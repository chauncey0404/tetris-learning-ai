from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

from tetrio.control.keymap import Action, DEFAULT_KEYMAP, KeyBinding

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

if os.name == "nt":
    ULONG_PTR = wintypes.WPARAM

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class INPUTUNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", INPUTUNION),
        ]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _SendInput = _user32.SendInput
    _SendInput.argtypes = (
        wintypes.UINT,
        ctypes.POINTER(INPUT),
        ctypes.c_int,
    )
    _SendInput.restype = wintypes.UINT
else:
    ULONG_PTR = ctypes.c_size_t
    MOUSEINPUT = object
    KEYBDINPUT = object
    HARDWAREINPUT = object
    INPUTUNION = object
    INPUT = object
    _SendInput = None

@dataclass(frozen=True)
class InputTiming:
    tap_hold_seconds: float = 0.025
    inter_key_seconds: float = 0.018

def windows_input_abi() -> dict[str, int | str | bool]:
    return {
        "platform": os.name,
        "pointer_size": ctypes.sizeof(ctypes.c_void_p),
        "input_size": ctypes.sizeof(INPUT) if os.name == "nt" else -1,
        "keyboard_input_size": ctypes.sizeof(KEYBDINPUT) if os.name == "nt" else -1,
        "is_64_bit": ctypes.sizeof(ctypes.c_void_p) == 8,
    }

class WindowsInputController:
    def __init__(
        self,
        *,
        keymap: Mapping[Action, KeyBinding] = DEFAULT_KEYMAP,
        timing: InputTiming = InputTiming(),
        dry_run: bool = False,
    ) -> None:
        self.keymap = dict(keymap)
        self.timing = timing
        self.dry_run = bool(dry_run)
        if not self.dry_run and os.name != "nt":
            raise RuntimeError(
                "WindowsInputController requires Windows unless dry_run=True."
            )

    def _emit(self, binding: KeyBinding, *, key_up: bool) -> None:
        flags = KEYEVENTF_SCANCODE
        if binding.extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        if key_up:
            flags |= KEYEVENTF_KEYUP

        if self.dry_run:
            state = "UP" if key_up else "DOWN"
            print(f"[DRY] {binding.name} {state}")
            return

        assert _SendInput is not None
        event = INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(
                wVk=0,
                wScan=int(binding.scan_code),
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            ),
        )

        ctypes.set_last_error(0)
        sent = _SendInput(
            1,
            ctypes.byref(event),
            ctypes.sizeof(INPUT),
        )
        if sent != 1:
            error = ctypes.get_last_error()
            abi = windows_input_abi()
            raise OSError(
                error,
                (
                    f"SendInput failed for {binding.name}; "
                    f"INPUT size={abi['input_size']} "
                    f"KEYBDINPUT size={abi['keyboard_input_size']} "
                    f"ptr={abi['pointer_size']}. "
                    "If the simulator is running as Administrator, "
                    "run this terminal at the same privilege level."
                ),
            )

    def key_down(self, action: Action | str) -> None:
        action = Action(action)
        self._emit(self.keymap[action], key_up=False)

    def key_up(self, action: Action | str) -> None:
        action = Action(action)
        self._emit(self.keymap[action], key_up=True)

    def tap(
        self,
        action: Action | str,
        *,
        hold_seconds: float | None = None,
    ) -> None:
        action = Action(action)
        delay = (
            self.timing.tap_hold_seconds
            if hold_seconds is None
            else max(0.0, float(hold_seconds))
        )
        self.key_down(action)
        if delay > 0:
            time.sleep(delay)
        self.key_up(action)

    def tap_many(
        self,
        actions: Iterable[Action | str],
        *,
        inter_key_seconds: float | None = None,
    ) -> None:
        delay = (
            self.timing.inter_key_seconds
            if inter_key_seconds is None
            else max(0.0, float(inter_key_seconds))
        )
        first = True
        for action in actions:
            if not first and delay > 0:
                time.sleep(delay)
            self.tap(action)
            first = False

    def release_all(self) -> None:
        for action in self.keymap:
            try:
                self.key_up(action)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release_all()
        return False
