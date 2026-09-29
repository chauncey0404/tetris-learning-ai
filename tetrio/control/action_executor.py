from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
from tetrio.control.input_controller import WindowsInputController
from tetrio.control.keymap import Action

@dataclass(frozen=True)
class PlannedInput:
    action: Action

class ActionExecutor:
    def __init__(self, controller: WindowsInputController) -> None:
        self.controller = controller

    def execute(
        self,
        path: Iterable[Action | PlannedInput],
    ) -> None:
        for item in path:
            action = (
                item.action
                if isinstance(item, PlannedInput)
                else Action(item)
            )
            self.controller.tap(action)
