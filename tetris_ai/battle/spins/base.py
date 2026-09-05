from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np

from tetris_ai.battle.rules.base import MovementRuleset
from tetris_ai.battle.types import ReachablePlacement, RotationTrace


class SpinKind(str, Enum):
    NONE = "none"
    MINI = "mini"
    FULL = "full"


@dataclass(frozen=True)
class SpinResult:
    kind: SpinKind
    piece: str
    lines_cleared: int = 0
    corner_count: int = 0
    front_corner_count: int = 0
    immobile: bool = False
    rotation: RotationTrace | None = None
    reason: str = ""

    @property
    def is_spin(self) -> bool:
        return self.kind is not SpinKind.NONE

    @property
    def is_mini(self) -> bool:
        return self.kind is SpinKind.MINI

    @property
    def is_full(self) -> bool:
        return self.kind is SpinKind.FULL


class SpinSystem(ABC):
    name = "abstract-spin-system"

    @abstractmethod
    def classify(
        self,
        board_before_lock: np.ndarray,
        placement: ReachablePlacement,
        movement_ruleset: MovementRuleset,
        *,
        lines_cleared: int = 0,
    ) -> SpinResult:
        raise NotImplementedError
