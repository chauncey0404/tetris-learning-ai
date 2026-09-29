from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EarlyStoppingTracker:
    """Track meaningful validation improvement and request an early stop.

    ``min_delta`` controls whether an improvement is large enough to reset
    patience.  The training loop may still save a numerically better checkpoint
    even when the gain is smaller than ``min_delta``.

    ``patience <= 0`` disables early stopping.
    """

    patience: int = 3
    min_delta: float = 0.001
    mode: Literal["max", "min"] = "max"

    anchor_value: float | None = None
    anchor_epoch: int | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if self.mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def _meaningfully_better(self, value: float) -> bool:
        if self.anchor_value is None:
            return True
        if self.mode == "max":
            return value >= self.anchor_value + self.min_delta
        return value <= self.anchor_value - self.min_delta

    def update(self, value: float, epoch: int) -> bool:
        """Record one validation value and return True when training should stop."""
        value = float(value)
        epoch = int(epoch)

        if not self.enabled:
            return False

        if self._meaningfully_better(value):
            self.anchor_value = value
            self.anchor_epoch = epoch
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
