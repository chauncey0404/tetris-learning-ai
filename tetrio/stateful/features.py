from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


STATEFUL_FEATURE_NAMES = (
    "raw_combo_before",
    "raw_btb_before",
    "previous_cleared",
    "previous_t_spin_any",
    "previous_t_spin_mini",
    "previous_attack",
    "previous_garbage_cleared",
)
STATEFUL_FEATURE_SIZE = len(STATEFUL_FEATURE_NAMES)

_FALSE_SPIN = {
    "", "0", "false", "n", "no", "none", "null", "normal",
}


def _required_number(value: Any, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is NULL but V1.2A requires the audited field")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} is not finite: {value!r}")
    return out


def spin_flags(value: Any) -> tuple[float, float]:
    text = "" if value is None else str(value).strip().lower()
    if text in _FALSE_SPIN:
        return 0.0, 0.0
    return 1.0, float("mini" in text)


def encode_battle_state_row(
    *,
    raw_combo_before: Any,
    raw_btb_before: Any,
    previous_cleared: Any,
    previous_t_spin: Any,
    previous_attack: Any,
    previous_garbage_cleared: Any,
) -> np.ndarray:
    spin_any, spin_mini = spin_flags(previous_t_spin)
    out = np.asarray(
        [
            _required_number(raw_combo_before, name="raw_combo_before"),
            _required_number(raw_btb_before, name="raw_btb_before"),
            _required_number(previous_cleared, name="previous_cleared"),
            spin_any,
            spin_mini,
            _required_number(previous_attack, name="previous_attack"),
            _required_number(
                previous_garbage_cleared,
                name="previous_garbage_cleared",
            ),
        ],
        dtype=np.float32,
    )
    if out.shape != (STATEFUL_FEATURE_SIZE,):
        raise RuntimeError(f"stateful feature width mismatch: {out.shape}")
    return out


def normalize_stateful_features_numpy(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[-1] != STATEFUL_FEATURE_SIZE:
        raise ValueError(
            f"stateful features last dim must be {STATEFUL_FEATURE_SIZE}; "
            f"got {arr.shape}"
        )
    out = np.empty_like(arr, dtype=np.float32)
    out[..., 0] = np.tanh(arr[..., 0] / 4.0)
    out[..., 1] = np.tanh(arr[..., 1] / 4.0)
    out[..., 2] = np.clip(arr[..., 2] / 4.0, 0.0, 1.0)
    out[..., 3] = np.clip(arr[..., 3], 0.0, 1.0)
    out[..., 4] = np.clip(arr[..., 4], 0.0, 1.0)
    out[..., 5] = np.tanh(arr[..., 5] / 8.0)
    out[..., 6] = np.tanh(arr[..., 6] / 8.0)
    return out


def normalize_stateful_features_torch(features: torch.Tensor) -> torch.Tensor:
    if features.shape[-1] != STATEFUL_FEATURE_SIZE:
        raise ValueError(
            f"stateful features last dim must be {STATEFUL_FEATURE_SIZE}; "
            f"got {tuple(features.shape)}"
        )
    out = torch.empty_like(features)
    out[..., 0] = torch.tanh(features[..., 0] / 4.0)
    out[..., 1] = torch.tanh(features[..., 1] / 4.0)
    out[..., 2] = (features[..., 2] / 4.0).clamp(0.0, 1.0)
    out[..., 3] = features[..., 3].clamp(0.0, 1.0)
    out[..., 4] = features[..., 4].clamp(0.0, 1.0)
    out[..., 5] = torch.tanh(features[..., 5] / 8.0)
    out[..., 6] = torch.tanh(features[..., 6] / 8.0)
    return out
