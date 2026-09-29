from __future__ import annotations

import numpy as np
import torch


FEATURE_NAMES = (
    "holes_after",
    "hole_delta",
    "max_height_after",
    "aggregate_height_after",
    "bumpiness_after",
    "max_well_after",
    "lines_now",
    "next_candidate_count",
    "next_min_holes",
    "next_best_hole_delta",
    "next_min_height",
    "next_min_bumpiness",
    "next_max_lines",
    "next_no_new_hole_fraction",
    "t_active_next",
    "t_hold_next",
    "t_preview1",
    "t_preview2",
    "t_proxy_before",
    "t_proxy_after",
    "t_exact_full_after",
    "t_exact_mini_after",
    "t_opportunity_destroyed",
    "t_opportunity_created",
    "current_tspin_full",
    "current_tspin_mini",
    "current_tspin_lines",
    "t_cashout_deferred",
    "next_dead_end",
    "use_hold",
)
FEATURE_SIZE = len(FEATURE_NAMES)
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# These columns are useful for offline diagnostics, but path-sensitive exact-T
# reconstruction is far too expensive to run for every shortlisted candidate
# in a bulk cache. V1.1 bulk training intentionally uses the fast T-slot proxy
# and neutralizes exact-only columns so old exact shards and new fast shards are
# numerically compatible.
MODEL_NEUTRAL_FEATURES = (
    "t_exact_full_after",
    "t_exact_mini_after",
    "current_tspin_full",
    "current_tspin_mini",
    "current_tspin_lines",
)
_MODEL_NEUTRAL_INDEX = tuple(_FEATURE_INDEX[name] for name in MODEL_NEUTRAL_FEATURES)

# Fixed, transparent normalization. Values are clipped after division so no
# training-set statistics leak from held-out/test data.
_FEATURE_SCALE = np.asarray(
    [
        20.0,   # holes_after
        8.0,    # hole_delta
        20.0,   # max_height_after
        160.0,  # aggregate_height_after
        50.0,   # bumpiness_after
        12.0,   # max_well_after
        4.0,    # lines_now
        120.0,  # next_candidate_count
        20.0,   # next_min_holes
        8.0,    # next_best_hole_delta
        20.0,   # next_min_height
        50.0,   # next_min_bumpiness
        4.0,    # next_max_lines
        1.0,    # next_no_new_hole_fraction
        1.0, 1.0, 1.0, 1.0,
        8.0,    # t_proxy_before
        8.0,    # t_proxy_after
        4.0,    # t_exact_full_after
        4.0,    # t_exact_mini_after
        1.0, 1.0,
        1.0, 1.0,
        4.0,    # current_tspin_lines
        1.0,
        1.0,
        1.0,
    ],
    dtype=np.float32,
)


def feature_index(name: str) -> int:
    return _FEATURE_INDEX[name]


def normalize_future_features_numpy(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[-1] != FEATURE_SIZE:
        raise ValueError(
            f"future features last dimension must be {FEATURE_SIZE}; got {arr.shape}"
        )
    out = np.clip(arr / _FEATURE_SCALE, -3.0, 3.0).astype(
        np.float32,
        copy=True,
    )
    if _MODEL_NEUTRAL_INDEX:
        out[..., list(_MODEL_NEUTRAL_INDEX)] = 0.0
    return out


def normalize_future_features_torch(features: torch.Tensor) -> torch.Tensor:
    if features.shape[-1] != FEATURE_SIZE:
        raise ValueError(
            f"future features last dimension must be {FEATURE_SIZE}; "
            f"got {tuple(features.shape)}"
        )
    scale = torch.as_tensor(
        _FEATURE_SCALE,
        dtype=features.dtype,
        device=features.device,
    )
    out = (features / scale).clamp(-3.0, 3.0)
    if _MODEL_NEUTRAL_INDEX:
        out = out.clone()
        out[..., list(_MODEL_NEUTRAL_INDEX)] = 0.0
    return out
