"""Future-aware structural/tactical analysis for TETR.IO Expert-v1.1."""

from .features import (
    FEATURE_NAMES,
    FEATURE_SIZE,
    feature_index,
    normalize_future_features_numpy,
    normalize_future_features_torch,
)
from .state_transition import KnownFutureState, advance_after_lock
from .lookahead import FutureCandidateInput, FutureFeatureConfig, build_row_future_features

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SIZE",
    "feature_index",
    "normalize_future_features_numpy",
    "normalize_future_features_torch",
    "KnownFutureState",
    "advance_after_lock",
    "FutureCandidateInput",
    "FutureFeatureConfig",
    "build_row_future_features",
]
