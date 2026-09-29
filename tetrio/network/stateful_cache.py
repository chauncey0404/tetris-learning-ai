from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from tetrio.network.future_cache import FutureBatch, _gather_rows
from tetrio.stateful.features import STATEFUL_FEATURE_SIZE


def stateful_shard_paths(cache_dir: str | Path) -> list[Path]:
    paths = sorted(Path(cache_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No V1.2 state-cache shards found in {cache_dir}")
    return paths


def load_stateful_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


@dataclass(frozen=True)
class StatefulFutureBatch:
    future: FutureBatch
    battle_state: np.ndarray


def validate_stateful_pair(
    future: dict[str, np.ndarray],
    stateful: dict[str, np.ndarray],
) -> None:
    n = int(future["expert_local"].shape[0])
    features = np.asarray(stateful["state_features"], dtype=np.float32)
    if features.shape != (n, STATEFUL_FEATURE_SIZE):
        raise RuntimeError(
            "V1.2 state/future row mismatch: "
            f"future={n}, state_features={features.shape}"
        )
    for key in ("game_id", "subframe"):
        if key not in future or key not in stateful:
            raise RuntimeError(f"missing row identity column: {key}")
        if not np.array_equal(
            np.asarray(future[key], dtype=np.int64),
            np.asarray(stateful[key], dtype=np.int64),
        ):
            raise RuntimeError(f"V1.2 state/future {key} mismatch")


def batches_from_stateful_pair(
    *,
    future: dict[str, np.ndarray],
    stateful: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[StatefulFutureBatch]:
    validate_stateful_pair(future, stateful)
    n = int(future["expert_local"].shape[0])
    order = np.arange(n, dtype=np.int64)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, n, int(batch_size)):
        idx = order[start:start + int(batch_size)]
        yield StatefulFutureBatch(
            future=_gather_rows(future, idx),
            battle_state=np.ascontiguousarray(
                stateful["state_features"][idx].astype(np.float32, copy=False)
            ),
        )
