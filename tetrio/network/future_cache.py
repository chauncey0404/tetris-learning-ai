from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from tetrio.future.features import FEATURE_SIZE


def future_shard_paths(cache_dir: str | Path) -> list[Path]:
    paths = sorted(Path(cache_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No future-cache shards found in {cache_dir}")
    return paths


def load_future_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


@dataclass(frozen=True)
class FutureBatch:
    base_scores: np.ndarray
    raw_features: np.ndarray
    candidate_use_hold: np.ndarray
    inference_mask: np.ndarray
    valid_mask: np.ndarray
    expert_local: np.ndarray
    expert_use_hold: np.ndarray
    expert_in_shortlist: np.ndarray


def _gather_rows(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> FutureBatch:
    offsets = data["candidate_offsets"].astype(np.int64, copy=False)
    counts = offsets[indices + 1] - offsets[indices]
    k = int(counts.max())
    b = len(indices)

    base = np.full((b, k), -1e9, dtype=np.float32)
    feat = np.zeros((b, k, FEATURE_SIZE), dtype=np.float32)
    hold = np.zeros((b, k), dtype=np.uint8)
    inf = np.zeros((b, k), dtype=bool)
    valid = np.zeros((b, k), dtype=bool)

    for out_row, row in enumerate(indices):
        lo = int(offsets[row])
        hi = int(offsets[row + 1])
        n = hi - lo
        base[out_row, :n] = data["base_score"][lo:hi]
        feat[out_row, :n] = data["features"][lo:hi]
        hold[out_row, :n] = data["candidate_use_hold"][lo:hi]
        inf[out_row, :n] = data["inference_mask"][lo:hi].astype(bool)
        valid[out_row, :n] = True

    return FutureBatch(
        base_scores=base,
        raw_features=feat,
        candidate_use_hold=hold,
        inference_mask=inf,
        valid_mask=valid,
        expert_local=np.asarray(data["expert_local"][indices], dtype=np.int64),
        expert_use_hold=np.asarray(data["expert_use_hold"][indices], dtype=np.uint8),
        expert_in_shortlist=np.asarray(
            data["expert_in_shortlist"][indices],
            dtype=np.uint8,
        ),
    )


def batches_from_future_shard(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[FutureBatch]:
    n = int(data["expert_local"].shape[0])
    order = np.arange(n, dtype=np.int64)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, n, int(batch_size)):
        yield _gather_rows(data, order[start:start + int(batch_size)])
