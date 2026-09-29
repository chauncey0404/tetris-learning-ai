from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from tetrio.network.encoding import (
    CANDIDATE_SIZE,
    dense_candidate_batch,
    dense_state_batch,
)


@dataclass(frozen=True)
class ExpertV0Batch:
    state: np.ndarray
    candidates: np.ndarray
    candidate_mask: np.ndarray
    expert_index: np.ndarray
    use_hold: np.ndarray


@dataclass(frozen=True)
class ExpertV0CompactBatch:
    # State side: still packed/typed exactly as stored in the cache.
    state_board_packed: np.ndarray
    state_active: np.ndarray
    state_hold: np.ndarray
    state_preview: np.ndarray

    # Ragged candidates are flattened in sample-major order.  owner/local map
    # each flat score back to [B,K] only after GPU scoring.
    candidate_board_packed: np.ndarray
    candidate_piece: np.ndarray
    candidate_rotation: np.ndarray
    candidate_x: np.ndarray
    candidate_y: np.ndarray
    candidate_use_hold: np.ndarray
    candidate_lines: np.ndarray
    candidate_owner: np.ndarray
    candidate_local: np.ndarray
    candidate_counts: np.ndarray
    max_candidates: int

    expert_index: np.ndarray
    use_hold: np.ndarray


def shard_paths(cache_dir: str | Path) -> list[Path]:
    paths = sorted(Path(cache_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No Expert-v0 shards found in {cache_dir}")
    return paths


def load_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _candidate_gather_indices(
    offsets: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize ragged candidate gathering for an arbitrary sample order."""
    starts = offsets[sample_indices]
    counts = offsets[sample_indices + 1] - starts
    counts = counts.astype(np.int64, copy=False)
    total = int(counts.sum())

    owner = np.repeat(
        np.arange(len(sample_indices), dtype=np.int32),
        counts,
    )
    prefix = np.cumsum(counts, dtype=np.int64) - counts
    local = np.arange(total, dtype=np.int64) - np.repeat(prefix, counts)
    gather = np.repeat(starts, counts) + local

    return (
        gather.astype(np.int64, copy=False),
        owner,
        local.astype(np.int32, copy=False),
        counts.astype(np.int32, copy=False),
    )


def compact_batches_from_shard(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[ExpertV0CompactBatch]:
    """Yield cache-native compact batches without CPU dense feature expansion."""
    n = int(data["expert_index"].shape[0])
    order = np.arange(n, dtype=np.int64)
    if rng is not None:
        rng.shuffle(order)

    offsets = data["candidate_offsets"].astype(np.int64, copy=False)

    for start in range(0, n, int(batch_size)):
        indices = order[start : start + int(batch_size)]
        gather, owner, local, counts = _candidate_gather_indices(offsets, indices)

        yield ExpertV0CompactBatch(
            state_board_packed=np.ascontiguousarray(data["state_board_packed"][indices]),
            state_active=np.ascontiguousarray(data["state_active"][indices]),
            state_hold=np.ascontiguousarray(data["state_hold"][indices]),
            state_preview=np.ascontiguousarray(data["state_preview"][indices]),
            candidate_board_packed=np.ascontiguousarray(data["candidate_board_packed"][gather]),
            candidate_piece=np.ascontiguousarray(data["candidate_piece"][gather]),
            candidate_rotation=np.ascontiguousarray(data["candidate_rotation"][gather]),
            candidate_x=np.ascontiguousarray(data["candidate_x"][gather]),
            candidate_y=np.ascontiguousarray(data["candidate_y"][gather]),
            candidate_use_hold=np.ascontiguousarray(data["candidate_use_hold"][gather]),
            candidate_lines=np.ascontiguousarray(data["candidate_lines"][gather]),
            candidate_owner=owner,
            candidate_local=local,
            candidate_counts=counts,
            max_candidates=int(counts.max()),
            expert_index=np.ascontiguousarray(
                data["expert_index"][indices].astype(np.int64, copy=False)
            ),
            use_hold=np.ascontiguousarray(
                data["use_hold"][indices].astype(np.float32, copy=False)
            ),
        )


def batches_from_shard(
    data: dict[str, np.ndarray],
    *,
    batch_size: int,
    rng: np.random.Generator | None,
) -> Iterator[ExpertV0Batch]:
    """Dense compatibility path, now vectorized across all batch candidates.

    This preserves the original API but removes the old per-sample Python loop.
    """
    n = int(data["expert_index"].shape[0])
    order = np.arange(n, dtype=np.int64)
    if rng is not None:
        rng.shuffle(order)

    offsets = data["candidate_offsets"].astype(np.int64, copy=False)

    for start in range(0, n, int(batch_size)):
        indices = order[start : start + int(batch_size)]
        gather, owner, local, counts = _candidate_gather_indices(offsets, indices)
        b = len(indices)
        max_k = int(counts.max())

        state = dense_state_batch(
            data["state_board_packed"][indices],
            data["state_active"][indices],
            data["state_hold"][indices],
            data["state_preview"][indices],
        )

        dense = dense_candidate_batch(
            data["candidate_board_packed"][gather],
            data["candidate_piece"][gather],
            data["candidate_rotation"][gather],
            data["candidate_x"][gather],
            data["candidate_y"][gather],
            data["candidate_use_hold"][gather],
            data["candidate_lines"][gather],
        )

        candidates = np.zeros((b, max_k, CANDIDATE_SIZE), dtype=np.float32)
        candidates[owner, local] = dense

        mask = np.zeros((b, max_k), dtype=np.bool_)
        mask[owner, local] = True

        yield ExpertV0Batch(
            state=state,
            candidates=candidates,
            candidate_mask=mask,
            expert_index=data["expert_index"][indices].astype(np.int64, copy=False),
            use_hold=data["use_hold"][indices].astype(np.float32, copy=False),
        )
