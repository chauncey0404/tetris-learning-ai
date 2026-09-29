from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PIECES = ("I", "O", "T", "S", "Z", "J", "L")
PIECE_TO_ID = {piece: i for i, piece in enumerate(PIECES)}
EMPTY_PIECE_ID = 7

BOARD_HEIGHT = 40
BOARD_WIDTH = 10
BOARD_CELLS = BOARD_HEIGHT * BOARD_WIDTH
PACKED_BOARD_BYTES = BOARD_CELLS // 8
PREVIEW_DEPTH = 5

STATE_SIZE = BOARD_CELLS + 7 + 7 + PREVIEW_DEPTH * 7
CANDIDATE_SIZE = BOARD_CELLS + 7 + 4 + 2 + 1 + 1


def piece_id(piece: str | None) -> int:
    if piece in (None, "", "N"):
        return EMPTY_PIECE_ID
    try:
        return PIECE_TO_ID[str(piece)]
    except KeyError as exc:
        raise ValueError(f"Unknown canonical piece: {piece!r}") from exc


def pack_board(board: np.ndarray) -> np.ndarray:
    arr = (np.asarray(board).reshape(-1) != 0).astype(np.uint8)
    if arr.shape != (BOARD_CELLS,):
        raise ValueError(f"board must contain {BOARD_CELLS} cells")
    return np.packbits(arr, bitorder="little")


def unpack_boards(packed: np.ndarray) -> np.ndarray:
    arr = np.asarray(packed, dtype=np.uint8)
    if arr.shape[-1] != PACKED_BOARD_BYTES:
        raise ValueError(
            f"packed board last dim must be {PACKED_BOARD_BYTES}; got {arr.shape}"
        )
    return np.unpackbits(
        arr,
        axis=-1,
        count=BOARD_CELLS,
        bitorder="little",
    ).astype(np.float32, copy=False)


def _one_hot_piece_ids(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    out = np.zeros(ids.shape + (7,), dtype=np.float32)
    valid = (ids >= 0) & (ids < 7)
    if np.any(valid):
        flat_out = out.reshape(-1, 7)
        flat_ids = ids.reshape(-1)
        flat_valid = valid.reshape(-1)
        rows = np.nonzero(flat_valid)[0]
        flat_out[rows, flat_ids[rows]] = 1.0
    return out


def dense_state_batch(
    board_packed: np.ndarray,
    active_ids: np.ndarray,
    hold_ids: np.ndarray,
    preview_ids: np.ndarray,
) -> np.ndarray:
    board = unpack_boards(board_packed)
    active = _one_hot_piece_ids(active_ids)
    hold = _one_hot_piece_ids(hold_ids)
    preview = _one_hot_piece_ids(preview_ids).reshape(len(board), PREVIEW_DEPTH * 7)
    out = np.concatenate((board, active, hold, preview), axis=1).astype(np.float32)
    if out.shape[1] != STATE_SIZE:
        raise RuntimeError(f"state feature width mismatch: {out.shape}")
    return out


def dense_candidate_batch(
    board_packed: np.ndarray,
    piece_ids: np.ndarray,
    rotations: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    use_hold: np.ndarray,
    lines_cleared: np.ndarray,
) -> np.ndarray:
    board = unpack_boards(board_packed)
    piece = _one_hot_piece_ids(piece_ids)

    rotations = np.asarray(rotations, dtype=np.int64)
    rot = np.zeros((len(board), 4), dtype=np.float32)
    rot[np.arange(len(board)), rotations % 4] = 1.0

    x = np.asarray(xs, dtype=np.float32).reshape(-1, 1) / 10.0
    y = np.asarray(ys, dtype=np.float32).reshape(-1, 1) / 40.0
    hold = np.asarray(use_hold, dtype=np.float32).reshape(-1, 1)
    lines = np.asarray(lines_cleared, dtype=np.float32).reshape(-1, 1) / 4.0

    out = np.concatenate((board, piece, rot, x, y, hold, lines), axis=1).astype(np.float32)
    if out.shape[1] != CANDIDATE_SIZE:
        raise RuntimeError(f"candidate feature width mismatch: {out.shape}")
    return out


def torch_unpack_boards(packed: torch.Tensor) -> torch.Tensor:
    """Decode little-endian packed boards on the current device.

    This is the compact-GPU training path.  The cache stays packed on the CPU
    and only ~50 board bytes per state/candidate cross PCIe.
    """
    if packed.shape[-1] != PACKED_BOARD_BYTES:
        raise ValueError(
            f"packed board last dim must be {PACKED_BOARD_BYTES}; got {tuple(packed.shape)}"
        )
    packed_u8 = packed.to(dtype=torch.uint8)
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed_u8.unsqueeze(-1) >> shifts) & 1).reshape(
        *packed.shape[:-1], BOARD_CELLS
    )
    return bits.to(dtype=torch.float32)


def _torch_one_hot_piece_ids(ids: torch.Tensor) -> torch.Tensor:
    # EMPTY_PIECE_ID=7 becomes all-zero after slicing away class 7, matching
    # the original numpy encoding exactly.
    encoded = F.one_hot(ids.to(dtype=torch.long), num_classes=8)[..., :7]
    return encoded.to(dtype=torch.float32)


def torch_dense_state_batch(
    board_packed: torch.Tensor,
    active_ids: torch.Tensor,
    hold_ids: torch.Tensor,
    preview_ids: torch.Tensor,
) -> torch.Tensor:
    board = torch_unpack_boards(board_packed)
    active = _torch_one_hot_piece_ids(active_ids)
    hold = _torch_one_hot_piece_ids(hold_ids)
    preview = _torch_one_hot_piece_ids(preview_ids).reshape(
        board.shape[0], PREVIEW_DEPTH * 7
    )
    out = torch.cat((board, active, hold, preview), dim=1)
    if out.shape[1] != STATE_SIZE:
        raise RuntimeError(f"state feature width mismatch: {tuple(out.shape)}")
    return out


def torch_dense_candidate_batch(
    board_packed: torch.Tensor,
    piece_ids: torch.Tensor,
    rotations: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    use_hold: torch.Tensor,
    lines_cleared: torch.Tensor,
) -> torch.Tensor:
    board = torch_unpack_boards(board_packed)
    piece = _torch_one_hot_piece_ids(piece_ids)
    rot = F.one_hot(
        rotations.to(dtype=torch.long).remainder(4),
        num_classes=4,
    ).to(dtype=torch.float32)
    x = xs.to(dtype=torch.float32).reshape(-1, 1) / 10.0
    y = ys.to(dtype=torch.float32).reshape(-1, 1) / 40.0
    hold = use_hold.to(dtype=torch.float32).reshape(-1, 1)
    lines = lines_cleared.to(dtype=torch.float32).reshape(-1, 1) / 4.0
    out = torch.cat((board, piece, rot, x, y, hold, lines), dim=1)
    if out.shape[1] != CANDIDATE_SIZE:
        raise RuntimeError(f"candidate feature width mismatch: {tuple(out.shape)}")
    return out
