from __future__ import annotations

import pathlib
from pathlib import Path

import torch

from tetrio.network.model import TetrioExpertV0Network
from tetrio.network.model_v1 import TetrioExpertV1Network
from tetrio.network.model_v1_1 import TetrioExpertV11Network
from tetrio.network.model_v1_2 import TetrioExpertV12Network


def _safe_path_globals():
    return [
        pathlib.Path,
        pathlib.PurePath,
        pathlib.PurePosixPath,
        pathlib.PureWindowsPath,
        pathlib.PosixPath,
        pathlib.WindowsPath,
    ]


def safe_load_checkpoint(path: str | Path, *, map_location="cpu") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with torch.serialization.safe_globals(_safe_path_globals()):
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )


def load_expert_v0(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[TetrioExpertV0Network, dict]:
    ckpt = safe_load_checkpoint(path, map_location="cpu")
    if ckpt.get("format") != "tetrio_expert_v0":
        raise RuntimeError(
            f"Expected tetrio_expert_v0 checkpoint, got {ckpt.get('format')!r}"
        )
    model = TetrioExpertV0Network().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, ckpt


def load_expert_v1(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[TetrioExpertV1Network, dict]:
    ckpt = safe_load_checkpoint(path, map_location="cpu")
    if ckpt.get("format") != "tetrio_expert_v1":
        raise RuntimeError(
            f"Expected tetrio_expert_v1 checkpoint, got {ckpt.get('format')!r}"
        )
    model = TetrioExpertV1Network().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, ckpt



def load_expert_v1_1(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[TetrioExpertV11Network, dict]:
    ckpt = safe_load_checkpoint(path, map_location="cpu")
    if ckpt.get("format") != "tetrio_expert_v1_1":
        raise RuntimeError(
            f"Expected tetrio_expert_v1_1 checkpoint, got {ckpt.get('format')!r}"
        )
    config = ckpt.get("config", {})
    model = TetrioExpertV11Network(
        reranker_hidden_size=int(config.get("reranker_hidden_size", 96)),
        max_adjustment=float(config.get("max_adjustment", 2.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.freeze_scorer()
    model.eval()
    return model, ckpt



def load_expert_v1_2(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[TetrioExpertV12Network, dict]:
    ckpt = safe_load_checkpoint(path, map_location="cpu")
    if ckpt.get("format") != "tetrio_expert_v1_2_stateful":
        raise RuntimeError(
            "Expected tetrio_expert_v1_2_stateful checkpoint, got "
            f"{ckpt.get('format')!r}"
        )
    config = ckpt.get("config", {})
    base_cfg = ckpt.get("base_v1_1_config", {})
    model = TetrioExpertV12Network(
        v11_reranker_hidden_size=int(base_cfg.get("reranker_hidden_size", 96)),
        v11_max_adjustment=float(base_cfg.get("max_adjustment", 2.0)),
        state_hidden_size=int(config.get("state_hidden_size", 64)),
        state_max_adjustment=float(config.get("state_max_adjustment", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.freeze_base()
    model.eval()
    return model, ckpt
