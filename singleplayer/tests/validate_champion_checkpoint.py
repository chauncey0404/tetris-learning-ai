from __future__ import annotations

import argparse
from pathlib import Path

import torch

from singleplayer.network.q_network import ObservableSafeQNetwork

DEFAULT_CHECKPOINT = Path("models/v8_8_6_affinity_sharedweight_cuda_graph_td_31200k.pt")


def main() -> None:
    p = argparse.ArgumentParser(description="Load the formal V8.8.6 Champion after architecture migration.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise SystemExit("Unsupported checkpoint: missing model_state_dict")

    model = ObservableSafeQNetwork()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.inference_mode():
        state = torch.zeros((1, 243), dtype=torch.float32)
        candidates = torch.zeros((1, 4, 215), dtype=torch.float32)
        scalars = torch.zeros((1, 4), dtype=torch.float32)
        q = model(
            state=state,
            candidates=candidates,
            rewards=scalars,
            teacher_scores=scalars,
            teacher_ranks=scalars,
        )
    if tuple(q.shape) != (1, 4) or not torch.isfinite(q).all():
        raise SystemExit(f"Checkpoint forward smoke failed: shape={tuple(q.shape)} q={q}")

    env_steps = int(checkpoint.get("env_steps", -1))
    print("CHAMPION CHECKPOINT LOAD: PASS")
    print("checkpoint:", args.checkpoint)
    print("env_steps :", env_steps)
    print("q_shape   :", tuple(q.shape))


if __name__ == "__main__":
    main()
