from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ai.observable_q_network import ObservableSafeQNetwork
from v8_7_scale_invariant_policy import (
    normalized_margin_actions_torch,
)
from v8_8_2_graph_safe_policy import (
    normalized_margin_actions_top4_graphsafe,
)
from v8_8_1_packed_replay import (
    V881PackedReplayBuffer,
)
from v8_8_2_cuda_graph_train_common import (
    CudaGraphDDQNLearner,
    make_capturable_adamw,
)


def max_optimizer_step(optimizer):
    values = []
    for state in optimizer.state.values():
        step = state.get("step")
        if torch.is_tensor(step):
            values.append(float(step.detach().cpu()))
        elif step is not None:
            values.append(float(step))
    return max(values) if values else 0.0


def model_snapshot(model):
    return {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }


def assert_model_exact(model, snap):
    for k, v in model.state_dict().items():
        if not torch.equal(v, snap[k]):
            diff = float(
                (v - snap[k]).abs().max().detach().cpu()
            )
            raise AssertionError(
                f"Graph capture changed model before counted training: "
                f"{k} max_abs={diff}"
            )


def policy_parity():
    for seed in (1, 7, 88, 882):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)

        q = torch.randn(
            (8192, 4),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        )
        mask = (
            torch.rand(
                (8192, 4),
                device="cuda",
                generator=gen,
            )
            > 0.15
        )
        mask[:, 0] = True
        q[:32].fill_(1.0)
        mask[32:64, 1:] = False

        old = normalized_margin_actions_torch(
            q,
            mask,
            0.600,
        )
        new = normalized_margin_actions_top4_graphsafe(
            q,
            mask,
            0.600,
        )

        for label, a, b in zip(
            ("action", "margin", "raw_gap", "span"),
            old,
            new,
        ):
            if not torch.equal(a, b):
                raise AssertionError(
                    f"Policy parity failed: {label} seed={seed}"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default=(
            "models/"
            "v8_8_1_longtraj_gpu_replay_td_200k.pt"
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )
    ap.add_argument(
        "--stamp",
        default=(
            "data/"
            "v8_8_2_cuda_graph_production_gate_pass.json"
        ),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the production graph gate."
        )

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)

    print("=" * 80)
    print("V8.8.2 PRODUCTION CUDA GRAPH INTEGRATION GATE")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", args.checkpoint)
    print("Batch:", args.batch_size)
    print()

    policy_parity()
    print("Graph-safe normalized policy parity: PASS")

    ckpt = torch.load(
        args.checkpoint,
        map_location="cuda",
        weights_only=False,
    )

    model = ObservableSafeQNetwork().cuda()
    model.load_state_dict(
        ckpt["model_state_dict"]
    )
    model.train()

    target = ObservableSafeQNetwork().cuda()
    target.load_state_dict(
        ckpt.get(
            "target_model_state_dict",
            ckpt["model_state_dict"],
        )
    )
    target.eval()

    optimizer, lr_tensor, resumed = make_capturable_adamw(
        model=model,
        lr=5e-5,
        weight_decay=1e-4,
        checkpoint_optimizer_state=ckpt.get(
            "optimizer_state_dict"
        ),
        resume=(
            "optimizer_state_dict"
            in ckpt
        ),
    )

    before_model = model_snapshot(model)
    before_step = max_optimizer_step(optimizer)

    replay = V881PackedReplayBuffer(
        capacity=max(
            20000,
            args.batch_size * 2,
        ),
        device=torch.device("cuda"),
        seed=882250,
    )

    with torch.no_grad():
        replay.data.normal_(0.0, 0.25)
        lo, hi = replay.layout.offsets["next_mask"]
        replay.data[:, lo:hi].fill_(1.0)
        lo, hi = replay.layout.offsets["done"]
        replay.data[:, lo:hi].bernoulli_(0.01)
        replay.size = replay.capacity
        replay.position = 0

    learner = CudaGraphDDQNLearner(
        model=model,
        target_model=target,
        optimizer=optimizer,
        replay=replay,
        batch_size=args.batch_size,
        gamma=0.99,
        target_gate=0.600,
        terminal_penalty=1.0,
    )

    torch.cuda.synchronize()

    assert_model_exact(
        model,
        before_model,
    )

    after_capture_step = max_optimizer_step(
        optimizer
    )
    if after_capture_step != before_step:
        raise AssertionError(
            "Graph warmup/capture changed resumed AdamW step: "
            f"before={before_step} after={after_capture_step}"
        )

    print("Non-destructive model capture: PASS")
    print("Non-destructive optimizer capture: PASS")
    print(
        "Optimizer resume:",
        "YES" if resumed else "NO",
        f"(step={before_step:g})",
    )

    metric = learner.step(
        collect_metrics=True
    )
    torch.cuda.synchronize()

    after_real_step = max_optimizer_step(
        optimizer
    )
    if after_real_step != before_step + 1.0:
        raise AssertionError(
            "Counted graph gradient did not increment AdamW step exactly once: "
            f"before={before_step} after={after_real_step}"
        )

    values = metric.detach().cpu()
    if not torch.isfinite(values).all():
        raise AssertionError(
            f"Non-finite graph metric: {values.tolist()}"
        )

    changed = False
    for k, v in model.state_dict().items():
        if not torch.equal(v, before_model[k]):
            changed = True
            break
    if not changed:
        raise AssertionError(
            "Counted CUDA Graph gradient did not update model parameters."
        )

    print("Counted CUDA Graph optimizer step: PASS")
    print("Finite learner metrics: PASS")
    print("Model parameter update: PASS")

    stamp = {
        "status": "PASS",
        "version": (
            "V8_8_2_PRODUCTION_CUDA_GRAPH_INTEGRATION_GATE"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "policy_parity": "BIT_EXACT_PASS",
        "capture_non_destructive": True,
        "optimizer_step_exact": True,
    }

    out = Path(args.stamp)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(
            stamp,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Parity/gate stamp:",
        out,
    )
    print(
        "V8.8.2 PRODUCTION CUDA GRAPH GATE: PASS"
    )


if __name__ == "__main__":
    main()
