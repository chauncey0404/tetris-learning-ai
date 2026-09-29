from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tetrio.network.checkpoint import load_expert_v1_1
from tetrio.network.model_v1_2 import TetrioExpertV12Network
from tetrio.tools.train_expert_v1_2 import evaluate_ablation


def parse_args():
    p=argparse.ArgumentParser(description="Verify zero-init V1.2A is exactly V1.1")
    p.add_argument("--base",type=Path,default=Path(r"models\tetrio_expert_v1_1_future_500k.pt"))
    p.add_argument("--future-cache",type=Path,required=True)
    p.add_argument("--state-cache",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--state-hidden-size",type=int,default=64)
    p.add_argument("--state-max-adjustment",type=float,default=1.0)
    p.add_argument("--device",default="cuda")
    return p.parse_args()


def main():
    a=parse_args(); device=torch.device(a.device)
    base,ckpt=load_expert_v1_1(a.base,device=device); cfg=ckpt.get("config",{})
    model=TetrioExpertV12Network(
        v11_reranker_hidden_size=int(cfg.get("reranker_hidden_size",96)),
        v11_max_adjustment=float(cfg.get("max_adjustment",2.0)),
        state_hidden_size=a.state_hidden_size,
        state_max_adjustment=a.state_max_adjustment,
    ).to(device)
    model.base.load_state_dict(base.state_dict(),strict=True); model.freeze_base(); del base
    ab=evaluate_ablation(model=model,future_dir=a.future_cache,state_dir=a.state_cache,device=device,batch_size=a.batch_size)
    passed=(ab["max_true_residual"]==0.0 and ab["max_zero_residual"]==0.0 and ab["max_zero_score_delta_vs_v11"]==0.0 and ab["decision_change_rate"]==0.0 and ab["true"]==ab["zero"])
    print("="*100); print("TETR.IO EXPERT V1.2A — E00 PARITY") ; print("="*100)
    print(f"Rows              : {ab['rows']:,}")
    print(f"Max TRUE residual : {ab['max_true_residual']}")
    print(f"Max ZERO residual : {ab['max_zero_residual']}")
    print(f"Max ZERO Δscore   : {ab['max_zero_score_delta_vs_v11']}")
    print(f"Decision change   : {ab['decision_change_rate']}")
    print(f"Result            : {'PASS' if passed else 'FAIL'}")
    if not passed: raise SystemExit(2)

if __name__=="__main__": main()
