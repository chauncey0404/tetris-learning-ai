from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
from tetrio.network.checkpoint import load_expert_v1_2
from tetrio.tools.train_expert_v1_2 import evaluate_ablation


def parse_args():
    p=argparse.ArgumentParser(description="Evaluate TRUE vs ZERO battle-state ablation")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--future-cache",type=Path,required=True)
    p.add_argument("--state-cache",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",type=Path,default=Path(r"artifacts\tetrio\expert_v1_2_state_ablation.json"))
    return p.parse_args()


def main():
    a=parse_args(); device=torch.device(a.device)
    model,ckpt=load_expert_v1_2(a.checkpoint,device=device)
    ab=evaluate_ablation(model=model,future_dir=a.future_cache,state_dir=a.state_cache,device=device,batch_size=a.batch_size)
    report={"checkpoint":str(a.checkpoint),"epoch":ckpt.get("epoch"),"ablation":ab}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    t,z=ab["true"],ab["zero"]
    print("="*108); print("TETR.IO EXPERT V1.2A — TRUE/ZERO STATE ABLATION"); print("="*108)
    print(f"TRUE: top1={t['top1']:.4f} top3={t['top3']:.4f} branch={t['branch_acc']:.4f} avoid={t['avoidable_rate']:.4f} Tdestroy={t['t_destroyed_rate']:.4f} Tdefer={t['t_cashout_deferred_rate']:.4f} Q={ab['true_quality_cost']:.6f}")
    print(f"ZERO: top1={z['top1']:.4f} top3={z['top3']:.4f} branch={z['branch_acc']:.4f} avoid={z['avoidable_rate']:.4f} Tdestroy={z['t_destroyed_rate']:.4f} Tdefer={z['t_cashout_deferred_rate']:.4f} Q={ab['zero_quality_cost']:.6f}")
    print(f"ΔTop1 TRUE-ZERO : {ab['top1_delta_true_minus_zero']:+.6f}")
    print(f"ΔQ TRUE-ZERO    : {ab['quality_delta_true_minus_zero']:+.6f} (negative is better)")
    print(f"Decision change : {ab['decision_change_rate']:.4f}")
    print(f"ZERO max residual: {ab['max_zero_residual']}")
    print(f"Report: {a.output}")

if __name__=="__main__": main()
