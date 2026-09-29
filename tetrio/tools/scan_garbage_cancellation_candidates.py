from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from typing import Any
from tetrio.replay.garbage_scan import approx_equal,current_rounds,garbage_stats,inbound_garbage_packets,load_ttrm,number,usernames,validate_current_ttrm_object

def parse_args():
    p=argparse.ArgumentParser(description="Scan multiplayer .ttrm files for low-complexity aggregate garbage cancellation candidates.")
    p.add_argument("replay_dir",type=Path,nargs="?",default=Path(r"data\tetrio\replays\top_players"))
    p.add_argument("--top",type=int,default=30)
    p.add_argument("--min-cancel",type=float,default=1.0)
    p.add_argument("--max-inbound-packets",type=int,default=8)
    p.add_argument("--output",type=Path,default=Path(r"artifacts\tetrio\garbage_cancellation_candidates.json"))
    p.add_argument("--csv",type=Path,default=Path(r"artifacts\tetrio\garbage_cancellation_candidates.csv"))
    return p.parse_args()

def _complexity(n,cancel,sent_exact,received_exact,max_packets):
    score=n*10+int(round(cancel*2))
    if not sent_exact: score+=1000
    if received_exact is False: score+=100
    if n>max_packets: score+=(n-max_packets)*20
    if sent_exact and received_exact is not False and n<=2 and cancel<=2: label="VERY_LOW"
    elif sent_exact and received_exact is not False and n<=4 and cancel<=4: label="LOW"
    elif sent_exact and n<=8: label="MEDIUM"
    else: label="HIGH"
    return score,label

def _files(root):
    if root.is_file(): return [root]
    return sorted(root.glob("**/*.ttrm"))

def _scan_file(path,*,min_cancel,max_inbound_packets):
    obj=load_ttrm(path); ok,reason=validate_current_ttrm_object(obj)
    if not ok: return [],{"path":str(path),"status":"INVALID_OR_UNSUPPORTED","reason":reason}
    names=usernames(obj); rows=[]; rounds2=streams=0
    for ri,rr in enumerate(current_rounds(obj)):
        if not isinstance(rr,list) or len(rr)!=2: continue
        rounds2+=1
        packs=[inbound_garbage_packets(rr[0]),inbound_garbage_packets(rr[1])]
        inbound=[sum(float(x["amt"]) for x in packs[0]),sum(float(x["amt"]) for x in packs[1])]
        stats=[garbage_stats(rr[0]),garbage_stats(rr[1])]
        for p in (0,1):
            streams+=1; other=1-p
            attack=number(stats[p].get("attack")); sent=number(stats[p].get("sent")); received=number(stats[p].get("received"))
            if attack is None or sent is None: continue
            cancel=attack-sent
            if cancel+1e-9<min_cancel or inbound[p]<=0: continue
            observed_out=inbound[other]
            sent_exact=bool(approx_equal(sent,observed_out))
            received_exact=approx_equal(received,inbound[p])
            score,label=_complexity(len(packs[p]),cancel,sent_exact,received_exact,max_inbound_packets)
            rows.append({
                "file":str(path),"replay_id":obj.get("id") if isinstance(obj,dict) else None,
                "replay_ts":obj.get("ts") if isinstance(obj,dict) else None,"round":ri,"player":p,
                "username":names[p] if p<len(names) else None,"opponent":names[other] if other<len(names) else None,
                "attack":attack,"sent":sent,"cancel_candidate":cancel,"received":received,
                "inbound_total":inbound[p],"inbound_packet_count":len(packs[p]),
                "inbound_packet_amounts":[float(x["amt"]) for x in packs[p]],
                "inbound_packet_outer_frames":[x.get("outer_frame") for x in packs[p] if isinstance(x.get("outer_frame"),int)],
                "observed_outbound_from_opponent_stream":observed_out,"sent_packet_exact":sent_exact,
                "received_inbound_exact":received_exact,"complexity_score":score,"complexity":label
            })
    return rows,{"path":str(path),"status":"PASS","rounds_2p":rounds2,"player_round_streams":streams,"candidate_count":len(rows)}

def main():
    a=parse_args(); files=_files(a.replay_dir)
    if not files: raise SystemExit(f"No .ttrm files found under: {a.replay_dir}")
    print("="*104); print("TETR.IO BATCH GARBAGE CANCELLATION CANDIDATE SCANNER"); print("="*104)
    print(f"Files: {len(files)}"); print("attack-sent remains a CANDIDATE cancellation amount, not frame-level ground truth.")
    allrows=[]; reports=[]
    for i,path in enumerate(files,1):
        try: rows,rep=_scan_file(path,min_cancel=a.min_cancel,max_inbound_packets=a.max_inbound_packets)
        except Exception as exc: rows=[]; rep={"path":str(path),"status":"ERROR","reason":str(exc)}
        allrows+=rows; reports.append(rep)
        if i%25==0 or i==len(files): print(f"Scanned {i}/{len(files)} | candidates={len(allrows)}")
    allrows.sort(key=lambda x:(not bool(x["sent_packet_exact"]),int(x["complexity_score"]),int(x["inbound_packet_count"]),float(x["cancel_candidate"]),str(x["file"])))
    clean=[x for x in allrows if x["sent_packet_exact"]]; top=clean[:a.top]
    a.output.parent.mkdir(parents=True,exist_ok=True); a.csv.parent.mkdir(parents=True,exist_ok=True)
    report={"format":"tetrio_garbage_cancellation_candidate_scan","replay_dir":str(a.replay_dir),
            "files_scanned":len(files),"file_reports":reports,
            "candidate_definition":{"minimum_attack_minus_sent":a.min_cancel,"requires_inbound_total_gt_zero":True,
              "preferred_packet_accounting":"stats.garbage.sent == opponent inbound interaction total",
              "warning":"attack-sent is aggregate cancellation candidate evidence, not exact placement/frame cancellation."},
            "candidate_count":len(allrows),"packet_exact_candidate_count":len(clean),"top_candidates":top,"all_candidates":allrows}
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    cols=["complexity","complexity_score","file","round","player","username","opponent","attack","sent","cancel_candidate","inbound_total","inbound_packet_count","received","sent_packet_exact","received_inbound_exact"]
    with a.csv.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in allrows: w.writerow({k:r.get(k) for k in cols})
    print(); print("BEST CANCELLATION CANDIDATES")
    for i,c in enumerate(top,1):
        print(f"#{i:02d} {c['complexity']:8s} {Path(c['file']).name} R{c['round']} P{c['player']} "
              f"{c['username'] or '?'} vs {c['opponent'] or '?'} | in={c['inbound_total']:.0f} "
              f"packets={c['inbound_packet_count']} attack={c['attack']:.0f} sent={c['sent']:.0f} cancel?={c['cancel_candidate']:.0f}")
        print(f"     packet amounts={c['inbound_packet_amounts']} frames={c['inbound_packet_outer_frames']}")
    print(f"JSON: {a.output}"); print(f"CSV : {a.csv}")

if __name__=="__main__": main()
