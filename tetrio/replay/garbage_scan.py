from __future__ import annotations
import json, math
from pathlib import Path
from typing import Any

def number(value: Any) -> float | None:
    if isinstance(value, bool): return None
    if isinstance(value, (int, float)):
        x=float(value)
        if math.isfinite(x): return x
    return None

def load_ttrm(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f: return json.load(f)

def current_rounds(obj: Any) -> list:
    if not isinstance(obj, dict): return []
    replay=obj.get("replay")
    if not isinstance(replay, dict): return []
    rounds=replay.get("rounds")
    return rounds if isinstance(rounds, list) else []

def usernames(obj: Any) -> list[str | None]:
    raw=obj.get("users") if isinstance(obj, dict) else None
    if not isinstance(raw, list): return []
    return [str(x.get("username")) if isinstance(x,dict) and x.get("username") is not None else None for x in raw]

def player_replay(player_obj: Any) -> dict[str, Any]:
    if not isinstance(player_obj, dict): return {}
    rep=player_obj.get("replay")
    return rep if isinstance(rep, dict) else {}

def garbage_stats(player_obj: Any) -> dict[str, Any]:
    rep=player_replay(player_obj)
    results=rep.get("results")
    if not isinstance(results, dict): return {}
    stats=results.get("stats")
    if not isinstance(stats, dict): return {}
    garbage=stats.get("garbage")
    return dict(garbage) if isinstance(garbage, dict) else {}

def inbound_garbage_packets(player_obj: Any) -> list[dict[str, Any]]:
    rep=player_replay(player_obj)
    events=rep.get("events")
    if not isinstance(events, list): return []
    out=[]
    for idx,event in enumerate(events):
        if not isinstance(event,dict) or event.get("type")!="ige": continue
        data=event.get("data")
        if not isinstance(data,dict) or data.get("type")!="interaction": continue
        inner=data.get("data")
        if not isinstance(inner,dict) or inner.get("type")!="garbage": continue
        amt=number(inner.get("amt"))
        if amt is None: continue
        out.append({
            "event_index":idx,"outer_frame":event.get("frame"),"envelope_frame":data.get("frame"),
            "packet_frame":inner.get("frame"),"amt":amt,"cid":inner.get("cid"),"iid":inner.get("iid"),
            "ackiid":inner.get("ackiid"),"gameid":inner.get("gameid"),"x":inner.get("x"),
            "y":inner.get("y"),"size":inner.get("size")
        })
    return out

def approx_equal(a: float | None,b: float | None,tol: float=1e-9)->bool|None:
    if a is None or b is None: return None
    return abs(a-b)<=tol

def validate_current_ttrm_object(obj: Any)->tuple[bool,str]:
    if not isinstance(obj,dict): return False,"root is not an object"
    rounds=current_rounds(obj)
    if not rounds: return False,"missing replay.rounds"
    streams=sum(1 for rr in rounds if isinstance(rr,list) for p in rr if player_replay(p))
    if streams<2: return False,"fewer than two player replay streams"
    return True,"ok"

def replay_identity(obj: Any)->dict[str,Any]:
    if not isinstance(obj,dict): return {}
    return {"id":obj.get("id"),"gamemode":obj.get("gamemode"),"ts":obj.get("ts"),
            "version":obj.get("version"),"users":usernames(obj)}
