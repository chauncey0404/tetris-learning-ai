from __future__ import annotations
import argparse, hashlib, http.client, json, shutil, subprocess, time, urllib.error, urllib.parse, urllib.request, uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from tetrio.replay.garbage_scan import load_ttrm,replay_identity,validate_current_ttrm_object

CHANNEL_API="https://ch.tetr.io/api"
INOUE_API="https://inoue.szy.lol/api/replay"

@dataclass
class ReplayRecord:
    replayid:str
    owner_username:str
    opponent_usernames:list[str]
    ts:str|None
    stub:bool
    source_rank:int|None
    source_user_id:str|None

def parse_args():
    p=argparse.ArgumentParser(description="Harvest recent high-level TETRA LEAGUE replays via TETRA CHANNEL metadata + Inoue replay forwarding.")
    p.add_argument("--output-dir",type=Path,default=Path(r"data\tetrio\replays\top_players"))
    p.add_argument("--top-players",type=int,default=20)
    p.add_argument("--recent-per-player",type=int,default=10)
    p.add_argument("--max-replays",type=int,default=100)
    p.add_argument("--channel-delay",type=float,default=1.05)
    p.add_argument("--download-delay",type=float,default=1.25)
    p.add_argument("--timeout",type=float,default=30.0)
    p.add_argument("--retries",type=int,default=4)
    p.add_argument("--users",nargs="*",default=None)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument(
        "--skip-stubs",
        action="store_true",
        help=(
            "Skip TETRA CHANNEL records marked stub/pruned. By default the "
            "harvester still attempts them through Inoue because its replay "
            "cache can temporarily retain replays after TETR.IO pruning."
        ),
    )
    p.add_argument("--manifest",type=Path,default=Path(r"artifacts\tetrio\top_replay_harvest_manifest.json"))
    return p.parse_args()

def _http(url,*,headers,timeout,retries,expect_json):
    for attempt in range(retries+1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=timeout) as resp:
                body=resp.read()
                return (json.loads(body.decode("utf-8")) if expect_json else body)
        except urllib.error.HTTPError as exc:
            if exc.code==429 and attempt<retries:
                ra=exc.headers.get("Retry-After")
                try: delay=float(ra) if ra else min(60.0,2.0**(attempt+1))
                except ValueError: delay=min(60.0,2.0**(attempt+1))
                print(f"HTTP 429; backing off {delay:.1f}s")
                time.sleep(delay); continue
            if 500<=exc.code<600 and attempt<retries:
                delay=min(30.0,2.0**attempt); time.sleep(delay); continue
            raise
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            http.client.BadStatusLine,
            ConnectionResetError,
            BrokenPipeError,
            OSError,
        ) as exc:
            # Inoue/CDN responses can occasionally terminate a chunked body
            # before the final zero-length chunk. A partial .ttrm must never be
            # accepted as valid replay data, so discard the transport attempt
            # and retry the whole request from byte zero.
            if attempt>=retries:
                raise
            delay=min(30.0,2.0**attempt)
            print(
                f"Transport error ({type(exc).__name__}); "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")

def _api(path,*,sid,timeout,retries):
    obj=_http(f"{CHANNEL_API}/{path.lstrip('/')}",headers={
        "User-Agent":"tetris-learning-ai/garbage-parity-research",
        "Accept":"application/json","X-Session-ID":sid
    },timeout=timeout,retries=retries,expect_json=True)
    if not isinstance(obj,dict) or not obj.get("success"):
        raise RuntimeError(f"TETRA CHANNEL request failed: {obj}")
    data=obj.get("data")
    if not isinstance(data,dict): raise RuntimeError("Unexpected TETRA CHANNEL response")
    return data

def _leaders(n,*,sid,timeout,retries):
    q=urllib.parse.urlencode({"limit":min(100,n)})
    data=_api(f"users/by/league?{q}",sid=sid,timeout=timeout,retries=retries)
    entries=data.get("entries")
    return [x for x in entries if isinstance(x,dict)][:n] if isinstance(entries,list) else []

def _recent(username,limit,*,sid,timeout,retries):
    q=urllib.parse.urlencode({"limit":min(100,max(1,limit))})
    data=_api(f"users/{urllib.parse.quote(username.lower())}/records/league/recent?{q}",
              sid=sid,timeout=timeout,retries=retries)
    entries=data.get("entries")
    return [x for x in entries if isinstance(x,dict)] if isinstance(entries,list) else []

def _record(entry,rank,user_id,owner_username=None):
    # The general Record schema documents a `user` object, but the current
    # user-scoped recent-League endpoint can omit it because the owner is
    # already determined by /users/:user/records/league/recent.
    #
    # Therefore the endpoint username is the authoritative fallback owner.
    if isinstance(entry.get("record"), dict):
        entry = entry["record"]

    rid=entry.get("replayid")
    if not isinstance(rid,str) or not rid:
        return None

    user=entry.get("user")
    if isinstance(user,dict) and isinstance(user.get("username"),str):
        owner=user["username"]
    elif isinstance(owner_username,str) and owner_username:
        owner=owner_username
    else:
        return None

    opponents=[
        o["username"]
        for o in entry.get("otherusers",[])
        if isinstance(o,dict) and isinstance(o.get("username"),str)
    ]
    return ReplayRecord(
        rid.removeprefix("R:"),
        owner,
        opponents,
        entry.get("ts") if isinstance(entry.get("ts"),str) else None,
        bool(entry.get("stub",False)),
        rank,
        user_id,
    )

def _filename(r):
    clean=lambda s:"".join(c if c.isalnum() or c in "-_." else "_" for c in s.lower())[:40]
    opp=r.opponent_usernames[0] if r.opponent_usernames else "unknown"
    h=hashlib.sha1(r.replayid.encode()).hexdigest()[:10]
    return f"{clean(r.owner_username)}_vs_{clean(opp)}_{h}.ttrm"

def _download_with_curl(url, dest, *, timeout, retries):
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        return None, "curl not found"

    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.unlink(missing_ok=True)

    # Inoue's own documentation uses curl, and Inoue itself is libcurl-based.
    # Use curl's mature HTTP/chunked handling for replay bodies on Windows.
    cmd = [
        curl,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry", str(max(0, int(retries))),
        "--retry-all-errors",
        "--retry-delay", "2",
        "--connect-timeout", str(max(1, int(min(timeout, 30)))),
        "--max-time", str(max(10, int(timeout))),
        "--header", "Accept: application/octet-stream,application/json",
        "--header", "User-Agent: tetris-learning-ai/garbage-parity-research",
        "--output", str(tmp),
        url,
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        msg = (proc.stderr or proc.stdout or "").strip()
        if proc.returncode != 0:
            return False, f"curl exit {proc.returncode}: {msg}", None
        return False, "curl returned an empty replay body", None

    # Some Inoue/CDN responses terminate the HTTP chunked stream incorrectly
    # after the complete JSON payload has already been delivered. curl reports
    # this as exit 18 ("transfer closed with outstanding read data"), while the
    # saved body may still be a complete, valid .ttrm. Never trust transport
    # success alone: validate the replay payload itself.
    transport_error = None
    if proc.returncode != 0:
        transport_error = (
            f"curl exit {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )

    try:
        obj = load_ttrm(tmp)
    except Exception:
        tmp.unlink(missing_ok=True)
        if transport_error is not None:
            return False, transport_error, None
        return False, "curl body is not valid replay JSON", None

    ok, reason = validate_current_ttrm_object(obj)
    if not ok:
        tmp.unlink(missing_ok=True)
        if transport_error is not None:
            return False, f"{transport_error}; replay validation failed: {reason}", None
        return False, f"replay validation failed: {reason}", None

    # Payload integrity outranks a malformed trailing transfer terminator.
    status = (
        "VALID_BODY_DESPITE_TRANSPORT_ERROR"
        if transport_error is not None
        else "VALID_BODY"
    )
    return True, tmp, status


def _download(r,dest,*,timeout,retries):
    url = f"{INOUE_API}/{urllib.parse.quote(r.replayid)}"

    curl_result = _download_with_curl(
        url,
        dest,
        timeout=timeout,
        retries=retries,
    )
    curl_ok = curl_result[0]

    if curl_ok is True:
        tmp = curl_result[1]
        curl_status = curl_result[2]
    elif curl_ok is False:
        return False, str(curl_result[1]), None
    else:
        curl_status = None
        # Fallback only when curl is unavailable.
        try:
            body=_http(
                url,
                headers={
                    "User-Agent":"tetris-learning-ai/garbage-parity-research",
                    "Accept":"application/octet-stream,application/json",
                    "Connection":"close",
                },
                timeout=timeout,
                retries=retries,
                expect_json=False,
            )
        except urllib.error.HTTPError as exc:
            return False,f"HTTP {exc.code}",None
        except http.client.IncompleteRead as exc:
            # Last-resort salvage path if curl is unavailable. Keep the body
            # bytes, then rely on JSON + replay schema validation below.
            body = exc.partial
            curl_status = "URLLIB_INCOMPLETE_READ_BODY"
        except (
            urllib.error.URLError,
            TimeoutError,
            http.client.HTTPException,
            ConnectionError,
            OSError,
        ) as exc:
            return False,f"transport failure after retries: {type(exc).__name__}: {exc}",None

        tmp=dest.with_suffix(dest.suffix+".part")
        tmp.parent.mkdir(parents=True,exist_ok=True)
        tmp.write_bytes(body)

    try:
        obj=load_ttrm(tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return False,f"download is not valid replay JSON: {exc}",None

    ok,reason=validate_current_ttrm_object(obj)
    if not ok:
        tmp.unlink(missing_ok=True)
        return False,f"replay validation failed: {reason}",None

    tmp.replace(dest)
    identity = replay_identity(obj)
    if curl_status:
        identity = dict(identity)
        identity["download_transport_status"] = curl_status
    return True,"ok",identity

def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); a.manifest.parent.mkdir(parents=True,exist_ok=True)
    sid=f"tetris-learning-ai-{uuid.uuid4()}"
    print("="*104); print("TETR.IO TOP-PLAYER REPLAY HARVESTER"); print("="*104)
    print("Metadata: official TETRA CHANNEL API"); print("Replay bytes: Inoue public replay API")
    if a.users:
        leaders=[{"username":u,"_id":None,"_rank":None} for u in a.users]
    else:
        leaders=_leaders(a.top_players,sid=sid,timeout=a.timeout,retries=a.retries)
        for i,x in enumerate(leaders,1): x["_rank"]=i
    records={}
    user_report=[]
    for i,u in enumerate(leaders,1):
        name=u.get("username")
        if not isinstance(name,str): continue
        rank=u.get("_rank"); uid=u.get("_id") if isinstance(u.get("_id"),str) else None
        print(f"[{i}/{len(leaders)}] metadata {name}")
        try: entries=_recent(name,a.recent_per_player,sid=sid,timeout=a.timeout,retries=a.retries)
        except Exception as exc:
            print("  ERROR",exc); user_report.append({"username":name,"error":str(exc)}); time.sleep(a.channel_delay); continue
        added=stubs=live=parseable=0
        sample_keys = sorted(entries[0].keys()) if entries and isinstance(entries[0], dict) else []
        for e in entries:
            r=_record(
                e,
                rank if isinstance(rank,int) else None,
                uid,
                owner_username=name,
            )
            if r is None:
                continue
            parseable += 1
            if r.stub:
                stubs += 1
                if a.skip_stubs:
                    continue
            else:
                live += 1
            if r.replayid not in records:
                records[r.replayid]=r
                added+=1

        user_report.append({
            "username":name,
            "rank":rank,
            "records":len(entries),
            "parseable_records":parseable,
            "live_records":live,
            "stub_records":stubs,
            "new_unique_replays":added,
            "sample_entry_keys":sample_keys,
        })
        print(
            f"  records={len(entries)} parseable={parseable} "
            f"live={live} stub={stubs} new={added}"
        )
        if entries and parseable == 0:
            print(f"  WARNING: no parseable records; sample keys={sample_keys}")
        time.sleep(max(0,a.channel_delay))
    recs=list(records.values())[:a.max_replays]
    print(f"Unique replay IDs (live+stub candidates): {len(recs)}")
    downloads=[]; valid=existing=0
    for i,r in enumerate(recs,1):
        dest=a.output_dir/_filename(r); item={**asdict(r),"path":str(dest)}
        if dest.is_file():
            try: obj=load_ttrm(dest); ok,_=validate_current_ttrm_object(obj)
            except Exception: ok=False
            if ok:
                item.update({"status":"EXISTING_VALID","identity":replay_identity(obj)}); downloads.append(item)
                valid+=1; existing+=1; print(f"[{i}/{len(recs)}] existing {dest.name}"); continue
        if a.dry_run:
            item["status"]="DRY_RUN"; downloads.append(item); continue
        print(
            f"[{i}/{len(recs)}] download "
            f"{r.owner_username} vs {(r.opponent_usernames or ['?'])[0]}"
            f"{' [stub/pruned metadata]' if r.stub else ''}"
        )
        ok,reason,identity=_download(r,dest,timeout=a.timeout,retries=a.retries)
        item.update({"status":"DOWNLOADED" if ok else "FAILED","reason":reason,"identity":identity}); downloads.append(item); valid+=int(ok)
        if i<len(recs): time.sleep(max(0,a.download_delay))
    report={"format":"tetrio_top_replay_harvest_manifest","created_utc":datetime.now(timezone.utc).isoformat(),
            "metadata_source":"https://ch.tetr.io/api/","replay_source":"https://inoue.szy.lol/api/replay/{replayid}",
            "config":vars(a)|{"output_dir":str(a.output_dir),"manifest":str(a.manifest)},
            "leader_users":user_report,"unique_live_replay_ids":len(recs),"valid_local_replays":valid,
            "existing_valid":existing,"downloads":downloads}
    a.manifest.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    selected_live = sum(1 for r in recs if not r.stub)
    selected_stub = sum(1 for r in recs if r.stub)
    failed = sum(1 for item in downloads if item.get("status") == "FAILED")
    print(f"Selected live IDs  : {selected_live}")
    print(f"Selected stub IDs  : {selected_stub}")
    print(f"Valid local replays: {valid}")
    print(f"Failed downloads   : {failed}")
    print(f"Manifest           : {a.manifest}")

if __name__=="__main__": main()
