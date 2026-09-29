from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from tetrio.replay.garbage_scan import inbound_garbage_packets,validate_current_ttrm_object
from tetrio.tools.scan_garbage_cancellation_candidates import _scan_file

def player(attack,sent,received,inbound_packets):
    events=[]
    for i,amt in enumerate(inbound_packets,1):
        payload={"type":"garbage","amt":amt,"gameid":1,"frame":i*100,"cid":i,"iid":i,"ackiid":0,"x":4,"y":38,"size":1}
        events += [
            {"type":"ige","frame":i*100+10,"data":{"frame":i*100+5,"type":"interaction","data":payload}},
            {"type":"ige","frame":i*100+15,"data":{"frame":i*100+5,"type":"interaction_confirm","data":dict(payload)}},
        ]
    return {"replay":{"frames":1000,"events":events,"results":{"stats":{"garbage":{"attack":attack,"sent":sent,"received":received,"cleared":0}}}}}

class Tests(unittest.TestCase):
    def test_schema(self):
        obj={"replay":{"rounds":[[player(3,2,4,[4]),player(4,4,2,[2])]]}}
        ok,reason=validate_current_ttrm_object(obj); self.assertTrue(ok,reason)
    def test_confirm_not_double_counted(self):
        self.assertEqual(len(inbound_garbage_packets(player(3,2,4,[4]))),1)
    def test_candidate(self):
        obj={"id":"abc","ts":"2026-09-22T00:00:00Z","users":[{"username":"a"},{"username":"b"}],
             "replay":{"rounds":[[player(3,2,4,[4]),player(4,4,2,[2])]]}}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"x.ttrm"; p.write_text(json.dumps(obj),encoding="utf-8")
            rows,rep=_scan_file(p,min_cancel=1,max_inbound_packets=8)
        self.assertEqual(rep["status"],"PASS"); self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["cancel_candidate"],1); self.assertTrue(rows[0]["sent_packet_exact"])
if __name__=="__main__": unittest.main()
