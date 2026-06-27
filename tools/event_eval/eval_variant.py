"""Eval a CANDIDATE vision/events.py variant WITHOUT touching the live file.
Usage: venv/bin/python tools/event_eval/eval_variant.py /path/to/candidate_events.py
Imports the candidate as the `vision.events` module, runs the demo3 methodo path,
prints confusion_H->B, confusion_B->H, bounce F1, hit F1, and events in 120..210."""
import sys, json, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

cand = sys.argv[1] if len(sys.argv) > 1 else str(ROOT/"vision"/"events.py")
spec = importlib.util.spec_from_file_location("vision.events", cand)
mod = importlib.util.module_from_spec(spec)
sys.modules["vision.events"] = mod
spec.loader.exec_module(mod)

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust
from vision.shots import detect_hits
from vision.events import classify_events, detect_turning_points, detect_sharp_turns
from tools.event_eval.event_eval import match_events, scores

CACHE = ROOT/"tests"/"fixtures"/"cache"/"demo3_event.json"
c = json.loads(CACHE.read_text())
fps, fw, fh = c["fps"], c["width"], c["height"]
kal=[tuple(x) if x else None for x in c["kalman_centers"]]; kal_real=c["kalman_is_real"]
wasb=[tuple(x) if x else None for x in c["wasb_centers"]]; wasb_real=c["wasb_is_real"]
ppf=c["players_per_frame"]
gt_b=[int(b["frame"]) for b in json.loads((ROOT/"tests"/"fixtures"/"bounces"/"tennis_demo3.bounces.json").read_text())["bounces"]]
gt_h=[int(s["frame"]) for s in json.loads((ROOT/"tests"/"fixtures"/"shots"/"tennis_demo3.shots.json").read_text())["shots"]]

sm = smooth_ball_trajectory(kal, max_gap=int(fps*0.4))
b_cands,_,_ = detect_bounces_robust(sm,[0]*len(sm),fps,fh,fw)
turns = sorted(set(detect_turning_points(sm,fps,fh)) | set(detect_sharp_turns(kal,kal_real,fps,fw,fh)))
hits = detect_hits(sm,ppf,fps,fh,fw,bounce_frames=[])
events,_ = classify_events(b_cands,hits,kal,kal_real,ppf,fps,fw,fh,turning_frames=turns,smoothed_centers=sm,wasb_centers=wasb,wasb_is_real=wasb_real)
res = match_events([{"frame":e["frame"],"label":e["label"]} for e in events],gt_b,gt_h,round(0.15*fps))
s = scores(res)
print(f"H->B={res['conf_HtoB']} B->H={res['conf_BtoH']} bounceF1={s['bounce']['F1']:.3f} hitF1={s['hit']['F1']:.3f}")
print("leaks H->B:", [(p['pred_frame'],p['gt_frame']) for p in res['pairs'] if p['gt_class']=='shot' and p['pred_label']=='BOUNCE'])
print("leaks B->H:", [(p['pred_frame'],p['gt_frame']) for p in res['pairs'] if p['gt_class']=='bounce' and p['pred_label']=='HIT'])
print("f174 matched:", [p['pred_frame'] for p in res['pairs'] if p['gt_class']=='bounce' and p['gt_frame']==174] or "MISSED")
print("events 120..210:", [(e['frame'],e['label']) for e in events if 120<=e['frame']<=210])
