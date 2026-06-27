"""Sweep wrist_prox_max on the legacy harness; report bounce/hit F1 + confusion."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory
from vision.shots import detect_hits
from tools.event_eval.event_eval import match_events, scores

c = json.loads((ROOT/"tests/fixtures/cache/demo3_event.json").read_text())
fps, fw, fh = c["fps"], c["width"], c["height"]
kal=[tuple(x) if x else None for x in c["kalman_centers"]]; kal_real=c["kalman_is_real"]; ppf=c["players_per_frame"]
gt_b=[int(b["frame"]) for b in json.loads((ROOT/"tests/fixtures/bounces/tennis_demo3.bounces.json").read_text())["bounces"]]
gt_h=[int(s["frame"]) for s in json.loads((ROOT/"tests/fixtures/shots/tennis_demo3.shots.json").read_text())["shots"]]
tol=round(0.15*fps)
sm=smooth_ball_trajectory(kal,max_gap=int(fps*0.4))
b=detect_bounces_from_trajectory(sm,[0.0]*len(sm),fps,fh,fw,raw_centers=kal,is_real=kal_real)
bf=[x[0] for x in b]
print(f"{'prox':>6} {'bF1':>5} {'hF1':>5} {'H>B':>4} {'B>H':>4} {'hTP':>4} {'hFP':>4} {'hFN':>4}")
for pm in [1.0,1.2,1.5,1.8,2.0,2.5,3.0,float('inf')]:
    hits=detect_hits(sm,ppf,fps,fh,fw,bounce_frames=bf,wrist_prox_max=pm)
    evs=[{"frame":int(x[0]),"label":"BOUNCE"} for x in b]+[{"frame":int(h["frame"]),"label":"HIT"} for h in hits]
    r=match_events(evs,gt_b,gt_h,tol); s=scores(r)
    print(f"{pm:>6} {s['bounce']['F1']:>5.2f} {s['hit']['F1']:>5.2f} {r['conf_HtoB']:>4} {r['conf_BtoH']:>4} {s['hit']['tp']:>4} {s['hit']['fp']:>4} {s['hit']['fn']:>4}")
