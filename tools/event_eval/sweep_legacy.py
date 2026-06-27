"""Anti-overfit robustness sweep for the LEGACY separation: vary the turn-angle
threshold (bounce recall) AND the wrist proximity (hit precision) and confirm the
confusion stays 0/0 and the F1 gains hold across a BAND (not a knife-edge tuned to
one value). This is the plateau proof the CR requires."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vision.bounce import (smooth_ball_trajectory, detect_bounces_from_trajectory,
                           _velocity_turn_candidates)
import vision.bounce as B
from vision.shots import detect_hits
from tools.event_eval.event_eval import match_events, scores

c = json.loads((ROOT/"tests/fixtures/cache/demo3_event.json").read_text())
fps, fw, fh = c["fps"], c["width"], c["height"]
kal=[tuple(x) if x else None for x in c["kalman_centers"]]; kal_real=c["kalman_is_real"]
ppf=c["players_per_frame"]
gt_b=[int(b["frame"]) for b in json.loads((ROOT/"tests/fixtures/bounces/tennis_demo3.bounces.json").read_text())["bounces"]]
gt_h=[int(s["frame"]) for s in json.loads((ROOT/"tests/fixtures/shots/tennis_demo3.shots.json").read_text())["shots"]]
tol=round(0.15*fps)
sm=smooth_ball_trajectory(kal,max_gap=int(fps*0.4))

import functools
orig = B._velocity_turn_candidates
def run(angle, prox):
    # monkeypatch the angle into the turn generator via partial
    B._velocity_turn_candidates = functools.partial(orig, angle_deg=angle)
    b=detect_bounces_from_trajectory(sm,[0.0]*len(sm),fps,fh,fw,raw_centers=kal,is_real=kal_real)
    B._velocity_turn_candidates = orig
    bf=[x[0] for x in b]
    hits=detect_hits(sm,ppf,fps,fh,fw,bounce_frames=bf,wrist_prox_max=prox)
    evs=[{"frame":int(x[0]),"label":"BOUNCE"} for x in b]+[{"frame":int(h["frame"]),"label":"HIT"} for h in hits]
    r=match_events(evs,gt_b,gt_h,tol); s=scores(r)
    return s['bounce']['F1'], s['hit']['F1'], r['conf_HtoB'], r['conf_BtoH'], s['hit']['fp']

print("ANGLE sweep (prox=1.2):")
print(f"{'angle':>6} {'bF1':>5} {'hF1':>5} {'H>B':>4} {'B>H':>4} {'hFP':>4}")
for a in [60,65,70,75,80]:
    bF1,hF1,hb,bh,hfp=run(a,1.2)
    print(f"{a:>6} {bF1:>5.2f} {hF1:>5.2f} {hb:>4} {bh:>4} {hfp:>4}")
print("\nPROX sweep (angle=70):")
print(f"{'prox':>6} {'bF1':>5} {'hF1':>5} {'H>B':>4} {'B>H':>4} {'hFP':>4}")
for p in [0.8,1.0,1.2,1.3]:
    bF1,hF1,hb,bh,hfp=run(70,p)
    print(f"{p:>6} {bF1:>5.2f} {hF1:>5.2f} {hb:>4} {bh:>4} {hfp:>4}")
