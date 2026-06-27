"""Deeper hit diagnostic: min ball<->wrist distance over the contact search window,
plus ball direction-change near the wrist. The goal: find a discriminator that
keeps real shots (incl. f396 prox=2.45-at-contact) and kills spurious wrist peaks.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory
from vision.shots import (detect_hits, _wrist_of, _side_of, _player_height,
                          detect_handedness)

CACHE = ROOT / "tests/fixtures/cache/demo3_event.json"
GT_S = ROOT / "tests/fixtures/shots/tennis_demo3.shots.json"
GT_B = ROOT / "tests/fixtures/bounces/tennis_demo3.bounces.json"

c = json.loads(CACHE.read_text())
fps, fw, fh = c["fps"], c["width"], c["height"]
kal = [tuple(x) if x else None for x in c["kalman_centers"]]
kal_real = c["kalman_is_real"]
ppf = c["players_per_frame"]
gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
net_y = fh * 0.47

sm = smooth_ball_trajectory(kal, max_gap=int(fps*0.4))
b_cands = detect_bounces_from_trajectory(sm, [0.0]*len(sm), fps, fh, fw,
                                         raw_centers=kal, is_real=kal_real)
bf = [b[0] for b in b_cands]
hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=bf)
handed = detect_handedness(ppf, net_y)
ball_win = int(fps*0.20)

print(f"bounce preds: {bf}")
print(f"{'frame':>6} {'side':>5} {'wspd':>5} {'minProx':>8} {'@frame':>7} {'dirChg°':>7}  tag")
for h in hits:
    f = h["frame"]; side = h["player_side"]
    # min ball<->wrist over the window
    lo, hi = max(0, f-ball_win), min(len(sm), f+ball_win+1)
    best_d, best_j = float("inf"), None
    for j in range(lo, hi):
        if sm[j] is None: continue
        players = ppf[j] if j < len(ppf) else None
        if not players: continue
        me = None
        for p in players:
            if _side_of(p, net_y) == side:
                if me is None or _player_height(p) > _player_height(me): me = p
        if me is None: continue
        w = _wrist_of(me, handed[side])
        if w is None: continue
        hh = max(_player_height(me), 1.0)
        d = float(np.hypot(sm[j][0]-w[0], sm[j][1]-w[1]))/hh
        if d < best_d: best_d, best_j = d, j
    # ball direction change at best_j (turn angle on smoothed track)
    dchg = None
    if best_j is not None:
        k = max(2, round(fps*0.06))
        if best_j-k >= 0 and best_j+k < len(sm) and sm[best_j-k] and sm[best_j] and sm[best_j+k]:
            a = (sm[best_j][0]-sm[best_j-k][0], sm[best_j][1]-sm[best_j-k][1])
            b = (sm[best_j+k][0]-sm[best_j][0], sm[best_j+k][1]-sm[best_j][1])
            na=(a[0]**2+a[1]**2)**.5; nb=(b[0]**2+b[1]**2)**.5
            if na>1 and nb>1:
                dchg=np.degrees(np.arccos(max(-1,min(1,(a[0]*b[0]+a[1]*b[1])/(na*nb)))))
    near_shot = any(abs(f-g)<=8 for g in gt_h)
    near_bnc = any(abs(f-g)<=8 for g in gt_b)
    tag = "REAL-SHOT" if near_shot else ("BOUNCE!" if near_bnc else "SPURIOUS")
    bd = "%.2f"%best_d if best_d<float("inf") else "None"
    dc = "%.0f"%dchg if dchg is not None else "None"
    print(f"{f:>6} {side:>5} {h['wrist_speed']:>5.0f} {bd:>8} {best_j if best_j else -1:>7} {dc:>7}  {tag}")
