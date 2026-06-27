"""Print the EXACT windowed-min wrist proximity the gate computes, for the hits
the harness path produces (with the new bounce_frames). Lets us pick a threshold
that keeps real shots and drops spurious swings."""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory
from vision.shots import (_wrist_speed_series, _side_of, _player_height, detect_handedness)

c = json.loads((ROOT/"tests/fixtures/cache/demo3_event.json").read_text())
fps, fw, fh = c["fps"], c["width"], c["height"]
kal = [tuple(x) if x else None for x in c["kalman_centers"]]
kal_real = c["kalman_is_real"]
ppf = c["players_per_frame"]
gt_h = [int(s["frame"]) for s in json.loads((ROOT/"tests/fixtures/shots/tennis_demo3.shots.json").read_text())["shots"]]
gt_b = [int(b["frame"]) for b in json.loads((ROOT/"tests/fixtures/bounces/tennis_demo3.bounces.json").read_text())["bounces"]]
net_y = fh*0.47
ball_win = int(fps*0.20)

sm = smooth_ball_trajectory(kal, max_gap=int(fps*0.4))
# replicate the candidate loop with NO prox gate to see all candidates + their prox
handed = detect_handedness(ppf, net_y)
min_gap = int(fps*0.30)
from vision.shots import _adaptive_threshold, _fwhm_at
amp_floor = _adaptive_threshold(None, fh, fps)
fwhm_min = max(3, int(fps*0.06))
cands=[]
for side in ("near","far"):
    speed, wrist_xy = _wrist_speed_series(ppf, side, handed[side], net_y, fps)
    i=1
    while i < len(sm)-1:
        if (not np.isnan(speed[i]) and speed[i]>=amp_floor and speed[i]>=speed[i-1]
                and speed[i]>speed[i+1] and _fwhm_at(speed,i)>=fwhm_min):
            cands.append((i,side,float(speed[i]),wrist_xy)); i+=min_gap
        else: i+=1
cands.sort(key=lambda x:x[0])
print(f"{'peak':>5} {'side':>5} {'wmin/h':>7}  truth")
for i,side,ws,wrist_xy in cands:
    lo,hi=max(0,i-ball_win),min(len(sm),i+ball_win+1)
    # me
    players=ppf[i] if i<len(ppf) else None
    if not players: continue
    me=None
    for p in players:
        if _side_of(p,net_y)==side and (me is None or _player_height(p)>_player_height(me)): me=p
    if me is None: continue
    h=max(_player_height(me),1.0)
    wmin=float("inf")
    for j in range(lo,hi):
        if sm[j] is None: continue
        wj=wrist_xy[j] if j<len(wrist_xy) else None
        if wj is None or wj is False: continue
        w=wj[1]
        dw=np.hypot(sm[j][0]-w[0], sm[j][1]-w[1])
        if dw<wmin: wmin=dw
    near_s=any(abs(i-g)<=8 for g in gt_h); near_b=any(abs(i-g)<=8 for g in gt_b)
    t = "REAL" if near_s else ("BNC" if near_b else "SPUR")
    wm = "%.2f"%(wmin/h) if wmin<float("inf") else "None"
    print(f"{i:>5} {side:>5} {wm:>7}  {t}")
