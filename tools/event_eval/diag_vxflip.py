"""For each wrist-peak hit candidate, measure the ball's vx-flip and turn-angle on
the RAW track in a contact window. Hypothesis: a real HIT inverts vx (player sends
ball back) / sharply turns the ball; a phantom swing does not move the ball."""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory
from vision.shots import (_wrist_speed_series, _side_of, _player_height, detect_handedness,
                          _adaptive_threshold, _fwhm_at)

c = json.loads((ROOT/"tests/fixtures/cache/demo3_event.json").read_text())
fps, fw, fh = c["fps"], c["width"], c["height"]
kal = [tuple(x) if x else None for x in c["kalman_centers"]]
kal_real = c["kalman_is_real"]
ppf = c["players_per_frame"]
gt_h = [int(s["frame"]) for s in json.loads((ROOT/"tests/fixtures/shots/tennis_demo3.shots.json").read_text())["shots"]]
gt_b = [int(b["frame"]) for b in json.loads((ROOT/"tests/fixtures/bounces/tennis_demo3.bounces.json").read_text())["bounces"]]
net_y = fh*0.47; ball_win=int(fps*0.20)
sm = smooth_ball_trajectory(kal, max_gap=int(fps*0.4))

def vx_flip_and_turn(f, half):
    """Read raw track ±half around f. Return (vx_flip, turn_deg, netdx_ok)."""
    n=len(kal)
    pre=[kal[k] for k in range(max(0,f-half),f+1) if 0<=k<n and kal_real[k] and kal[k]]
    post=[kal[k] for k in range(f,min(n,f+half+1)) if 0<=k<n and kal_real[k] and kal[k]]
    if len(pre)<2 or len(post)<2: return None,None,False
    def slope(pts):
        ks=np.arange(len(pts)); xs=np.array([p[0] for p in pts])
        A=np.vstack([ks,np.ones(len(ks))]).T
        return float(np.linalg.lstsq(A,xs,rcond=None)[0][0])
    vxp=slope(pre); vxq=slope(post)
    netdx=abs(pre[-1][0]-pre[0][0])>fw*0.004 and abs(post[-1][0]-post[0][0])>fw*0.004
    flip = (np.sign(vxp)!=np.sign(vxq)) if netdx else None
    # turn angle
    k=half
    turn=None
    if f-k>=0 and f+k<n and kal[f-k] and kal[f] and kal[f+k]:
        a=(kal[f][0]-kal[f-k][0],kal[f][1]-kal[f-k][1])
        b=(kal[f+k][0]-kal[f][0],kal[f+k][1]-kal[f][1])
        na=(a[0]**2+a[1]**2)**.5; nb=(b[0]**2+b[1]**2)**.5
        if na>1 and nb>1: turn=np.degrees(np.arccos(max(-1,min(1,(a[0]*b[0]+a[1]*b[1])/(na*nb)))))
    return flip,turn,netdx

handed=detect_handedness(ppf,net_y)
amp_floor=_adaptive_threshold(None,fh,fps); fwhm_min=max(3,int(fps*0.06)); min_gap=int(fps*0.30)
cands=[]
for side in ("near","far"):
    speed,wxy=_wrist_speed_series(ppf,side,handed[side],net_y,fps)
    i=1
    while i<len(sm)-1:
        if (not np.isnan(speed[i]) and speed[i]>=amp_floor and speed[i]>=speed[i-1]
                and speed[i]>speed[i+1] and _fwhm_at(speed,i)>=fwhm_min):
            cands.append(i); i+=min_gap
        else: i+=1
cands=sorted(set(cands))
half=max(2,round(fps*0.08))
print(f"{'peak':>5} {'vxflip':>7} {'turn':>5}  truth")
for i in cands:
    # use the contact frame near i (closest ball to a near-player wrist within window) ~ i
    flip,turn,nd=vx_flip_and_turn(i,half)
    near_s=any(abs(i-g)<=8 for g in gt_h); near_b=any(abs(i-g)<=8 for g in gt_b)
    t="REAL" if near_s else ("BNC" if near_b else "SPUR")
    fl = {True:'FLIP',False:'keep',None:'?'}[flip]
    tn = "%.0f"%turn if turn is not None else "?"
    print(f"{i:>5} {fl:>7} {tn:>5}  {t}")
