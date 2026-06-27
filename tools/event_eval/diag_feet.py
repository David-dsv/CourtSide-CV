"""For each GT bounce and GT hit, measure (over the nearest player) the ball's
distance to the WRIST vs to the FEET (box bottom) vs to the box CENTER, normalized
by box height. Hypothesis: GT HITS have ball-near-WRIST; GT BOUNCES that are near a
player have ball-near-FEET but FAR-from-wrist. If true, a 'ball_at_feet' signal can
let f174 escape the proximity firewall safely."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa
from vision.bounce import smooth_ball_trajectory  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"
L_WRIST, R_WRIST = 9, 10


def feats(ball, p):
    box = p["box"]
    h = max(box[3] - box[1], 1.0)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    feet = ((box[0] + box[2]) / 2, box[3])
    dcenter = float(np.hypot(ball[0] - cx, ball[1] - cy)) / h
    dfeet = float(np.hypot(ball[0] - feet[0], ball[1] - feet[1])) / h
    dwrist = float("inf")
    kx, kc = p.get("kps_xy"), p.get("kps_conf")
    if kx is not None and kc is not None:
        kx, kc = np.asarray(kx), np.asarray(kc)
        for wi in (L_WRIST, R_WRIST):
            if wi < len(kc) and kc[wi] >= 0.3:
                dwrist = min(dwrist, float(np.hypot(ball[0] - kx[wi][0], ball[1] - kx[wi][1])) / h)
    return dwrist, dfeet, dcenter


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))

    def best_player(f, ball):
        best, bd = None, float("inf")
        for p in (ppf[f] or []):
            if p is None or len(p.get("box", [])) < 4:
                continue
            dw, dfe, dc = feats(ball, p)
            d = min(dw, dc)
            if d < bd:
                bd, best = d, (dw, dfe, dc)
        return best

    print(f"{'evt':<8} {'frame':>5} {'wrist':>6} {'feet':>6} {'center':>6}  verdict")
    for f in sorted(gt_h):
        ball = sm[f]
        if ball is None:
            print(f"{'HIT':<8} {f:>5}  no ball")
            continue
        bp = best_player(f, ball)
        if bp is None:
            print(f"{'HIT':<8} {f:>5}  no player")
            continue
        dw, dfe, dc = bp
        v = "wrist<feet" if dw < dfe else "feet<wrist"
        print(f"{'HIT':<8} {f:>5} {dw:>6.2f} {dfe:>6.2f} {dc:>6.2f}  {v}")
    print()
    for f in sorted(gt_b):
        ball = sm[f]
        if ball is None:
            print(f"{'BOUNCE':<8} {f:>5}  no ball")
            continue
        bp = best_player(f, ball)
        if bp is None:
            print(f"{'BOUNCE':<8} {f:>5}  no player")
            continue
        dw, dfe, dc = bp
        v = "wrist<feet" if dw < dfe else "feet<wrist"
        print(f"{'BOUNCE':<8} {f:>5} {dw:>6.2f} {dfe:>6.2f} {dc:>6.2f}  {v}")


if __name__ == "__main__":
    main()
