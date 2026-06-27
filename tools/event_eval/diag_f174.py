"""f174 deep probe: is the ball at the far player's FEET (bounce) or WRIST (hit)?
Measure ball->wrist vs ball->feet(box bottom) distance, normalized by box height,
for the far player across f168..f182. A bounce-at-feet should have ball FAR from
wrist but NEAR the feet; a hit has ball NEAR the wrist."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa
from vision.bounce import smooth_ball_trajectory  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    ppf = c["players_per_frame"]
    net_y = fh * 0.47
    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))

    print("f : ball_xy | far box | ball->wrist/h  ball->feet/h  ball->center/h  ball->ankle/h")
    for ff in range(166, 184):
        b = all_centers[ff]
        pl = ppf[ff] or []
        far = None
        for p in pl:
            if p and len(p.get("box", [])) >= 4 and p["box"][3] < net_y:
                far = p
                break
        if b is None or far is None:
            print(f"  {ff}: ball={b} far={'none' if far is None else 'yes'}")
            continue
        box = far["box"]
        h = max(box[3] - box[1], 1.0)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        feet = ((box[0] + box[2]) / 2, box[3])
        kx = far.get("kps_xy")
        kc = far.get("kps_conf")
        dwrist = float("inf")
        dankle = float("inf")
        if kx is not None and kc is not None:
            kx = np.asarray(kx)
            kc = np.asarray(kc)
            for wi in (L_WRIST, R_WRIST):
                if wi < len(kc) and kc[wi] >= 0.3:
                    dwrist = min(dwrist, float(np.hypot(b[0] - kx[wi][0], b[1] - kx[wi][1])) / h)
            for ai in (L_ANKLE, R_ANKLE):
                if ai < len(kc) and kc[ai] >= 0.3:
                    dankle = min(dankle, float(np.hypot(b[0] - kx[ai][0], b[1] - kx[ai][1])) / h)
        dfeet = float(np.hypot(b[0] - feet[0], b[1] - feet[1])) / h
        dcenter = float(np.hypot(b[0] - cx, b[1] - cy)) / h
        print(f"  {ff}: ball=({b[0]:.0f},{b[1]:.0f}) box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) "
              f"wrist={dwrist:5.2f}  feet={dfeet:5.2f}  center={dcenter:5.2f}  ankle={dankle:5.2f}")


if __name__ == "__main__":
    main()
