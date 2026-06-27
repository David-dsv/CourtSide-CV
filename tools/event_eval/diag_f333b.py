"""f333 deeper: which player is near the ball, near vs far wrist distance, and what
detect_hits does. The ball at f333 is in the near-right court (y~760); is this a
NEAR hit detect_hits missed, or a far hit with the ball mislocated?"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa
from vision.bounce import smooth_ball_trajectory  # noqa
from vision.shots import detect_hits, _wrist_speed_series, detect_handedness  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
L_WRIST, R_WRIST = 9, 10


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    ppf = c["players_per_frame"]
    net = fh * 0.47
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))

    print("ball + BOTH players near f333:")
    for ff in range(326, 342):
        ball = sm[ff]
        line = f"  f{ff}: ball={tuple(int(v) for v in ball) if ball else None}"
        for p in (ppf[ff] or []):
            if not p or len(p.get("box", [])) < 4:
                continue
            box = p["box"]
            side = "near" if box[3] >= net else "far"
            h = max(box[3] - box[1], 1.0)
            kx, kc = p.get("kps_xy"), p.get("kps_conf")
            dw = float("inf")
            if kx is not None and kc is not None:
                kx, kc = np.asarray(kx), np.asarray(kc)
                for wi in (L_WRIST, R_WRIST):
                    if wi < len(kc) and kc[wi] >= 0.3 and ball is not None:
                        dw = min(dw, np.hypot(ball[0]-kx[wi][0], ball[1]-kx[wi][1])/h)
            line += f" | {side} box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) dwrist={dw:.2f}"
        print(line)

    # NEAR wrist speed around f333
    net_y = fh * 0.47
    handed = detect_handedness(ppf, net_y)
    print(f"\nNEAR wrist-speed (px/frame), handed near={handed['near']}:")
    speed_near, _ = _wrist_speed_series(ppf, "near", handed["near"], net_y, fps)
    amp_floor = 0.008 * fh
    print(f"  amp_floor (near, =0.008*fh) = {amp_floor:.1f}px")
    for ff in range(326, 342):
        print(f"  f{ff}: {speed_near[ff]:.2f}")

    hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[])
    print(f"\nhit cands near f333: {[h for h in hits if abs(h['frame']-333)<=10]}")
    print(f"all hit frames: {sorted(h['frame'] for h in hits)}")


if __name__ == "__main__":
    main()
