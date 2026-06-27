"""Why didn't detect_far_wrist_hits catch f333? Dump the far wrist (x,y), its
box-normalized speed, the ball-at-wrist distance, around f333 (and f525 for
contrast — that one WAS caught)."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa
from vision.bounce import smooth_ball_trajectory  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
L_WRIST, R_WRIST = 9, 10


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    ppf = c["players_per_frame"]
    net = fh * 0.47
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))

    def far_wrist(f):
        best, sc = None, -1
        for p in (ppf[f] or []):
            if not p or len(p.get("box", [])) < 4 or p["box"][3] >= net:
                continue
            kc = p.get("kps_conf")
            if kc is None:
                continue
            s = sum((kc[i] if i < len(kc) else 0) for i in (L_WRIST, R_WRIST))
            if s > sc:
                sc, best = s, p
        if best is None:
            return None, None, None
        kx, kc = np.asarray(best["kps_xy"]), np.asarray(best["kps_conf"])
        w, wc = None, 0.30
        for wi in (L_WRIST, R_WRIST):
            if wi < len(kc) and kc[wi] >= wc:
                w, wc = (float(kx[wi][0]), float(kx[wi][1])), kc[wi]
        box = best["box"]
        return w, max(box[3] - box[1], 1.0), box

    for label, c0 in [("f333", 333), ("f525", 525)]:
        print(f"\n=== {label} ===")
        print(f"  f : far_wrist(x,y)  box_h  |dwrist/frame|/h  ball  ball->wrist/h")
        prev = None
        for ff in range(c0 - 10, c0 + 11):
            w, h, box = far_wrist(ff)
            ball = sm[ff]
            spd = "—"
            if w is not None and prev is not None and h:
                spd = f"{np.hypot(w[0]-prev[0], w[1]-prev[1])/h:.3f}"
            dball = "—"
            if w is not None and ball is not None and h:
                dball = f"{np.hypot(ball[0]-w[0], ball[1]-w[1])/h:.2f}"
            wstr = f"({w[0]:.0f},{w[1]:.0f})" if w else "none"
            bstr = f"({ball[0]:.0f},{ball[1]:.0f})" if ball else "none"
            print(f"  {ff}: {wstr:<12} {h if h else 0:>5.0f}  {spd:>8}  {bstr:<12} {dball}")
            prev = w


if __name__ == "__main__":
    main()
