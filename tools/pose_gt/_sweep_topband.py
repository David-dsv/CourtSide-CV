"""THROWAWAY: offline (no-pose) sweep of the top-of-frame penalty knee, ranking
ALL above-net candidates with the production pre-filter + _far_score variants, to
pick the most robust 'court-band' vertical term WITHOUT overfitting. Reports
#1-correct for each knee fraction of net_y and each penalty shape (cliff vs ramp).

  python tools/pose_gt/_sweep_topband.py [det_conf]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
DET_CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    frames = list(rdr.iter_frames())
    rdr.release()

    cache = {}
    for fidx in gtbf:
        if fidx >= len(frames):
            continue
        dres = det.person_model(frames[fidx], conf=DET_CONF, imgsz=1280,
                                classes=[0], device="cpu", verbose=False)
        pboxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        pconfs = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        rows = []
        for b, c in zip(pboxes, pconfs):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            xr = cx / fw
            if xr < 0.05 or xr > 0.95:
                continue
            if y2 >= net_y:
                continue
            if h < 0.012 * fh:
                continue
            rows.append((cx, cy, h / max(w, 1.0), float(c)))
        cache[fidx] = rows

    def score(cx, cy, asp, conf, knee, shape, pen):
        central = 1.0 - abs(cx / fw - 0.5) * 2.0
        human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
        s = central + 0.5 * human + 0.2 * conf
        ky = knee * net_y
        if shape == "cliff":
            if cy < ky:
                s -= pen
        else:  # ramp: full penalty at cy=0, zero at cy>=ky
            if cy < ky:
                s -= pen * (1.0 - cy / ky)
        return s

    def evalc(knee, shape, pen):
        ok = n = 0
        for fidx, (gx, gy) in gtbf.items():
            rows = cache.get(fidx, [])
            if not rows:
                continue
            n += 1
            best = max(rows, key=lambda r: score(r[0], r[1], r[2], r[3], knee, shape, pen))
            d = ((best[0] - gx) ** 2 + (best[1] - gy) ** 2) ** 0.5
            if d < 0.10 * fw:
                ok += 1
        return ok, n

    print(f"det_conf={DET_CONF}  net_y={net_y:.0f}\n")
    print("  knee*net_y  shape   pen :  #1-correct%")
    for shape in ("cliff", "ramp"):
        for knee in (0.0, 0.08, 0.15, 0.22, 0.28, 0.33):
            for pen in (1.0,):
                ok, n = evalc(knee, shape, pen)
                tag = "  (no top term)" if knee == 0 else ""
                print(f"   {knee:.2f}      {shape:5s}  {pen:.1f} :  "
                      f"{ok}/{n}={100*ok/n:.1f}%{tag}")
        print()


if __name__ == "__main__":
    main()
