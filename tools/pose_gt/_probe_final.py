"""THROWAWAY: lock the FINAL stateless scoring design with NET-RELATIVE thresholds
(zero-hardcoding: everything derived from net_y / frame, no demo3 pixels).

Far-player score for an above-net candidate:
  central   = 1 - |cx/W - 0.5| * 2          (camera frames the court -> court central)
  human     = aspect penalty for thin slivers (asp > HUMAN_ASP)
  band      = feet in the LOWER part of the above-net region (closer to net than top)
  conf      = detector confidence (mild)
weights w_c, w_h, w_b, w_q swept.

  python tools/pose_gt/_probe_final.py [conf] [imgsz]
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
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.06
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
HUMAN_ASP = 2.2


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
        dres = det.person_model(frames[fidx], conf=CONF, imgsz=IMGSZ,
                                classes=[0], device="cpu", verbose=False)
        boxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        confs = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        rows = []
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if y2 >= net_y:
                continue
            w, h = x2 - x1, y2 - y1
            rows.append((cx, cy, w, h, h / max(w, 1.0), float(c), float(y2)))
        cache[fidx] = rows

    def make_score(wc, wh, wb, wq):
        def fn(r):
            cx, cy, w, h, asp, conf, feet = r
            central = 1.0 - abs(cx / fw - 0.5) * 2.0
            human = 1.0 if asp <= HUMAN_ASP else max(0.0, 1.0 - (asp - HUMAN_ASP) * 0.5)
            # NET-RELATIVE band: feet in the lower half of the above-net region
            # (between net_y/2 and net_y). spectators are higher (feet < net_y/2).
            band = 1.0 if feet >= 0.5 * net_y else 0.0
            return wc * central + wh * human + wb * band + wq * conf
        return fn

    def eval_fn(fn):
        correct = present = 0
        dists = []
        for fidx, (gx, gy) in gtbf.items():
            rows = cache.get(fidx, [])
            if not rows:
                continue
            present += 1
            best = max(rows, key=fn)
            d = ((best[0] - gx) ** 2 + (best[1] - gy) ** 2) ** 0.5
            dists.append(d)
            if d < 0.10 * fw:
                correct += 1
        return correct, present, float(np.mean(dists))

    print(f"conf={CONF} imgsz={IMGSZ}  net_y={net_y:.0f}  GT frames={len(gtbf)}\n")
    print("  w_central w_human w_band w_conf :  correct%   mean_d")
    grid = [
        (1.0, 0.5, 0.5, 0.2),
        (1.0, 0.5, 1.0, 0.2),
        (1.0, 0.3, 0.5, 0.1),
        (1.0, 0.0, 0.5, 0.2),   # no human term
        (1.0, 0.5, 0.0, 0.2),   # no band term
        (0.7, 0.5, 0.5, 0.2),   # softer centrality
        (1.0, 0.5, 0.5, 0.0),   # no conf term
        (1.2, 0.5, 0.7, 0.2),
    ]
    best = None
    for wc, wh, wb, wq in grid:
        c, p, md = eval_fn(make_score(wc, wh, wb, wq))
        pct = 100 * c / max(1, p)
        flag = ""
        if best is None or pct > best[0]:
            best = (pct, (wc, wh, wb, wq), md); flag = "  <-- best"
        print(f"   {wc:.1f}      {wh:.1f}     {wb:.1f}    {wq:.1f}    :  "
              f"{c}/{p}={pct:.1f}%   {md:.0f}px{flag}")
    print(f"\nBEST: {best[1]}  {best[0]:.1f}%  mean_d={best[2]:.0f}px")


if __name__ == "__main__":
    main()
