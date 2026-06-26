"""THROWAWAY v2: refine the stateless far-candidate scoring + add a MOTION scheme
(static spectator rejection across consecutive GT frames). Caches detections per
GT frame, then evaluates schemes including a temporal one.

  python tools/pose_gt/_probe_score_v2.py [conf] [imgsz]
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

CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 1280


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

    def central_human(r):
        cx, cy, w, h, asp, conf, feet = r
        xr = cx / fw
        central = 1.0 - abs(xr - 0.5) * 2.0
        human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
        return central + 0.5 * human + 0.3 * conf

    def central_human_band(r):
        cx, cy, w, h, asp, conf, feet = r
        xr = cx / fw
        central = 1.0 - abs(xr - 0.5) * 2.0
        human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
        fr = feet / fh
        bandness = 1.0 if 0.18 <= fr <= 0.50 else 0.0
        return central + 0.5 * human + 0.5 * bandness + 0.2 * conf

    def central_human_size(r):
        # add a mild size preference: the far PLAYER is bigger than the thin
        # spectator slivers but not huge. area-normalized.
        cx, cy, w, h, asp, conf, feet = r
        xr = cx / fw
        central = 1.0 - abs(xr - 0.5) * 2.0
        human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
        area = (w * h) / (fw * fh)
        sizep = min(1.0, area / 0.004)   # saturate; far player ~0.003-0.006
        return central + 0.5 * human + 0.3 * sizep + 0.2 * conf

    schemes = {
        "central+human": central_human,
        "central+human+band": central_human_band,
        "central+human+size": central_human_size,
    }
    print(f"conf={CONF} imgsz={IMGSZ}  GT frames={len(gtbf)}\n")
    for name, fn in schemes.items():
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
        print(f"  {name:24s}: {correct}/{present} = {100*correct/max(1,present):.1f}%"
              f"  mean_d={np.mean(dists):.0f}px")

    # MOTION scheme: a candidate that is STATIC across nearby frames is a
    # spectator. Build per-frame static penalty by matching each box to boxes in
    # the previous cached GT frame (consecutive GT frames are 1-5 video frames
    # apart at 50fps; static => near-zero displacement).
    fids = sorted(cache.keys())
    # precompute displacement: for each box, min dist to any box in prev frame
    def motion_central_human(rows, prev_rows):
        out = []
        for r in rows:
            cx, cy = r[0], r[1]
            if prev_rows:
                dmin = min(((cx - p[0]) ** 2 + (cy - p[1]) ** 2) ** 0.5
                           for p in prev_rows)
            else:
                dmin = 0.0
            # static if moved < 0.5% of W between GT samples
            static = dmin < 0.005 * fw
            xr = cx / fw
            central = 1.0 - abs(xr - 0.5) * 2.0
            asp = r[4]
            human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
            score = central + 0.5 * human + 0.3 * r[5]
            if static:
                score -= 1.5     # heavy static penalty
            out.append((score, r))
        return max(out, key=lambda e: e[0])[1]

    correct = present = 0
    dists = []
    for k, fidx in enumerate(fids):
        rows = cache.get(fidx, [])
        if not rows:
            continue
        prev_rows = cache.get(fids[k - 1], []) if k > 0 else []
        gx, gy = gtbf[fidx]
        present += 1
        best = motion_central_human(rows, prev_rows)
        d = ((best[0] - gx) ** 2 + (best[1] - gy) ** 2) ** 0.5
        dists.append(d)
        if d < 0.10 * fw:
            correct += 1
    print(f"  {'motion+central+human':24s}: {correct}/{present} = "
          f"{100*correct/max(1,present):.1f}%  mean_d={np.mean(dists):.0f}px")


if __name__ == "__main__":
    main()
