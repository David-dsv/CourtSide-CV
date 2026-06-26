"""THROWAWAY: for each GT frame, find the true far player among above-net
detections and report its RANK in _far_score order (1 = best). Tells us how many
pool slots per side we actually need, and whether _far_score puts the player top-k.

  python tools/pose_gt/_diag_rank.py [det_conf]
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.video_utils import VideoReader
from vision.pose import _far_score
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

    ranks = Counter()
    not_detected = 0
    for fidx, (gx, gy) in gtbf.items():
        if fidx >= len(frames):
            continue
        dres = det.person_model(frames[fidx], conf=DET_CONF, imgsz=1280,
                                classes=[0], device="cpu", verbose=False)
        pboxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        pconfs = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        far = []
        true_idx = None
        for i, b in enumerate(pboxes):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2
            xr = cx / fw
            if xr < 0.05 or xr > 0.95 or h < 40:
                continue
            asp = h / max(w, 1.0)
            if asp < 1.1:
                continue
            if y2 >= net_y:
                continue
            s = _far_score(xr, asp, float(pconfs[i]))
            d = ((cx - gx) ** 2 + ((y1 + y2) / 2 - gy) ** 2) ** 0.5
            far.append((s, len(far), d))
            if d < 0.10 * fw and (true_idx is None or d < far[true_idx][2]):
                true_idx = len(far) - 1
        if true_idx is None:
            not_detected += 1
            continue
        order = sorted(range(len(far)), key=lambda k: -far[k][0])
        rk = order.index(true_idx) + 1
        ranks[rk] += 1

    total = sum(ranks.values()) + not_detected
    print(f"det_conf={DET_CONF}  GT frames with true far in detections: "
          f"{sum(ranks.values())}/{total} (not detected: {not_detected})")
    cum = 0
    for rk in sorted(ranks):
        cum += ranks[rk]
        print(f"  true far at far-score rank {rk}: {ranks[rk]:3d}  "
              f"(cum top-{rk}: {cum}/{total} = {100*cum/total:.1f}%)")


if __name__ == "__main__":
    main()
