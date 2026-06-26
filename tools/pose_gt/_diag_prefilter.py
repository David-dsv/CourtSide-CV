"""THROWAWAY: of the GT frames where the true far player is detected in the RAW
above-net set but FAILS our pre-filter (h<40 / aspect<1.1 / xr margin), which clause
killed it? Restores the lost detections by relaxing the right filter.

  python tools/pose_gt/_diag_prefilter.py [det_conf]
"""
import json
import sys
from collections import Counter
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

    raw_detected = passed = 0
    killed_by = Counter()
    far_box_dims = []
    for fidx, (gx, gy) in gtbf.items():
        if fidx >= len(frames):
            continue
        dres = det.person_model(frames[fidx], conf=DET_CONF, imgsz=1280,
                                classes=[0], device="cpu", verbose=False)
        pboxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        # find the RAW above-net detection closest to GT
        best = None
        for b in pboxes:
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if y2 >= net_y:
                continue
            d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            if d < 0.10 * fw and (best is None or d < best[0]):
                best = (d, x1, y1, x2, y2)
        if best is None:
            continue
        raw_detected += 1
        _, x1, y1, x2, y2 = best
        w, h = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        xr = cx / fw
        asp = h / max(w, 1.0)
        far_box_dims.append((w, h, asp, xr))
        reasons = []
        if xr < 0.05 or xr > 0.95:
            reasons.append("xr-margin")
        if h < 40:
            reasons.append("h<40")
        if asp < 1.1:
            reasons.append("aspect<1.1")
        if reasons:
            killed_by[",".join(reasons)] += 1
        else:
            passed += 1

    print(f"det_conf={DET_CONF}")
    print(f"true far in RAW above-net set: {raw_detected}/{len(gtbf)}")
    print(f"  passes current pre-filter:    {passed}/{raw_detected}")
    print(f"  killed by pre-filter:         {raw_detected - passed}/{raw_detected}")
    for k, v in killed_by.most_common():
        print(f"      {k}: {v}")
    dims = np.array(far_box_dims)
    if len(dims):
        print(f"\n  true-far box dims: w[{dims[:,0].min():.0f}-{dims[:,0].max():.0f}] "
              f"h[{dims[:,1].min():.0f}-{dims[:,1].max():.0f}] "
              f"aspect[{dims[:,2].min():.2f}-{dims[:,2].max():.2f}]")
        print(f"  h percentiles: 5%={np.percentile(dims[:,1],5):.0f} "
              f"10%={np.percentile(dims[:,1],10):.0f} 25%={np.percentile(dims[:,1],25):.0f}")
        print(f"  aspect percentiles: 5%={np.percentile(dims[:,2],5):.2f} "
              f"10%={np.percentile(dims[:,2],10):.2f} 25%={np.percentile(dims[:,2],25):.2f}")


if __name__ == "__main__":
    main()
