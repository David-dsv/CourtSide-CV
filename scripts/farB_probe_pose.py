"""Probe: can the pose model resolve the TINY far crop with different settings?
If yes → we get correct box AND a real skeleton. If no → the far box must be
selected geometrically (skeleton-independent), which is approach B's premise."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"


def crop_pose(pm, frame, box, pad_frac, imgsz, conf):
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    pw = int((x2-x1)*pad_frac); ph = int((y2-y1)*pad_frac)
    cx1, cy1 = max(0, x1-pw), max(0, y1-ph)
    cx2, cy2 = min(fw, x2+pw), min(fh, y2+ph)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return 0
    res = pm(crop, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
    if res[0].keypoints is None or res[0].boxes is None or len(res[0].boxes) == 0:
        return 0
    kcf = res[0].keypoints.conf
    if kcf is None:
        return 0
    kcf = kcf.cpu().numpy()
    areas = ((res[0].boxes.xyxy[:, 2]-res[0].boxes.xyxy[:, 0]) *
             (res[0].boxes.xyxy[:, 3]-res[0].boxes.xyxy[:, 1])).cpu().numpy()
    bi = int(np.argmax(areas))
    return int((kcf[bi] > 0.30).sum())


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP)); fw, fh = rdr.width, rdr.height
    frames = list(rdr.iter_frames()); rdr.release()
    det = TennisYOLODetector(device="cpu"); pm = YOLO("yolov8m-pose.pt")
    sample = [0, 100, 150, 200, 332, 471, 570]
    configs = [
        ("pad.25 imgsz960 c.15 (current)", 0.25, 960, 0.15),
        ("pad.25 imgsz320 c.05", 0.25, 320, 0.05),
        ("pad.50 imgsz640 c.05", 0.50, 640, 0.05),
        ("pad1.0 imgsz640 c.05", 1.0, 640, 0.05),
        ("pad1.0 imgsz960 c.01", 1.0, 960, 0.01),
    ]
    for fidx in sample:
        gx, gy = gtbf[fidx]; frame = frames[fidx]
        dres = det.person_model(frame, conf=0.12, imgsz=1280, classes=[0],
                                device="cpu", verbose=False)
        pboxes = dres[0].boxes.xyxy.cpu().numpy()
        cxs = (pboxes[:, 0]+pboxes[:, 2])/2; cys = (pboxes[:, 1]+pboxes[:, 3])/2
        bi = int(np.argmin(np.hypot(cxs-gx, cys-gy)))
        b = pboxes[bi]
        outs = [f"f{fidx}(h={b[3]-b[1]:.0f})"]
        for name, pad, imgsz, conf in configs:
            outs.append(f"{crop_pose(pm, frame, b, pad, imgsz, conf):2d}")
        print("  ".join(outs), flush=True)
    print("cols:", " | ".join(c[0] for c in configs))


if __name__ == "__main__":
    main()
