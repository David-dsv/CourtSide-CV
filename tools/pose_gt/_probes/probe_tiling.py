"""Probe: can an UPSCALED top-region crop recover the 24 detection-loss far frames?
Compares, on the GT frames, full-frame YOLO vs YOLO on an upscaled crop of the
upper court band. Reports per-frame whether a far candidate appears within 10%W of GT.
Read-only experiment; writes nothing to prod.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import cv2
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector

GT = ROOT / "tests/fixtures/pose_gt/tennis_demo3.pose_gt.json"
CLIP = ROOT / "data/output/tennis_demo3.mp4"

def far_cands_fullframe(det, frame, conf, imgsz, net_y):
    r = det.person_model(frame, conf=conf, imgsz=imgsz, classes=[0], device="cpu", verbose=False)
    out = []
    if r[0].boxes is not None and len(r[0].boxes):
        for b, c in zip(r[0].boxes.xyxy.cpu().numpy(), r[0].boxes.conf.cpu().numpy()):
            cy = (b[1] + b[3]) / 2
            if cy < net_y:
                out.append((float((b[0]+b[2])/2), float((b[1]+b[3])/2), float(c), float(b[3]-b[1])))
    return out

def far_cands_topcrop(det, frame, conf, imgsz, net_y, top_frac=0.55, scale=2.0):
    """Upscale a crop of the TOP region (where the far player lives) and detect there."""
    fh, fw = frame.shape[:2]
    cy_cut = int(net_y * top_frac * 2)  # crop down to ~net (top_frac of upper-half... keep generous)
    cy_cut = int(net_y)                  # crop the whole above-net region
    crop = frame[0:cy_cut, :]
    up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    r = det.person_model(up, conf=conf, imgsz=imgsz, classes=[0], device="cpu", verbose=False)
    out = []
    if r[0].boxes is not None and len(r[0].boxes):
        for b, c in zip(r[0].boxes.xyxy.cpu().numpy(), r[0].boxes.conf.cpu().numpy()):
            # map back: divide by scale, crop has no x/y offset (starts at 0,0)
            x1, y1, x2, y2 = b / scale
            cyb = (y1 + y2) / 2
            if cyb < net_y:
                out.append((float((x1+x2)/2), float((y1+y2)/2), float(c), float(y2-y1)))
    return out

def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    det = TennisYOLODetector(device="cpu")
    rdr = VideoReader(str(CLIP)); fw, fh = rdr.width, rdr.height
    frames = list(rdr.iter_frames()); rdr.release()
    net_y = 0.5 * fh
    det_loss = [25,30,50,55,59,60,61,62,63,64,65,70,225,310,311,445,450,505,527,528,530,580,635,640]
    rec_full = rec_crop = 0
    for fi in det_loss:
        gx, gy = gtbf[fi]
        frame = frames[fi]
        # baseline at the prod settings + a hotter low-conf full-frame at imgsz1920
        ff = far_cands_fullframe(det, frame, 0.05, 1920, net_y)
        tc = far_cands_topcrop(det, frame, 0.08, 1280, net_y, scale=2.0)
        df = min([((c[0]-gx)**2+(c[1]-gy)**2)**0.5 for c in ff], default=9e9)
        dc = min([((c[0]-gx)**2+(c[1]-gy)**2)**0.5 for c in tc], default=9e9)
        okf = df < 0.10*fw; okc = dc < 0.10*fw
        rec_full += okf; rec_crop += okc
        print(f"f{fi}: GT=({gx:.0f},{gy:.0f}) | full1920c05 {'OK' if okf else 'MISS'}({df:.0f}px,{len(ff)}c) | topcrop2x {'OK' if okc else 'MISS'}({dc:.0f}px,{len(tc)}c)", flush=True)
    print(f"\nRECOVERED of 24 det-loss frames: full-frame@1920/0.05 = {rec_full}/24 | top-crop 2x = {rec_crop}/24", flush=True)

if __name__ == "__main__":
    main()
