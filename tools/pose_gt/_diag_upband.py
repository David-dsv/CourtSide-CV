"""THROWAWAY: does re-detecting an UPSCALED above-net band raise the detection
ceiling for the true far player (vs full-frame conf=0.08)? Reports how many of the
'currently undetected' GT frames are recovered by a 2x upper-band crop re-detect.

  python tools/pose_gt/_diag_upband.py [det_conf] [scale]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
DET_CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0


def detect(det, img, conf):
    r = det.person_model(img, conf=conf, imgsz=1280, classes=[0], device="cpu",
                         verbose=False)
    return (r[0].boxes.xyxy.cpu().numpy() if r[0].boxes is not None else [])


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

    full_only = both = upband_recovered = neither = 0
    for fidx, (gx, gy) in gtbf.items():
        if fidx >= len(frames):
            continue
        frame = frames[fidx]
        # full-frame detect
        fb = detect(det, frame, DET_CONF)
        full_hit = any(b[3] < net_y and
                       (((b[0]+b[2])/2 - gx)**2 + ((b[1]+b[3])/2 - gy)**2)**0.5 < 0.10*fw
                       for b in fb)
        # upper-band detect: crop [0 : net_y], upscale, detect, map back
        band = frame[0:int(net_y), :]
        up = cv2.resize(band, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        ub = detect(det, up, DET_CONF)
        up_hit = False
        for b in ub:
            cx, cy = (b[0]+b[2])/2/SCALE, (b[1]+b[3])/2/SCALE
            if ((cx - gx)**2 + (cy - gy)**2)**0.5 < 0.10*fw:
                up_hit = True
                break
        if full_hit and up_hit:
            both += 1
        elif full_hit:
            full_only += 1
        elif up_hit:
            upband_recovered += 1
        else:
            neither += 1

    tot = both + full_only + upband_recovered + neither
    print(f"det_conf={DET_CONF} scale={SCALE}  GT={tot}")
    print(f"  full-frame detects true far:        {both+full_only}/{tot} = "
          f"{100*(both+full_only)/tot:.1f}%")
    print(f"  upband ALSO recovers (new):         +{upband_recovered}  -> ceiling "
          f"{100*(both+full_only+upband_recovered)/tot:.1f}%")
    print(f"  neither (truly invisible):          {neither}/{tot} = "
          f"{100*neither/tot:.1f}%")


if __name__ == "__main__":
    main()
