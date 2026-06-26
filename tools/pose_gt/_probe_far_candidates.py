"""THROWAWAY probe: for a sample of GT frames, dump EVERY above-net person box
the detector returns at a low conf, annotated with distance to the human GT far
point. Answers: at what conf does the true far player appear, and can we tell it
apart from the static spectator WITHIN a single frame (size/aspect/conf/position)?

  python tools/pose_gt/_probe_far_candidates.py [conf] [imgsz]
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

CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
SAMPLE = [0, 50, 100, 150, 200, 300, 400, 500, 600]  # spread across the clip


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    frames = list(rdr.iter_frames())
    rdr.release()
    print(f"frame {fw}x{fh}  net_y={net_y:.0f}  conf={CONF} imgsz={IMGSZ}\n")

    for fidx in SAMPLE:
        if fidx not in gtbf or fidx >= len(frames):
            continue
        gx, gy = gtbf[fidx]["center"]
        dres = det.person_model(frames[fidx], conf=CONF, imgsz=IMGSZ,
                                classes=[0], device="cpu", verbose=False)
        boxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        confs = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        print(f"=== frame {fidx}  GT far=({gx:.0f},{gy:.0f}) ===")
        rows = []
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            feet = y2
            if feet >= net_y:        # near side, skip
                continue
            w, h = x2 - x1, y2 - y1
            aspect = h / max(w, 1.0)
            d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            rows.append((d, cx, cy, w, h, aspect, c))
        rows.sort()
        for d, cx, cy, w, h, aspect, c in rows:
            tag = "  <<< TRUE FAR" if d < 0.10 * fw else ""
            print(f"  d={d:6.0f}  c=({cx:6.0f},{cy:5.0f})  wxh={w:4.0f}x{h:4.0f}  "
                  f"asp={aspect:4.1f}  conf={c:.2f}{tag}")
        print()


if __name__ == "__main__":
    main()
