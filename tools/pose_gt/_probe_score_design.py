"""THROWAWAY: offline scoring-design probe. For EVERY GT frame, detect persons at
low conf, restrict to above-net, and test several candidate-scoring schemes to see
which one ranks the TRUE far player (closest to GT point) #1 among above-net boxes.

Pure in-frame (stateless) schemes only here — this is what the official stateless
scorer can reward. Reports per-scheme: how often the #1-ranked above-net box is
within 10%W of the GT point.

  python tools/pose_gt/_probe_score_design.py [conf] [imgsz]
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

    # detect once per GT frame, cache above-net boxes
    cache = {}   # fidx -> list of (cx,cy,w,h,aspect,conf,feet_y)
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

    # scheme definitions: each takes a row -> score, higher = more "far player"
    def s_conf_h(r):  # current-ish: h * conf
        return r[3] * r[5]

    def s_central(r):  # prefer central-x above-net
        cx = r[0]
        return -abs(cx / fw - 0.5)        # closest to center wins

    def s_central_human(r):
        cx, cy, w, h, asp, conf, feet = r
        xr = cx / fw
        central = 1.0 - abs(xr - 0.5) * 2.0   # 1 at center, 0 at edges
        # human aspect ~1.0-2.2; penalize very thin slivers (asp>2.6 = structure)
        human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
        return central * 1.0 + human * 0.5 + conf * 0.3

    def s_central_band(r):
        # central-x AND in the player y-band (just above net, not high stands)
        cx, cy, w, h, asp, conf, feet = r
        xr = cx / fw
        central = 1.0 - abs(xr - 0.5) * 2.0
        # far player feet are in a band below the high stands. derive band from
        # net: far player feet between ~0.15H and net. spectators are higher.
        fr = feet / fh
        bandness = 1.0 if 0.15 <= fr <= 0.50 else 0.0
        return central * 1.0 + bandness * 1.0 + conf * 0.2

    schemes = {
        "h*conf (baseline-ish)": s_conf_h,
        "central-x": s_central,
        "central+human-aspect": s_central_human,
        "central+feet-band": s_central_band,
    }

    print(f"conf={CONF} imgsz={IMGSZ}  GT frames={len(gtbf)}  net_y={net_y:.0f}\n")
    for name, fn in schemes.items():
        correct = present = 0
        for fidx, (gx, gy) in gtbf.items():
            rows = cache.get(fidx, [])
            if not rows:
                continue
            present += 1
            best = max(rows, key=fn)
            d = ((best[0] - gx) ** 2 + (best[1] - gy) ** 2) ** 0.5
            if d < 0.10 * fw:
                correct += 1
        print(f"  {name:28s}: #1-correct {correct}/{present} = "
              f"{100*correct/max(1,present):.1f}%")

    # also: ceiling — is the true far player even IN the detected set?
    ceil = present2 = 0
    for fidx, (gx, gy) in gtbf.items():
        rows = cache.get(fidx, [])
        if not rows:
            continue
        present2 += 1
        dmin = min(((r[0] - gx) ** 2 + (r[1] - gy) ** 2) ** 0.5 for r in rows)
        if dmin < 0.10 * fw:
            ceil += 1
    print(f"\n  CEILING (true far present in detections at all): "
          f"{ceil}/{present2} = {100*ceil/max(1,present2):.1f}%")


if __name__ == "__main__":
    main()
