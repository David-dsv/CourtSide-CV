"""Measure REAL far-player detection coverage against the human pose GT.
The honest number that replaces the misleading 23%-vs-78%: how often does the
pose path actually find a far player where the human says there is one?

  python tools/pose_gt/measure_far_coverage.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.video_utils import VideoReader
from vision.pose import estimate_players_pose
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    print(f"GT: {len(gtbf)} frames with a verified far-player point", flush=True)

    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    frames = list(rdr.iter_frames())
    rdr.release()

    hit = matched = checked = 0
    dists = []
    for fidx, gtp in gtbf.items():
        if fidx >= len(frames):
            continue
        checked += 1
        players = estimate_players_pose(det.person_model, pm, frames[fidx], "cpu",
                                        court_zones=None) or []
        far = []
        for p in players:
            b = p.get("box")
            if b is None:
                continue
            if (float(b[1]) + float(b[3])) / 2.0 < net_y:
                far.append(p)
        if len(far) > 0:
            hit += 1
            gx, gy = gtp["center"]
            best = min(((float(p["box"][0]) + float(p["box"][2])) / 2.0,
                        (float(p["box"][1]) + float(p["box"][3])) / 2.0) for p in far)
            d = min((((float(p["box"][0]) + float(p["box"][2])) / 2.0 - gx) ** 2 +
                     ((float(p["box"][1]) + float(p["box"][3])) / 2.0 - gy) ** 2) ** 0.5
                    for p in far)
            dists.append(d)
            if d < 0.10 * fw:
                matched += 1
        if checked % 50 == 0:
            print(f"  ...{checked}/{len(gtbf)}", flush=True)

    print("", flush=True)
    print(f"=== REAL far-player coverage vs human GT (clip demo3) ===", flush=True)
    print(f"far DETECTED (any far box, no gap-fill): {hit}/{checked} = {100*hit/max(1,checked):.1f}%", flush=True)
    print(f"far CORRECT (within 10%W={0.1*fw:.0f}px of your point): {matched}/{checked} = {100*matched/max(1,checked):.1f}%", flush=True)
    if dists:
        print(f"mean dist when a far box exists: {np.mean(dists):.0f}px (frame W={fw})", flush=True)


if __name__ == "__main__":
    main()
