"""THROWAWAY: authoritative felix far-coverage on the PROD path. Runs the real
estimate_players_pose on the felix proxy-GT frames (built from OLD's reliable picks)
and reports far-CORRECT % within 0.10*fw of the proxy point. Mirrors
measure_far_coverage.py but for felix, so the felix number is a LIVE pipeline number,
not a cache replay.

  python tools/pose_gt/_measure_felix.py
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
CLIP = ROOT / "felix.mp4"
GT = {int(k): v for k, v in json.load(open("/tmp/felix_far_gt.json")).items()}


def main():
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    want = set(GT)
    maxf = max(want)
    frames = {}
    for i, f in enumerate(rdr.iter_frames()):
        if i in want:
            frames[i] = f
        if i >= maxf:
            break
    rdr.release()

    matched = hit = checked = 0
    dists = []
    for fidx, (gx, gy) in GT.items():
        if fidx not in frames:
            continue
        checked += 1
        players = estimate_players_pose(det.person_model, pm, frames[fidx], "cpu",
                                        court_zones=None) or []
        far = [p for p in players if p.get("box") is not None
               and (float(p["box"][1]) + float(p["box"][3])) / 2 < net_y]
        if far:
            hit += 1
            d = min((((float(p["box"][0]) + float(p["box"][2])) / 2 - gx) ** 2 +
                     ((float(p["box"][1]) + float(p["box"][3])) / 2 - gy) ** 2) ** 0.5
                    for p in far)
            dists.append(d)
            if d < 0.10 * fw:
                matched += 1

    print(f"\n=== REAL far-player coverage vs felix proxy-GT (PROD path) ===")
    print(f"far DETECTED: {hit}/{checked} = {100*hit/max(1,checked):.1f}%")
    print(f"far CORRECT (within {0.1*fw:.0f}px): {matched}/{checked} = "
          f"{100*matched/max(1,checked):.1f}%")
    if dists:
        print(f"mean dist: {np.mean(dists):.0f}px (frame W={fw})")


if __name__ == "__main__":
    main()
