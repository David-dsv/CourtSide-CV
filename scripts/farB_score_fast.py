"""Fast far-CORRECT scorer: identical to tools/pose_gt/measure_far_coverage.py but
passes a tiny far_pose_imgsz so the (skeleton-only) far pose call is cheap. The far
BOX — and therefore far-CORRECT — is selected GEOMETRICALLY, independent of the pose
call, so this returns the SAME far-CORRECT % as the full scorer, ~3x faster. Use for
the inner loop; confirm the final number with the real scorer at production settings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.video_utils import VideoReader
from vision.pose import estimate_players_pose
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
FAR_POSE_IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 160


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
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
                                        court_zones=None,
                                        far_pose_imgsz=FAR_POSE_IMGSZ) or []
        far = [p for p in players if p.get("box") is not None
               and (float(p["box"][1]) + float(p["box"][3])) / 2.0 < net_y]
        if far:
            hit += 1
            gx, gy = gtp["center"]
            d = min((((float(p["box"][0]) + float(p["box"][2])) / 2.0 - gx) ** 2 +
                     ((float(p["box"][1]) + float(p["box"][3])) / 2.0 - gy) ** 2) ** 0.5
                    for p in far)
            dists.append(d)
            if d < 0.10 * fw:
                matched += 1
        if checked % 50 == 0:
            print(f"  ...{checked}/{len(gtbf)}", flush=True)

    print(f"\n=== fast far-coverage (far_pose_imgsz={FAR_POSE_IMGSZ}) ===", flush=True)
    print(f"far DETECTED: {hit}/{checked} = {100*hit/max(1,checked):.1f}%", flush=True)
    print(f"far CORRECT (<10%W={0.1*fw:.0f}px): {matched}/{checked} = "
          f"{100*matched/max(1,checked):.1f}%", flush=True)
    if dists:
        print(f"mean dist when far box exists: {np.mean(dists):.0f}px", flush=True)


if __name__ == "__main__":
    main()
