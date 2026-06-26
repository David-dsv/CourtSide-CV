"""THROWAWAY: faithful (sequential, stateful) far-coverage measurement that runs
the pipeline's pose path frame-by-frame in ORDER (as run_pipeline_8s does) and
threads an optional motion state into estimate_players_pose, so a static-spectator
penalty can be evaluated the way it would actually run (consecutive frames), not
on the non-uniform GT subsampling. Scores only the verified GT frames.

  python tools/pose_gt/_measure_motion.py [det_conf] [use_motion 0|1]
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

DET_CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
USE_MOTION = (len(sys.argv) > 2 and sys.argv[2] == "1")


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    last_gt = max(gtbf)
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    print(f"GT {len(gtbf)} frames; det_conf={DET_CONF} use_motion={USE_MOTION}",
          flush=True)

    state = {} if USE_MOTION else None
    hit = matched = checked = 0
    dists = []
    for fidx, frame in enumerate(rdr.iter_frames()):
        if fidx > last_gt:
            break
        kwargs = dict(court_zones=None, det_conf=DET_CONF)
        if USE_MOTION:
            kwargs["track_state"] = state
        players = estimate_players_pose(det.person_model, pm, frame, "cpu",
                                        **kwargs) or []
        if fidx not in gtbf:
            continue
        checked += 1
        far = [p for p in players if p.get("box") is not None
               and (float(p["box"][1]) + float(p["box"][3])) / 2.0 < net_y]
        if far:
            hit += 1
            gx, gy = gtbf[fidx]["center"]
            d = min((((float(p["box"][0]) + float(p["box"][2])) / 2.0 - gx) ** 2 +
                     ((float(p["box"][1]) + float(p["box"][3])) / 2.0 - gy) ** 2) ** 0.5
                    for p in far)
            dists.append(d)
            if d < 0.10 * fw:
                matched += 1
        if checked % 50 == 0:
            print(f"  ...{checked}/{len(gtbf)}", flush=True)
    rdr.release()
    print(f"\n=== sequential far-coverage (det_conf={DET_CONF} motion={USE_MOTION}) ===")
    print(f"far DETECTED: {hit}/{checked} = {100*hit/max(1,checked):.1f}%")
    print(f"far CORRECT (within 10%W={0.1*fw:.0f}px): {matched}/{checked} = "
          f"{100*matched/max(1,checked):.1f}%")
    if dists:
        print(f"mean dist when a far box exists: {np.mean(dists):.0f}px")


if __name__ == "__main__":
    main()
