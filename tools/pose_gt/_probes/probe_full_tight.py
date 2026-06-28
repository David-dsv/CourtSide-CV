"""FULL 225-frame live far-CORRECT at multiple thresholds + per-frame worst offenders.
The honest, no-sampling verification of whether the current code already nails the
far player. Prints any frame whose selected far box is >3%W (58px) from GT.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from utils.video_utils import VideoReader
from vision.pose import estimate_players_pose
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

GT = ROOT / "tests/fixtures/pose_gt/tennis_demo3.pose_gt.json"
CLIP = ROOT / "data/output/tennis_demo3.mp4"

def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP)); fw, fh = rdr.width, rdr.height
    frames = list(rdr.iter_frames()); rdr.release()
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    ths = [0.10, 0.05, 0.03, 0.02, 0.015]
    correct = {t: 0 for t in ths}; checked = 0; hit = 0; dists = []; worst = []
    for fidx, center in sorted(gtbf.items()):
        if fidx >= len(frames): continue
        checked += 1
        players = estimate_players_pose(det.person_model, pm, frames[fidx], "cpu",
                                        court_zones=None) or []
        far = [p for p in players if p.get("box") is not None
               and (float(p["box"][1])+float(p["box"][3]))/2 < net_y]
        if not far:
            worst.append((fidx, -1)); continue
        hit += 1
        gx, gy = center
        d = min((((float(p["box"][0])+float(p["box"][2]))/2-gx)**2 +
                 ((float(p["box"][1])+float(p["box"][3]))/2-gy)**2)**0.5 for p in far)
        dists.append(d)
        for t in ths:
            if d < t*fw: correct[t] += 1
        if d >= 0.03*fw: worst.append((fidx, int(d)))
        if checked % 75 == 0: print(f"  ...{checked}/{len(gtbf)}", flush=True)
    print(f"\n=== FULL {checked} frames, current code (det_conf=0.08 imgsz=1280) ===", flush=True)
    for t in ths:
        print(f"  far-CORRECT @ {t*100:.1f}%W ({t*fw:.0f}px): {correct[t]}/{checked} = {100*correct[t]/checked:.1f}%", flush=True)
    print(f"  far DETECTED (a far box exists): {hit}/{checked} = {100*hit/checked:.1f}%", flush=True)
    da = np.array(dists)
    print(f"  dist px: median={np.median(da):.0f} mean={np.mean(da):.0f} p90={np.percentile(da,90):.0f} p99={np.percentile(da,99):.0f} max={np.max(da):.0f}", flush=True)
    print(f"  frames >3%W off (or no far box -1): {worst}", flush=True)

if __name__ == "__main__":
    main()
