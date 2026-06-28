"""Probe: measure LIVE far-CORRECT under different det_conf / det_imgsz settings.
Runs the FULL estimate_players_pose (selection included) on all 225 GT frames.
This is the faithful thermometer: it measures the SELECTED far player, not just
candidate existence. Read-only; prod untouched.

Usage: python probe_settings.py <det_conf> <det_imgsz>   (defaults 0.08 1280)
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
    det_conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
    det_imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP)); fw, fh = rdr.width, rdr.height
    frames = list(rdr.iter_frames()); rdr.release()
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    checked = correct = hit = 0
    miss_frames = []
    for fidx, center in sorted(gtbf.items()):
        if fidx >= len(frames): continue
        checked += 1
        players = estimate_players_pose(det.person_model, pm, frames[fidx], "cpu",
                                        court_zones=None, det_conf=det_conf,
                                        det_imgsz=det_imgsz) or []
        far = [p for p in players if p.get("box") is not None
               and (float(p["box"][1])+float(p["box"][3]))/2 < net_y]
        if far:
            hit += 1
            gx, gy = center
            d = min((((float(p["box"][0])+float(p["box"][2]))/2-gx)**2 +
                     ((float(p["box"][1])+float(p["box"][3]))/2-gy)**2)**0.5 for p in far)
            if d < 0.10*fw: correct += 1
            else: miss_frames.append((fidx, int(d)))
        else:
            miss_frames.append((fidx, -1))
        if checked % 75 == 0:
            print(f"  ...{checked}/{len(gtbf)} correct={correct}", flush=True)
    print(f"\n=== det_conf={det_conf} det_imgsz={det_imgsz} ===", flush=True)
    print(f"far DETECTED: {hit}/{checked} = {100*hit/checked:.1f}%", flush=True)
    print(f"far CORRECT : {correct}/{checked} = {100*correct/checked:.1f}%", flush=True)
    print(f"miss frames ({len(miss_frames)}): {miss_frames}", flush=True)

if __name__ == "__main__":
    main()
