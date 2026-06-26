"""THROWAWAY: trace the REAL estimate_players_pose internals on a few GT frames.
Monkeypatches is_valid_player to log each pooled candidate's box, far-distance to
GT, validity verdict, n-keypoints, and final pick. Tells us whether the true far
player is (a) not in the pool, (b) gated out by is_valid_player, or (c) out-sorted
in the final selection.

  python tools/pose_gt/_probe_realfn.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import vision.pose as P
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
SAMPLE = [0, 100, 200, 300, 600]


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    frames = list(rdr.iter_frames())
    rdr.release()

    cur = {"gx": 0, "gy": 0}
    orig_valid = P.is_valid_player

    def traced(box, kcf, frame_w, frame_h, court_zones=None, band=None):
        v = orig_valid(box, kcf, frame_w, frame_h, court_zones, band=band)
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = "near" if cy >= net_y else "FAR"
        nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
        d = ((cx - cur["gx"]) ** 2 + (cy - cur["gy"]) ** 2) ** 0.5
        tag = " <<TRUE" if (side == "FAR" and d < 0.10 * fw) else ""
        print(f"    [{side}] c=({cx:5.0f},{cy:4.0f}) h={y2-y1:3.0f} nkp={nkp:2d} "
              f"valid={int(v)} d={d:5.0f}{tag}")
        return v

    P.is_valid_player = traced
    for fidx in SAMPLE:
        if fidx not in gtbf:
            continue
        cur["gx"], cur["gy"] = gtbf[fidx]
        print(f"=== frame {fidx} GT far=({cur['gx']:.0f},{cur['gy']:.0f}) ===")
        players = P.estimate_players_pose(det.person_model, pm, frames[fidx], "cpu",
                                          court_zones=None, det_conf=0.08)
        for p in players:
            b = p["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            side = "near" if cy >= net_y else "FAR"
            d = ((cx - cur["gx"]) ** 2 + (cy - cur["gy"]) ** 2) ** 0.5
            print(f"  PICKED [{side}] c=({cx:5.0f},{cy:4.0f}) d={d:5.0f}")
        print()


if __name__ == "__main__":
    main()
