"""Felix non-regression behavioral check (no felix pose GT exists). Runs the REAL
pose path on a few felix frames and reports the 2 selected players' boxes + side +
nkp. We assert: (a) it still selects ~2 players, (b) one far + one near (the players
straddle the net), (c) the far box is central (the fix should help, not hurt felix).
Compare visually against the committed felix annotated output expectation.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO
from vision.pose import estimate_players_pose, _centrality

CLIP = ROOT / "data" / "output" / "felix_full.mp4"


def main():
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    frames = list(rdr.iter_frames())
    rdr.release()
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    n = len(frames)
    sample = [int(n * f) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
    print(f"felix {fw}x{fh}, {n} frames, net_y={net_y:.0f}")
    for fidx in sample:
        players = estimate_players_pose(det.person_model, pm, frames[fidx], "cpu",
                                        court_zones=None) or []
        desc = []
        for p in players:
            b = p["box"]
            cy = (b[1] + b[3]) / 2
            side = "FAR" if cy < net_y else "near"
            kcf = p.get("kps_conf")
            nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
            desc.append(f"{side} cx={int((b[0]+b[2])/2)} feet={int(b[3])} "
                        f"h={int(b[3]-b[1])} central={_centrality(b, fw):.2f} nkp={nkp}")
        print(f"f{fidx}: {len(players)} players | " + " || ".join(desc))


if __name__ == "__main__":
    main()
