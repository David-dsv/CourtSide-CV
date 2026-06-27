"""THROWAWAY: felix non-regression. Run OLD (base branch) vs NEW estimate_players_pose
on the same felix frames and compare the far/near picks. felix has no pose GT, so we
assert SANITY + STABILITY: the NEW far pick should stay on-court (central-ish, not a
corner), and should not drift wildly from the OLD pick on frames where OLD was good.

  python tools/pose_gt/_felix_regression.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

CLIP = ROOT / "felix.mp4"


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    new_mod = load_mod(ROOT / "vision" / "pose.py", "pose_new")
    old_mod = load_mod("/tmp/pose_OLD.py", "pose_old")

    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    frames = []
    for i, f in enumerate(rdr.iter_frames()):
        if 60 * int(rdr.fps) <= i < 60 * int(rdr.fps) + 200 and i % 25 == 0:
            frames.append((i, f))
        if i >= 60 * int(rdr.fps) + 200:
            break
    rdr.release()
    print(f"felix {fw}x{fh} net_y={net_y:.0f}; {len(frames)} sample frames\n")

    def far_near(players):
        far = near = None
        for p in players:
            b = p.get("box")
            if b is None:
                continue
            cy = (b[1] + b[3]) / 2
            c = ((b[0] + b[2]) / 2, cy)
            if cy < net_y:
                if far is None:
                    far = c
            else:
                if near is None:
                    near = c
        return far, near

    drift = []
    new_oncourt = old_oncourt = 0
    for fidx, frame in frames:
        po = old_mod.estimate_players_pose(det.person_model, pm, frame, "cpu",
                                           court_zones=None)
        pn = new_mod.estimate_players_pose(det.person_model, pm, frame, "cpu",
                                           court_zones=None)
        ofar, onear = far_near(po)
        nfar, nnear = far_near(pn)
        # "on court" sanity: far center within central 70% horizontally
        def central(c):
            return c is not None and 0.15 < c[0] / fw < 0.85
        new_oncourt += int(central(nfar))
        old_oncourt += int(central(ofar))
        d = (((ofar[0]-nfar[0])**2+(ofar[1]-nfar[1])**2)**0.5
             if ofar and nfar else None)
        if d is not None:
            drift.append(d)
        print(f"  f{fidx}: OLD far={_fmt(ofar)} near={_fmt(onear)} | "
              f"NEW far={_fmt(nfar)} near={_fmt(nnear)} | drift={d:.0f}"
              if d is not None else
              f"  f{fidx}: OLD far={_fmt(ofar)} | NEW far={_fmt(nfar)}")

    print(f"\n  far central(15-85%W): OLD {old_oncourt}/{len(frames)}  "
          f"NEW {new_oncourt}/{len(frames)}")
    if drift:
        print(f"  far-pick drift OLD->NEW: mean {np.mean(drift):.0f}px "
              f"median {np.median(drift):.0f}px max {np.max(drift):.0f}px")


def _fmt(c):
    return f"({c[0]:.0f},{c[1]:.0f})" if c else "None"


if __name__ == "__main__":
    main()
