"""THROWAWAY: build a felix far-player PROXY-GT. felix has no human pose GT, but the
OLD (base-branch) selector is RELIABLE on felix (validated visually: it picks the
real HUA-YU far player when it finds one). So we sample felix every K frames, run OLD,
and record OLD's far pick as the proxy-GT for the frames where OLD returned a far
player that ALSO passes the keypoint-validity gate (a real skeleton, not a guess).
Writes /tmp/felix_far_gt.json: {frame: [cx, cy]}.

  python tools/pose_gt/_build_felix_gt.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

CLIP = ROOT / "felix.mp4"


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    old = load("/tmp/pose_OLD.py", "po")
    r = VideoReader(str(CLIP))
    fw, fh = r.width, r.height
    ny = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    gt = {}
    n_far = 0
    for i, f in enumerate(r.iter_frames()):
        if i % 50 != 0:        # sample every 50 frames
            continue
        if i > 4000:
            break
        players = old.estimate_players_pose(det.person_model, pm, f, "cpu",
                                            court_zones=None)
        for p in players:
            b = p.get("box")
            kcf = p.get("kps_conf")
            if b is None:
                continue
            cy = (b[1] + b[3]) / 2
            if cy >= ny:
                continue
            # accept only a far pick with a REAL skeleton (>=10 kpts) -> trustworthy
            nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
            if nkp >= 10:
                gt[i] = [round(float(b[0] + b[2]) / 2, 1), round(float(cy), 1)]
                n_far += 1
            break
    r.release()
    json.dump(gt, open("/tmp/felix_far_gt.json", "w"))
    print(f"felix proxy-GT: {n_far} frames with a trustworthy OLD far pick "
          f"(of {len(range(0,4000,50))} sampled). frame {fw}x{fh}")
    if gt:
        xs = [v[0] / fw for v in gt.values()]
        print(f"  far xr range: {min(xs):.2f}-{max(xs):.2f} "
              f"(corner-angle -> mostly off-centre right)")


if __name__ == "__main__":
    main()
