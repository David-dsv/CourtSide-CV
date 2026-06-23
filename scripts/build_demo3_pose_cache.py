"""Build a pose+ball cache for the demo3 clip (tennis.mp4 -s 73 -d 13) so the
shots-regression test runs without re-running YOLO+pose inference every time.

Reproduces the pipeline's Pass 1 (ball detect + Kalman + smooth) and Pass 1.5
(per-frame estimate_players_pose) on the segment, then dumps:
  {video, fps, width, height,
   ball_centers_smoothed: [[x,y]|None]*N,
   players_per_frame: [[{box,kps_xy,kps_conf},...] | None]*N,
   bounce_frames: [int, ...]}

Run once:  venv/bin/python scripts/build_demo3_pose_cache.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cv2

from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory
from vision.pose import estimate_players_pose

VIDEO = ROOT / "tennis.mp4"
START_SEC = 73.0
DURATION = 13.0
OUT = ROOT / "tests" / "fixtures" / "cache" / "demo3_pose.json"


def main():
    device = "cpu"
    det = TennisYOLODetector(device=device)
    from ultralytics import YOLO
    pose_model = YOLO("yolov8m-pose.pt")

    reader = VideoReader(str(VIDEO))
    fps = reader.fps
    start_frame = int(START_SEC * fps)
    max_frames = int(DURATION * fps)
    reader.seek(start_frame)
    fw, fh = reader.width, reader.height

    # Pass 1: raw ball detections (COCO sports_ball via custom model)
    raw = []
    frames = []
    ball_conf_thresh = 0.2
    for i, fr in enumerate(reader.iter_frames()):
        if i >= max_frames:
            break
        frames.append(fr)
        d = det.detect(fr, classes=[32])
        ball_mask = d["class_ids"] == 32
        candidates = []
        if ball_mask.any():
            boxes = d["boxes"][ball_mask]
            scores = d["scores"][ball_mask]
            for j in range(len(boxes)):
                if scores[j] >= ball_conf_thresh:
                    bx = boxes[j]
                    cx, cy = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
                    candidates.append((cx, cy, float(scores[j])))
        raw.append(candidates)
    reader.release()
    n = len(frames)

    # Kalman track (port of pipeline config, reuse the regression-test impl)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tbr", ROOT / "tests" / "test_bounce_regression.py")
    tbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbr)
    centers = tbr._track(raw, fw, fh, fps)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
    # serialize None-safe
    ball_serial = [list(map(float, c)) if c is not None else None for c in smoothed]

    # bounces for gating (curve-fit detector on smoothed)
    speeds = [0.0] * len(smoothed)
    bounces = detect_bounces_from_trajectory(smoothed, speeds, fps, fh, fw)
    bounce_frames = [int(b[0]) for b in bounces]

    # Pass 1.5: pose per frame → lock identity via TwoPlayerTracker (the production
    # path). The LOCKED tracks (P1 far / P2 near, gap-filled, smoothed) are what
    # detect_hits consumes at runtime — storing the raw per-frame pose instead
    # would feed it unstable identity ordering, which injects fake wrist-speed
    # spikes. So we store the locked tracks.
    from vision.player_track import TwoPlayerTracker
    net_y = fh * 0.47
    ptracker = TwoPlayerTracker(fw, fh, net_y, fps)
    raw_pose = []
    for i, fr in enumerate(frames):
        ball_xy = smoothed[i] if i < len(smoothed) else None
        players = estimate_players_pose(
            det.person_model, pose_model, fr, device, ball_xy=ball_xy, court_zones=None)
        raw_pose.append(players)
        ptracker.add_frame(players)
        if i % 50 == 0:
            print(f"  pose {i}/{n}")
    player_tracks = ptracker.finalize()  # {1: [pose|None]*N (far), 2: [pose|None]*N (near)}

    # Serialize: players_per_frame[i] = [far_player|None, near_player|None] (locked).
    # Stash the raw pose too under a separate key for diagnostics if needed.
    def _ser(p):
        if p is None:
            return None
        kx = p.get("kps_xy")
        kc = p.get("kps_conf")
        return {
            "box": [float(v) for v in p["box"]],
            "kps_xy": kx.tolist() if kx is not None else None,
            "kps_conf": kc.tolist() if kc is not None else None,
        }
    players_per_frame = []
    for i in range(n):
        far = player_tracks[1][i] if i < len(player_tracks[1]) else None
        near = player_tracks[2][i] if i < len(player_tracks[2]) else None
        # match the runtime convention: players_per_frame[i] is a list of player
        # dicts; detect_hits buckets by box feet vs net_y, so order doesn't matter,
        # but we put near first then far to mirror typical detection order.
        slot = []
        if near is not None:
            slot.append(_ser(near))
        if far is not None:
            slot.append(_ser(far))
        players_per_frame.append(slot)
    cov = ptracker.coverage()
    print(f"  locked: P1(far) {cov[1]*100:.0f}%  P2(near) {cov[2]*100:.0f}%")

    out = {
        "video": str(VIDEO),
        "fps": fps,
        "width": fw,
        "height": fh,
        "n_frames": n,
        "start_frame": start_frame,
        "ball_centers_smoothed": ball_serial,
        "bounce_frames": bounce_frames,
        "players_per_frame": players_per_frame,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nSaved → {OUT}  ({n} frames, {len(bounce_frames)} bounces)")
    print(f"  size: {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
