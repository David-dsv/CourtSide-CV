"""
Test shot detection + FH/BH + quality on felix using the WASB trajectory.
Computes pose only at hit-candidate frames (fast).
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from utils.video_utils import VideoReader  # noqa: E402
from vision.pose import estimate_players_pose  # noqa: E402
from vision.shots import detect_hits, classify_forehand_backhand  # noqa: E402

W = "experiments/wasb_bounce/preds/felix_wasb_centers.json"
GT_BOUNCES = [52, 161, 245, 329, 423, 545, 720, 883, 979, 1140, 1253, 1346, 1440, 1518, 1667, 1795]


def main():
    w = json.load(open(W))
    n = w["n_frames"]
    fps = w["fps"]
    start = w.get("start_frame", 0)
    fw, fh = w.get("width", 1920), w.get("height", 1080)
    traj = [None] * n
    for e in w["centers"]:
        f = e["frame"] - start
        if 0 <= f < n and e.get("x") is not None:
            traj[f] = (e["x"], e["y"])

    # First pass: find hit candidate frames WITHOUT pose (proximity filter off)
    # by reusing detect_hits with empty players → we just want the vy reversals.
    # Simpler: replicate the vy reversal scan here to know which frames need pose.
    ys = np.array([c[1] if c else np.nan for c in traj])
    vy = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ys[i]) and not np.isnan(ys[i - 1]):
            vy[i] = ys[i] - ys[i - 1]
    win = max(3, int(fps * 0.07))
    bounce_guard = int(fps * 0.12)
    cand = []
    for i in range(win, n - win):
        if traj[i] is None:
            continue
        a = np.nanmean(vy[max(0, i - win):i])
        b = np.nanmean(vy[i:i + win + 1])
        if np.isnan(a) or np.isnan(b):
            continue
        if a > 1.0 and b < -1.0 and not any(abs(i - bf) <= bounce_guard for bf in GT_BOUNCES):
            cand.append(i)
    print(f"hit candidate frames (vy reversals, non-bounce): {len(cand)}")

    # Compute pose at candidate frames
    det = YOLO("yolov8m.pt")
    pose = YOLO("yolov8m-pose.pt")
    reader = VideoReader("felix.mp4")
    want = set(cand)
    players_per_frame = [None] * n
    for i, frame in enumerate(reader.iter_frames()):
        if i >= n:
            break
        if i in want:
            players_per_frame[i] = estimate_players_pose(det, pose, frame, "mps",
                                                         ball_xy=traj[i])
    reader.release()

    hits = detect_hits(traj, players_per_frame, fps, fh, fw, bounce_frames=GT_BOUNCES)
    print(f"\nattributed hits: {len(hits)}")
    for h in hits:
        fhb = classify_forehand_backhand(h, players_per_frame)
        t = h["frame"] / fps
        print(f"  f={h['frame']:4d} ({t:4.1f}s) player={h['player_side']:4s} "
              f"x={h['x']:4d} y={h['y']:4d} -> {fhb}")


if __name__ == "__main__":
    main()
