"""
Sweep WASB score_threshold x frame_step on felix, reading frames once.
Find the config that maximizes GT-window coverage + bounce F1.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from models.wasb_ball_tracker import HeatmapBallTracker  # noqa: E402
from utils.video_utils import VideoReader  # noqa: E402
from experiments.tracking_lab import load_gt, smooth_ball_trajectory, evaluate  # noqa: E402
from experiments.bounce_curvefit_v2 import detect_bounces_curvefit_v2  # noqa: E402

WEIGHTS = "training/wasb/wasb_tennis.pth.tar"
GT = "tests/fixtures/bounces/felix.bounces.json"
VIDEO = "felix.mp4"
NFRAMES = 1850


def cov(centers, gt_frames):
    c = 0
    for g in gt_frames:
        p = sum(1 for f in range(g-9, g+10) if 0 <= f < len(centers) and centers[f] is not None)
        if p / 19 >= 0.6:
            c += 1
    return c


def main():
    gt_frames, fps = load_gt(GT)
    reader = VideoReader(VIDEO)
    fh, fw = reader.height, reader.width
    frames = []
    for i, f in enumerate(reader.iter_frames()):
        if i >= NFRAMES:
            break
        frames.append(f)
    reader.release()
    logger.info(f"Read {len(frames)} frames")

    for thr in [0.3, 0.5]:
        tracker = HeatmapBallTracker(WEIGHTS, device="mps", score_threshold=thr)
        for step in [1, 2, 3]:
            t0 = time.time()
            centers = tracker.track_ball(frames, frame_step=step)
            dt = time.time() - t0
            dens = sum(1 for c in centers if c is not None)
            cv = cov(centers, gt_frames)
            sm = smooth_ball_trajectory(centers, max_gap=int(fps*0.4))
            b = detect_bounces_curvefit_v2(sm, fps, fh, fw, win=int(fps*0.28),
                                           min_drop=fh*0.008, max_side_rmse=fh*0.03,
                                           restitution_max=1.8, max_xjump_frac=0.04)
            ev = evaluate([x[0] for x in b], gt_frames, fps)
            print(f"thr={thr} step={step}: dens={100*dens/len(centers):3.0f}% cov={cv:2d}/16 "
                  f"P={ev['precision']:.3f} R={ev['recall']:.3f} F1={ev['f1']:.3f}  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
