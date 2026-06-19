"""
WASB smoke test on felix.mp4: run the heatmap tracker over the GT window, then
(a) report trajectory density + coverage of GT bounces, (b) run the curve-fit
bounce detector on the WASB trajectory and score vs GT — directly comparable to
the YOLO+Kalman numbers.
"""
import sys
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


def main():
    gt_frames, fps = load_gt(GT)

    logger.info("Reading frames...")
    reader = VideoReader(VIDEO)
    fh, fw = reader.height, reader.width
    frames = []
    for i, f in enumerate(reader.iter_frames()):
        if i >= NFRAMES:
            break
        frames.append(f)
    reader.release()
    logger.info(f"Read {len(frames)} frames {fw}x{fh}")

    tracker = HeatmapBallTracker(WEIGHTS, device="mps", score_threshold=0.5)
    import time
    t0 = time.time()
    centers = tracker.track_ball(frames)
    dt = time.time() - t0
    dens = sum(1 for c in centers if c is not None)
    sh = sum(1 for c in centers[700:] if c is not None)
    logger.info(f"WASB tracked in {dt:.1f}s ({len(frames)/dt:.1f} FPS)")
    print(f"\nWASB trajectory density: {dens}/{len(centers)} ({100*dens/len(centers):.0f}%)  "
          f"2nd-half(700+): {sh}/{len(centers)-700} ({100*sh/(len(centers)-700):.0f}%)")

    # GT-window coverage
    cov = 0
    for g in gt_frames:
        present = sum(1 for f in range(g-9, g+10) if 0 <= f < len(centers) and centers[f] is not None)
        if present / 19 >= 0.6:
            cov += 1
    print(f"GT-window coverage: {cov}/16")

    # bounce detection on WASB trajectory (no smoothing first, then with)
    for tag, traj in [("raw", centers),
                      ("smoothed", smooth_ball_trajectory(centers, max_gap=int(fps*0.4)))]:
        for xj in [0.0, 0.04]:
            b = detect_bounces_curvefit_v2(traj, fps, fh, fw, win=int(fps*0.28),
                                           min_drop=fh*0.008, max_side_rmse=fh*0.03,
                                           restitution_max=1.8, max_xjump_frac=xj)
            ev = evaluate([x[0] for x in b], gt_frames, fps)
            print(f"  [{tag:8} xj={xj:.2f}] P={ev['precision']:.3f} R={ev['recall']:.3f} "
                  f"F1={ev['f1']:.3f} (TP={ev['tp']} FP={ev['fp']} FN={ev['fn']})")

    # save WASB centers for reuse
    import json
    Path("experiments/cache").mkdir(parents=True, exist_ok=True)
    json.dump({"centers": [list(c) if c else None for c in centers],
               "fps": fps, "width": fw, "height": fh},
              open("experiments/cache/felix_wasb.json", "w"))
    print("\nSaved WASB trajectory -> experiments/cache/felix_wasb.json")


if __name__ == "__main__":
    main()
