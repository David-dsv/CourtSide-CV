"""
Grid-search curve-fit bounce params on the best tracking config, maximizing F1
vs GT. Reports the top configs. Keeps params as fps/frame-height ratios.
"""
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import (  # noqa: E402
    load_cache, load_gt, track, smooth_ball_trajectory, evaluate)
from experiments.bounce_curvefit import detect_bounces_curvefit  # noqa: E402

CACHE = "experiments/cache/felix_dets.json"
GT = "tests/fixtures/bounces/felix.bounces.json"
TRACK_CFG = {"reinit_gate_ratio": 0.50, "kf_max_coast_s": 0.12,
             "process_noise": 200, "kf_gate_base": 150, "kf_gate_growth": 80,
             "kf_gate_max_ratio": 0.35}


def main():
    cache = load_cache(CACHE)
    gt_frames, fps = load_gt(GT)
    fh, fw = cache["height"], cache["width"]
    frame_diag = float(np.hypot(fw, fh))
    raw = cache["detections"]
    centers, _ = track(raw, frame_diag, fps, TRACK_CFG)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))

    win_s = [0.10, 0.13, 0.17, 0.22]          # half-window seconds
    drop_r = [0.006, 0.009, 0.012, 0.016]      # min y-swing ratio of frame_height
    rmse_r = [0.012, 0.018, 0.025, 0.035]      # max side rmse ratio
    slope_r = [0.0005, 0.0010]                  # min slope ratio

    results = []
    for ws, dr, rr, sr in itertools.product(win_s, drop_r, rmse_r, slope_r):
        win = max(4, int(fps * ws))
        bounces = detect_bounces_curvefit(
            smoothed, fps, fh, fw,
            win=win,
            min_drop=fh * dr,
            max_side_rmse=fh * rr,
            min_slope=fh * sr,
        )
        ev = evaluate([b[0] for b in bounces], gt_frames, fps)
        results.append((ev['f1'], ev['precision'], ev['recall'], ev['tp'], ev['fp'], ev['fn'],
                        ws, dr, rr, sr))

    results.sort(key=lambda r: (-r[0], -r[2]))
    print(f"{'F1':>5} {'P':>5} {'R':>5} {'TP':>3} {'FP':>3} {'FN':>3} | win_s drop_r rmse_r slope_r")
    for r in results[:12]:
        print(f"{r[0]:>5.3f} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>3} {r[4]:>3} {r[5]:>3} | "
              f"{r[6]:>5} {r[7]:>5} {r[8]:>5} {r[9]:>6}")


if __name__ == "__main__":
    main()
