"""Grid-search curve-fit v2 on best tracking config."""
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import (  # noqa: E402
    load_cache, load_gt, track, smooth_ball_trajectory, evaluate)
from experiments.bounce_curvefit_v2 import detect_bounces_curvefit_v2  # noqa: E402

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

    win_s = [0.13, 0.17, 0.22, 0.28]
    drop_r = [0.008, 0.011, 0.015]
    rmse_r = [0.022, 0.030, 0.040]
    rest = [1.8, 2.2, 3.0]

    results = []
    for ws, dr, rr, re in itertools.product(win_s, drop_r, rmse_r, rest):
        win = max(4, int(fps * ws))
        bounces = detect_bounces_curvefit_v2(
            smoothed, fps, fh, fw, win=win,
            min_drop=fh * dr, max_side_rmse=fh * rr, restitution_max=re)
        ev = evaluate([b[0] for b in bounces], gt_frames, fps)
        results.append((ev['f1'], ev['precision'], ev['recall'], ev['tp'], ev['fp'], ev['fn'],
                        ws, dr, rr, re, ev['fn_frames'], ev['fp_frames']))

    results.sort(key=lambda r: (-r[0], -r[2]))
    print(f"{'F1':>5} {'P':>5} {'R':>5} {'TP':>3} {'FP':>3} {'FN':>3} | win drop rmse rest")
    for r in results[:10]:
        print(f"{r[0]:>5.3f} {r[1]:>5.2f} {r[2]:>5.2f} {r[3]:>3} {r[4]:>3} {r[5]:>3} | "
              f"{r[6]} {r[7]} {r[8]} {r[9]}")
    best = results[0]
    print(f"\nBEST v2: F1={best[0]:.3f} P={best[1]:.2f} R={best[2]:.2f}")
    print(f"  params: win_s={best[6]} drop_r={best[7]} rmse_r={best[8]} rest={best[9]}")
    print(f"  FN: {best[10]}")
    print(f"  FP: {best[11]}")


if __name__ == "__main__":
    main()
