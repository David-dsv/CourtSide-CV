"""
Find a tracking config that fills the holes at the missed GT bounces, measuring
coverage specifically in a ±9-frame window around every GT bounce (that's what
matters for bounce recall), plus overall clean density.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import load_cache, load_gt, track, smooth_ball_trajectory  # noqa: E402

CACHE = "experiments/cache/felix_dets.json"
GT = "tests/fixtures/bounces/felix.bounces.json"


def gt_window_coverage(smoothed, gt_frames, w=9):
    """How many GT bounces have at least 60% of their ±w window tracked."""
    covered = 0
    detail = []
    for g in gt_frames:
        present = sum(1 for f in range(g - w, g + w + 1)
                      if 0 <= f < len(smoothed) and smoothed[f] is not None)
        frac = present / (2 * w + 1)
        if frac >= 0.6:
            covered += 1
        else:
            detail.append(g)
    return covered, detail


def main():
    cache = load_cache(CACHE)
    gt_frames, fps = load_gt(GT)
    fh, fw = cache["height"], cache["width"]
    frame_diag = float(np.hypot(fw, fh))
    raw = cache["detections"]

    configs = [
        ("looser+sc(base)", {"reinit_gate_ratio": 0.25, "kf_max_coast_s": 0.17}),
        ("reinit0.35", {"reinit_gate_ratio": 0.35, "kf_max_coast_s": 0.17}),
        ("reinit0.5", {"reinit_gate_ratio": 0.50, "kf_max_coast_s": 0.12}),
        ("gate+reinit", {"reinit_gate_ratio": 0.40, "kf_max_coast_s": 0.12,
                         "kf_gate_base": 120, "kf_gate_growth": 70, "kf_gate_max_ratio": 0.30}),
        ("procnoise150", {"reinit_gate_ratio": 0.40, "kf_max_coast_s": 0.12,
                          "process_noise": 150, "kf_gate_base": 120, "kf_gate_growth": 70,
                          "kf_gate_max_ratio": 0.30}),
        ("aggressive", {"reinit_gate_ratio": 0.50, "kf_max_coast_s": 0.10,
                        "process_noise": 200, "kf_gate_base": 150, "kf_gate_growth": 80,
                        "kf_gate_max_ratio": 0.35, "static_pct": 0.20}),
    ]
    for label, cfg in configs:
        centers, _ = track(raw, frame_diag, fps, cfg)
        smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
        d = sum(1 for c in centers if c is not None)
        cov, missing = gt_window_coverage(smoothed, gt_frames)
        print(f"{label:<18} dens={100*d/len(centers):>3.0f}%  GT-window-cov={cov:>2}/16  "
              f"uncovered={missing}")


if __name__ == "__main__":
    main()
