"""
Sweep tracking configs to find the best CLEAN trajectory, and inspect the
y-trajectory around GT bounces to understand what a bounce looks like in the data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import (  # noqa: E402
    load_cache, load_gt, track, smooth_ball_trajectory, evaluate)
from experiments.bounce_baseline import detect_bounces_velocity_signchange  # noqa: E402

CACHE = "experiments/cache/felix_dets.json"
GT = "tests/fixtures/bounces/felix.bounces.json"


def density(centers):
    return sum(1 for c in centers if c is not None)


def main():
    cache = load_cache(CACHE)
    gt_frames, fps = load_gt(GT)
    fh, fw = cache["height"], cache["width"]
    frame_diag = float(np.hypot(fw, fh))
    raw = cache["detections"]

    configs = [
        ("current", {}),
        ("looser-reinit-0.25", {"reinit_gate_ratio": 0.25}),
        ("short-coast-10f", {"kf_max_coast_s": 0.17}),
        ("looser+shortcoast", {"reinit_gate_ratio": 0.25, "kf_max_coast_s": 0.17}),
        ("static-12pct", {"static_pct": 0.12}),
        ("static-20pct", {"static_pct": 0.20}),
        ("static-20pct+looser", {"static_pct": 0.20, "reinit_gate_ratio": 0.25, "kf_max_coast_s": 0.17}),
        ("static-30pct+looser", {"static_pct": 0.30, "reinit_gate_ratio": 0.30, "kf_max_coast_s": 0.17}),
        ("widegate", {"kf_gate_base": 120.0, "kf_gate_growth": 60.0, "static_pct": 0.20}),
    ]

    print(f"{'config':<28} {'dens%':>6} {'2nd%':>6} {'nP':>4} {'P':>5} {'R':>5} {'F1':>5}")
    for label, cfg in configs:
        centers, speeds = track(raw, frame_diag, fps, cfg)
        d = density(centers)
        sh = sum(1 for c in centers[700:] if c is not None)
        smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
        bounces = detect_bounces_velocity_signchange(smoothed, speeds, fps, fh, fw)
        ev = evaluate([b[0] for b in bounces], gt_frames, fps)
        print(f"{label:<28} {100*d/len(centers):>5.0f} {100*sh/(len(centers)-700):>5.0f} "
              f"{ev['n_pred']:>4} {ev['precision']:>5.2f} {ev['recall']:>5.2f} {ev['f1']:>5.2f}")

    # Inspect y-trajectory around a few GT bounces using the BEST-density clean config
    print("\n--- y-trajectory around GT bounces (static-20pct config) ---")
    cfg = {"static_pct": 0.20}
    centers, _ = track(raw, frame_diag, fps, cfg)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
    for g in [161, 246, 329, 547]:
        ys = []
        for f in range(g - 8, g + 9):
            c = smoothed[f] if 0 <= f < len(smoothed) else None
            ys.append("None" if c is None else str(c[1]))
        print(f"  GT {g}: y[{g-8}..{g+8}] = {ys}")


if __name__ == "__main__":
    main()
