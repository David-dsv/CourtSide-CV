"""
Evaluate the curve-fit bounce detector across tracking configs vs GT.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import (  # noqa: E402
    load_cache, load_gt, track, smooth_ball_trajectory, evaluate)
from experiments.bounce_curvefit import detect_bounces_curvefit  # noqa: E402

CACHE = "experiments/cache/felix_dets.json"
GT = "tests/fixtures/bounces/felix.bounces.json"


def main():
    cache = load_cache(CACHE)
    gt_frames, fps = load_gt(GT)
    fh, fw = cache["height"], cache["width"]
    frame_diag = float(np.hypot(fw, fh))
    raw = cache["detections"]

    track_configs = [
        ("current", {}),
        ("looser-reinit", {"reinit_gate_ratio": 0.25}),
        ("looser+shortcoast", {"reinit_gate_ratio": 0.25, "kf_max_coast_s": 0.17}),
        ("static20+looser+sc", {"static_pct": 0.20, "reinit_gate_ratio": 0.25, "kf_max_coast_s": 0.17}),
        ("static30+looser+sc", {"static_pct": 0.30, "reinit_gate_ratio": 0.30, "kf_max_coast_s": 0.17}),
    ]

    print(f"{'tracking':<22} {'dens%':>6} {'nP':>4} {'P':>5} {'R':>5} {'F1':>6}  misses")
    best = None
    for label, cfg in track_configs:
        centers, _ = track(raw, frame_diag, fps, cfg)
        d = sum(1 for c in centers if c is not None)
        smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
        bounces = detect_bounces_curvefit(smoothed, fps, fh, fw)
        ev = evaluate([b[0] for b in bounces], gt_frames, fps)
        print(f"{label:<22} {100*d/len(centers):>5.0f} {ev['n_pred']:>4} "
              f"{ev['precision']:>5.2f} {ev['recall']:>5.2f} {ev['f1']:>6.3f}  {ev['fn_frames']}")
        if best is None or ev['f1'] > best[1]['f1']:
            best = (label, ev, cfg)

    print(f"\nBEST tracking for curve-fit: {best[0]}  F1={best[1]['f1']:.3f} "
          f"(P={best[1]['precision']:.2f} R={best[1]['recall']:.2f})")
    print(f"  FP frames: {best[1]['fp_frames']}")
    print(f"  FN frames: {best[1]['fn_frames']}")


if __name__ == "__main__":
    main()
