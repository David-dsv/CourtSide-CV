"""
Evaluate curve-fit bounce detector on the best tracking config (reinit0.5),
print full match detail, and inspect remaining FN/FP.
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
    d = sum(1 for c in centers if c is not None)

    bounces = detect_bounces_curvefit(smoothed, fps, fh, fw)
    pred = [b[0] for b in bounces]
    ev = evaluate(pred, gt_frames, fps)
    print(f"tracking density: {100*d/len(centers):.0f}%")
    print(f"pred bounces ({len(pred)}): {pred}")
    print(f"P={ev['precision']:.3f} R={ev['recall']:.3f} F1={ev['f1']:.3f} "
          f"(TP={ev['tp']} FP={ev['fp']} FN={ev['fn']}, tol=±{ev['tol']})")
    print(f"FN (missed): {ev['fn_frames']}")
    print(f"FP (false):  {ev['fp_frames']}")

    # show trajectory shape at remaining FN to guide further tuning
    if ev['fn_frames']:
        print("\n--- trajectory at remaining misses ---")
        for g in ev['fn_frames']:
            present = sum(1 for f in range(g-9, g+10) if 0 <= f < len(smoothed) and smoothed[f])
            seg = []
            for f in range(g-6, g+7):
                c = smoothed[f] if 0 <= f < len(smoothed) else None
                seg.append("  .  " if c is None else f"{c[1]:>4d}")
            print(f"  GT {g} (present {present}/19): {' '.join(seg)}")


if __name__ == "__main__":
    main()
