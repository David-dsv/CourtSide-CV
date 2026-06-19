"""
Verify the x-continuity FP discriminator: for each predicted bounce (curve-fit v2
on best tracking), measure max |dx| in a +-window and compare TP vs FP.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import (  # noqa: E402
    load_cache, load_gt, track, smooth_ball_trajectory, evaluate, match_bounces)
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
    xs = np.array([c[0] if c is not None else np.nan for c in smoothed], dtype=float)

    bounces = detect_bounces_curvefit_v2(smoothed, fps, fh, fw, win=int(fps*0.28),
                                         min_drop=fh*0.008, max_side_rmse=fh*0.03,
                                         restitution_max=1.8)
    pred = [b[0] for b in bounces]
    m, fp, fn = match_bounces(pred, gt_frames, round(0.15*fps))
    tp_set = set(p for p, t in m)

    win = max(3, int(fps * 0.07))

    def maxdx(i):
        vals = []
        for k in range(i - win, i + win + 1):
            if 0 < k < len(xs) and not np.isnan(xs[k]) and not np.isnan(xs[k-1]):
                vals.append(abs(xs[k] - xs[k-1]))
        return max(vals) if vals else 0.0

    print(f"frame  kind   maxdx(px)  maxdx/fw")
    rows = []
    for p in pred:
        kind = "TP " if p in tp_set else "FP "
        mdx = maxdx(p)
        rows.append((p, kind, mdx, mdx/fw))
        print(f"{p:5d}  {kind}   {mdx:7.0f}    {mdx/fw:.3f}")

    # Apply gate at 6% fw and re-evaluate
    print("\n--- apply x-continuity gate at 0.06*fw ---")
    for thr in [0.04, 0.05, 0.06, 0.08]:
        kept = [p for p in pred if maxdx(p) <= thr * fw]
        ev = evaluate(kept, gt_frames, fps)
        print(f"  thr={thr:.2f}*fw: nP={ev['n_pred']:2d} P={ev['precision']:.3f} "
              f"R={ev['recall']:.3f} F1={ev['f1']:.3f}")


if __name__ == "__main__":
    main()
