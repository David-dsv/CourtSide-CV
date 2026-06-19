"""
Diagnostic: measure trajectory density + bounce F1 under different tracking
configs, to confirm the audit's root-cause hypothesis (static filter starves
the tracker -> bounce recall collapses).
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


def run(cfg, label, cache, gt_frames, fps, fh, fw, frame_diag):
    raw = cache["detections"]
    centers, speeds = track(raw, frame_diag, fps, cfg)
    raw_dens = density(centers)
    interp_gap = int(fps * 0.4)
    smoothed = smooth_ball_trajectory(centers, max_gap=interp_gap)
    sm_dens = density(smoothed)
    bounces = detect_bounces_velocity_signchange(smoothed, speeds, fps, fh, fw)
    pred_frames = [b[0] for b in bounces]
    ev = evaluate(pred_frames, gt_frames, fps)
    # second-half density (frames 700+)
    sh = sum(1 for c in centers[700:] if c is not None)
    print(f"\n=== {label} ===")
    print(f"  raw track density:   {raw_dens}/{len(centers)} ({100*raw_dens/len(centers):.0f}%)  "
          f"| 2nd-half(700+): {sh}/{len(centers)-700} ({100*sh/(len(centers)-700):.0f}%)")
    print(f"  after interp:        {sm_dens}/{len(smoothed)} ({100*sm_dens/len(smoothed):.0f}%)")
    print(f"  bounces predicted:   {ev['n_pred']}")
    print(f"  P={ev['precision']:.3f} R={ev['recall']:.3f} F1={ev['f1']:.3f}  "
          f"(TP={ev['tp']} FP={ev['fp']} FN={ev['fn']}, tol=±{ev['tol']})")
    if ev['fn_frames']:
        print(f"  misses: {ev['fn_frames']}")
    return ev


def main():
    cache = load_cache(CACHE)
    gt_frames, fps = load_gt(GT)
    fh, fw = cache["height"], cache["width"]
    frame_diag = float(np.hypot(fw, fh))

    # A: current committed config (static filter ON)
    cfg_current = {"static_enabled": True}
    run(cfg_current, "A. CURRENT (static filter ON)", cache, gt_frames, fps, fh, fw, frame_diag)

    # B: static filter OFF — the audit's cheap experiment
    cfg_nostatic = {"static_enabled": False}
    run(cfg_nostatic, "B. static filter OFF", cache, gt_frames, fps, fh, fw, frame_diag)


if __name__ == "__main__":
    main()
