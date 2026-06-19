"""
For each missed GT bounce, inspect the smoothed trajectory to see whether the
ball is tracked there and what the local y-shape looks like — to distinguish
'tracking hole' from 'detector threshold too strict'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.tracking_lab import load_cache, load_gt, track, smooth_ball_trajectory  # noqa: E402

CACHE = "experiments/cache/felix_dets.json"
GT = "tests/fixtures/bounces/felix.bounces.json"


def main():
    cache = load_cache(CACHE)
    gt_frames, fps = load_gt(GT)
    fh, fw = cache["height"], cache["width"]
    frame_diag = float(np.hypot(fw, fh))
    raw = cache["detections"]

    cfg = {"reinit_gate_ratio": 0.25, "kf_max_coast_s": 0.17}
    centers, _ = track(raw, frame_diag, fps, cfg)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))

    misses = [424, 547, 978, 1142, 1253, 1346, 1519, 1667, 1795]
    for g in misses:
        w = 9
        seg = []
        present = 0
        for f in range(g - w, g + w + 1):
            c = smoothed[f] if 0 <= f < len(smoothed) else None
            if c is None:
                seg.append("  .  ")
            else:
                seg.append(f"{c[1]:>4d}")
                present += 1
        print(f"GT {g:4d} (present {present}/{2*w+1}): y = {' '.join(seg)}")


if __name__ == "__main__":
    main()
