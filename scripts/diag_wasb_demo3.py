"""WASB diagnostic: why does it miss 6/9 bounces on demo3 (broadcast)?

Measures the raw heatmap peak (post-sigmoid) at each frame, both on the full
frame and on a top-half crop, for the 6 missed GT bounce frames. Decides:
  - cause A (under-fire): hm_full < 0.5 but hm_crop > 0.5 → multi-scale saves it.
  - cause B (gate-drop):  hm_full > 0.5 → the tracker saw it; the drop is in
    _gate_trajectory, not the detector → fix the gate, not multi-scale.
  - blind: both < 0.3 → WASB can't see the ball; multi-scale won't help.

Run:  venv/bin/python scripts/diag_wasb_demo3.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import cv2

from utils.video_utils import VideoReader
from models.wasb_ball_tracker import HeatmapBallTracker, FRAMES_IN

VIDEO = ROOT / "data" / "output" / "tennis_demo3.mp4"
GT = ROOT / "data" / "output" / "tennis_demo3_bounces_gt.json"
WEIGHTS = ROOT / "training" / "wasb" / "wasb_tennis.pth.tar"
CROP_RATIO = 0.55
MISSED_GT = [62, 120, 174, 308, 369, 511]  # WASB finds 254(~255) + 456; f81 is a hit not a bounce


def heatmap_max_series(tracker, frames, batch_size=16):
    """Return list of (hm_full_max, hm_crop_max) per frame."""
    n = len(frames)
    crop_h = int(frames[0].shape[0] * CROP_RATIO)
    cropped = [f[:crop_h, :, :] for f in frames]
    out_full = _run(tracker, frames, batch_size)
    out_crop = _run(tracker, cropped, batch_size)
    series = []
    for i in range(n):
        hf = float(out_full[i].max())
        hc = float(out_crop[i].max())
        series.append((hf, hc))
    return series


def _run(tracker, frames, batch_size):
    """Run HRNet on 3-frame stacks, return the per-frame LAST-stack heatmap."""
    n = len(frames)
    heats = [None] * n
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            stacks = []
            for t in range(start, end):
                i0, i1 = max(0, t - 2), max(0, t - 1)
                stacks.append([frames[i0], frames[i1], frames[t]])
            tens = torch.cat([tracker._preprocess_stack(s) for s in stacks], dim=0).to(tracker.device)
            out = tracker.model(tens)[0]
            out = torch.sigmoid(out).cpu().numpy()
            for k, t in enumerate(range(start, end)):
                heats[t] = out[k, FRAMES_IN - 1]
    return heats


def main():
    gt = json.loads(GT.read_text())
    gt_frames = [b["demo_frame"] for b in gt["bounces_true"]]

    reader = VideoReader(str(VIDEO))
    fps = reader.fps
    frames = []
    for fr in reader.iter_frames():
        frames.append(fr)
    reader.release()
    print(f"loaded {len(frames)} frames ({fps} fps, {frames[0].shape[1]}x{frames[0].shape[0]})")

    tracker = HeatmapBallTracker(str(WEIGHTS), device="mps")
    print("running WASB (full + crop)...")
    series = heatmap_max_series(tracker, frames)

    print(f"\n{'frame':>6} {'hm_full':>9} {'hm_crop':>9}  verdict")
    print("-" * 50)
    a = b = blind = 0
    for f in MISSED_GT:
        hf, hc = series[f]
        if hf > 0.5:
            verdict = "B (gate-drop: tracker SAW it)"; b += 1
        elif hc > 0.5:
            verdict = "A (under-fire: crop SAVES)"; a += 1
        elif hf > 0.3 or hc > 0.3:
            verdict = "weak (borderline)";
            if hc > 0.5: a += 1
        else:
            verdict = "BLIND (both <0.3)"; blind += 1
        print(f"{f:>6} {hf:>9.3f} {hc:>9.3f}  {verdict}")

    print("-" * 50)
    print(f"\nSummary on {len(MISSED_GT)} missed bounces:")
    print(f"  cause A (under-fire, multi-scale saves): {a}")
    print(f"  cause B (gate-drop, fix _gate_trajectory): {b}")
    print(f"  blind (multi-scale won't help):           {blind}")
    if a >= 4:
        print("\n>>> DECISION: implement multi-scale (cause A confirmed for >=4/6).")
    elif b >= 3:
        print("\n>>> DECISION: fix _gate_trajectory instead (cause B dominates).")
    else:
        print("\n>>> DECISION: WASB is largely blind here — multi-scale won't help. Reconsider.")


if __name__ == "__main__":
    main()
