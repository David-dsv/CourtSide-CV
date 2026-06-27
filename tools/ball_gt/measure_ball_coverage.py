"""Measure ball-TRACK coverage + accuracy against the human ball GT.
The honest thermometer for ball tracking (esp. cold-start at the clip start).

  python tools/ball_gt/measure_ball_coverage.py                 # default kalman
  python tools/ball_gt/measure_ball_coverage.py --ball-tracker wasb
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.video_utils import VideoReader

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "ball_gt" / "tennis_demo3.ball_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ball-tracker", default="kalman", choices=["kalman", "wasb"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol-frac", type=float, default=0.02,
                    help="a tracked ball is CORRECT within tol_frac*W of the GT point")
    args = ap.parse_args()

    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["ball"] for f in gt["frames"]
            if f.get("verified") and f.get("visible", True) and f.get("ball")}
    print(f"GT: {len(gtbf)} frames with a verified VISIBLE ball point", flush=True)

    rdr = VideoReader(str(CLIP))
    fw, fh, fps = rdr.width, rdr.height, rdr.fps
    total = rdr.frame_count
    frames = list(rdr.iter_frames())
    rdr.release()
    tol = args.tol_frac * fw

    # Build the pipeline's ball track the same way Pass 1 does.
    from models.yolo_detector import TennisYOLODetector
    det = TennisYOLODetector(device=args.device)
    track = [None] * total
    if args.ball_tracker == "wasb":
        from models.wasb_ball_tracker import HeatmapBallTracker
        tr = HeatmapBallTracker(device=args.device)
        centers = tr.track_ball(frames)
        for i, c in enumerate(centers):
            if i < total:
                track[i] = c
    else:
        # raw YOLO26 ball detection (highest-score), the candidate the Kalman uses
        for i, frame in enumerate(frames):
            d = det.detect(frame, classes=[32])
            m = d["class_ids"] == 32
            if m.any():
                b = d["boxes"][m]
                s = d["scores"][m]
                bi = int(s.argmax())
                track[i] = ((float(b[bi][0]) + float(b[bi][2])) / 2.0,
                            (float(b[bi][1]) + float(b[bi][3])) / 2.0)

    # score: coverage (any ball) + correct (within tol of GT) — overall and START
    start_n = int(4.0 * fps)  # cold-start window (first 4s)
    cov = corr = n = 0
    cov_s = corr_s = n_s = 0
    misses_start = []
    for fidx, g in gtbf.items():
        if fidx >= total:
            continue
        n += 1
        is_start = fidx < start_n
        if is_start:
            n_s += 1
        t = track[fidx]
        if t is not None:
            cov += 1
            if is_start:
                cov_s += 1
            d = ((t[0] - g[0]) ** 2 + (t[1] - g[1]) ** 2) ** 0.5
            if d < tol:
                corr += 1
                if is_start:
                    corr_s += 1
            elif is_start:
                misses_start.append((fidx, "wrong-pos", round(d)))
        elif is_start:
            misses_start.append((fidx, "no-ball", None))

    print(f"\n=== BALL TRACK ({args.ball_tracker}) vs human GT — clip demo3 ===", flush=True)
    print(f"coverage (any ball): {cov}/{n} = {100*cov/max(1,n):.1f}%", flush=True)
    print(f"CORRECT (within {tol:.0f}px of GT): {corr}/{n} = {100*corr/max(1,n):.1f}%", flush=True)
    print(f"--- COLD START (first 4s, {n_s} GT frames) ---", flush=True)
    print(f"  coverage: {cov_s}/{n_s} = {100*cov_s/max(1,n_s):.1f}%", flush=True)
    print(f"  CORRECT : {corr_s}/{n_s} = {100*corr_s/max(1,n_s):.1f}%", flush=True)
    if misses_start:
        print(f"  start misses: {misses_start[:15]}", flush=True)


if __name__ == "__main__":
    main()
