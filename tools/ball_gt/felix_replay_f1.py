"""Replay a felix det cache (built at a LOW floor) at any conf and score bounce F1
vs the felix GT — INSTANTLY. The cross-clip anti-overfit thermometer for the ball
conf, scored on the SAME prod tracking + bounce algorithm as test_bounce_regression.

  python tools/ball_gt/build_det_cache.py --clip felix.mp4 --imgsz 1280 --conf 0.03 \
      --max-frames 1850 --out /tmp/felix_det_cache_1280.json
  python tools/ball_gt/felix_replay_f1.py --cache /tmp/felix_det_cache_1280.json --sweep
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from vision.bounce import detect_bounces_from_trajectory, smooth_ball_trajectory  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("_tbr", str(ROOT / "tests" / "test_bounce_regression.py"))
_tbr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tbr)
_track, _match = _tbr._track, _tbr._match

GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"


def f1_at(cache, conf):
    fw, fh, fps = cache["width"], cache["height"], cache["fps"]
    # cache frames → prod raw-candidate format (cx, cy, score) at this conf
    raw = [[(int(c["cx"]), int(c["cy"]), c["score"])
            for c in fr if c["score"] >= conf] for fr in cache["frames"]]
    ncand = sum(len(f) for f in raw)
    centers = _track(raw, fw, fh, fps)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
    is_real = [c is not None for c in centers]
    bounces = detect_bounces_from_trajectory(
        smoothed, [0.0] * len(smoothed), fps, fh, fw,
        raw_centers=centers, is_real=is_real)
    pred = [b[0] for b in bounces]
    gt = json.loads(GT.read_text())
    gt_frames = [b["frame"] for b in gt["bounces"]]
    tol = round(0.15 * fps)
    tp, fp, fn = _match(pred, gt_frames, tol)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    cov = sum(1 for c in centers if c is not None)
    return dict(f1=f1, p=p, r=r, tp=tp, fp=fp, fn=fn, ncand=ncand,
                cov=cov, n=len(centers))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    print(f"felix cache {Path(args.cache).name}: imgsz={cache['imgsz']} "
          f"floor={cache['conf_floor']} {cache['n_frames']} frames", flush=True)
    confs = ([0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20] if args.sweep
             else [args.conf])
    for conf in confs:
        r = f1_at(cache, conf)
        flag = "" if r["f1"] >= 0.72 else "  <-- BELOW FLOOR 0.72"
        print(f"  conf={conf:.2f}  F1={r['f1']:.3f} P={r['p']:.3f} R={r['r']:.3f} "
              f"(TP={r['tp']} FP={r['fp']} FN={r['fn']}) cov={r['cov']}/{r['n']}="
              f"{100*r['cov']/r['n']:.1f}% ncand={r['ncand']}{flag}", flush=True)


if __name__ == "__main__":
    main()
