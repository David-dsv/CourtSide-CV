"""Density thermometer — measure ball-trajectory COVERAGE + candidate-pool RECALL
on the demo3 benchmark, the two upstream metrics the prompt names as the wall.

The methodology (vision/events.py) is FROZEN; the wall is the DENSITY/QUALITY of
the ball track feeding it. Two GT events (bounce 120, 369) sit in NO candidate
because the ball isn't tracked there. This script makes that measurable, per
source, BEFORE/AFTER any tracking change — re-run each iteration.

Metrics (all on demo3, 9 GT bounces + 8 GT hits = 17 events):
  1. Ball coverage      = % frames with a tracked ball (overall + windowed ±W
                          around the 17 GT events). Per source: Kalman / WASB.
  2. Candidate-pool      = of the 17 GT events, how many have >=1 candidate within
     recall              ±W. Candidates = turning-points ∪ robust-bounces ∪ hits.
                          (today capped at 15/17 — 120 & 369 missing.)
  3. Confusion + F1       = the run_demo3 methodology end-to-end (the final word).

Accepts an OPTIONAL alternate cache path so a candidate fix (e.g. a player-aware
re-track) can be scored against the committed baseline with one command:

  venv/bin/python tools/event_eval/measure_density.py                # baseline cache
  venv/bin/python tools/event_eval/measure_density.py path/to/alt.json
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vision.bounce import (  # noqa: E402
    smooth_ball_trajectory, detect_bounces_robust)
from vision.shots import detect_hits  # noqa: E402
from vision.events import detect_turning_points, detect_sharp_turns  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"

WIN = 8  # ±frames window around each GT event (matches the eval tolerance)


def load(cache_path):
    c = json.loads(Path(cache_path).read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    return fps, fw, fh, kal, wasb, ppf, gt_b, gt_h


def coverage(centers):
    n = len(centers)
    det = sum(1 for c in centers if c is not None)
    return det, n, (100.0 * det / n if n else 0.0)


def windowed_coverage(centers, events, win):
    """% frames tracked within ±win of any GT event."""
    n = len(centers)
    mask = np.zeros(n, dtype=bool)
    for f in events:
        lo, hi = max(0, f - win), min(n - 1, f + win)
        mask[lo:hi + 1] = True
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return 0, 0, 0.0
    det = sum(1 for i in idxs if centers[i] is not None)
    return det, len(idxs), 100.0 * det / len(idxs)


def per_event_coverage(centers, events, win):
    """For each GT event, the count of tracked frames in its ±win window."""
    n = len(centers)
    out = {}
    for f in events:
        lo, hi = max(0, f - win), min(n - 1, f + win)
        out[f] = sum(1 for i in range(lo, hi + 1) if centers[i] is not None)
    return out


def candidate_pool(centers, ppf, fps, fw, fh, raw_centers=None, is_real=None):
    """The RECALL union the methodology draws from: turning points ∪ sharp turns
    (far-court apex bounces) ∪ robust bounces ∪ wrist-speed hits. Turning points /
    robust bounces / hits read the spline-smoothed track; sharp turns read the RAW
    pre-spline track + is_real mask (passed in; defaults to `centers` non-None)."""
    sm = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(sm)
    b_cands, _re, _nd = detect_bounces_robust(sm, speeds, fps, fh, fw)
    turns = detect_turning_points(sm, fps, fh)
    hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[])
    rc = raw_centers if raw_centers is not None else centers
    ir = is_real if is_real is not None else [c is not None for c in rc]
    sharp = detect_sharp_turns(rc, ir, fps, fw, fh)
    pool = set(int(f) for (f, *_rest) in b_cands)
    pool |= set(int(f) for f in turns)
    pool |= set(int(f) for f in sharp)
    pool |= set(int(h["frame"]) for h in hits)
    return sorted(pool)


def pool_recall(pool, events, win):
    hit, miss = [], []
    for f in events:
        if any(abs(f - p) <= win for p in pool):
            hit.append(f)
        else:
            miss.append(f)
    return hit, miss


def report(cache_path, primary="kalman"):
    fps, fw, fh, kal, wasb, ppf, gt_b, gt_h = load(cache_path)
    c = json.loads(Path(cache_path).read_text())
    kal_real = c.get("kalman_is_real", [k is not None for k in kal])
    wasb_real = c.get("wasb_is_real", [w is not None for w in wasb])
    all_ev = sorted(gt_b + gt_h)
    print(f"\n=== DENSITY THERMOMETER  ({Path(cache_path).name}) ===")
    print(f"demo3: {len(kal)} frames @ {fps}fps {fw}x{fh} | "
          f"{len(gt_b)} GT bounces + {len(gt_h)} GT hits = {len(all_ev)} events | win=±{WIN}f\n")

    for name, centers in (("KALMAN", kal), ("WASB", wasb)):
        d, n, pct = coverage(centers)
        wd, wn, wpct = windowed_coverage(centers, all_ev, WIN)
        print(f"[{name}] coverage {d}/{n} ({pct:.0f}%) overall | "
              f"{wd}/{wn} ({wpct:.0f}%) within ±{WIN}f of events")

    print()
    src = kal if primary == "kalman" else wasb
    pec_b = per_event_coverage(src, gt_b, WIN)
    pec_h = per_event_coverage(src, gt_h, WIN)
    print(f"[{primary.upper()}] per-event tracked-frame count in ±{WIN}f window "
          f"(0 = ball not tracked at all there):")
    print("  bounces:", {f: pec_b[f] for f in gt_b})
    print("  hits:   ", {f: pec_h[f] for f in gt_h})

    print()
    src_real = kal_real if primary == "kalman" else wasb_real
    pool = candidate_pool(src, ppf, fps, fw, fh, raw_centers=src, is_real=src_real)
    hb, mb = pool_recall(pool, gt_b, WIN)
    hh, mh = pool_recall(pool, gt_h, WIN)
    total_hit = len(hb) + len(hh)
    print(f"[{primary.upper()}] CANDIDATE-POOL RECALL = {total_hit}/{len(all_ev)} events have >=1 candidate in ±{WIN}f")
    print(f"  bounce pool-recall: {len(hb)}/{len(gt_b)}  misses={mb}")
    print(f"  hit    pool-recall: {len(hh)}/{len(gt_h)}  misses={mh}")
    print(f"  pool size: {len(pool)} candidates")
    return {
        "fps": fps, "pool_recall": total_hit, "n_events": len(all_ev),
        "bounce_pool_miss": mb, "hit_pool_miss": mh,
        "kal_cov": coverage(kal)[2], "wasb_cov": coverage(wasb)[2],
        "kal_win_cov": windowed_coverage(kal, all_ev, WIN)[2],
        "wasb_win_cov": windowed_coverage(wasb, all_ev, WIN)[2],
        "per_event_bounce": pec_b, "per_event_hit": pec_h,
    }


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else str(CACHE)
    primary = sys.argv[2] if len(sys.argv) > 2 else "kalman"
    report(cache, primary)
