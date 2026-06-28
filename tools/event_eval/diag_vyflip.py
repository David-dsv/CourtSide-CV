"""S3 diag — find which vy_flip mandatory-anchor frames are PHANTOMS (driven by a
single jitter outlier) vs REAL bounces (a consistent vertical reflection).

For every merged-candidate cluster the classifier inspects, recompute the
pre/post vertical points and report:
  - the current (non-robust) vy_pre / vy_post slope (lstsq),
  - a despiked (median-window) vy_pre / vy_post slope,
  - whether vy_flip flips between the two (== a jitter-faked flip),
  - the max single-frame |dy| jump in each side (the outlier magnitude).

Run from the worktree with the main-repo venv python:
  /Users/vuong/Documents/CourtSide-CV/venv/bin/python tools/event_eval/diag_vyflip.py \
      tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import (  # noqa: E402
    smooth_ball_trajectory, detect_bounces_robust,
    _veto_collect_side, _veto_max_gap, _veto_drop_teleports)
from vision.events import detect_turning_points, detect_sharp_turns  # noqa: E402

GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]


def _lstsq(pts, axis):
    """The OLD non-robust slope, defined LOCALLY so the 'lstsq' column shows the
    true old-vs-new contrast even after vision.bounce._veto_slope became Theil-Sen
    (importing _veto_slope would make the 'lstsq' column == the 'theilsen' one)."""
    if len(pts) < 2:
        return None
    ks = np.array([p[0] for p in pts], float)
    vs = np.array([p[axis] for p in pts], float)
    A = np.vstack([ks, np.ones(len(ks))]).T
    (a, _b), *_ = np.linalg.lstsq(A, vs, rcond=None)
    return float(a)


def _despike_pts(pts, axis, k=1):
    """Replace each interior value by the median of a ±k window (frame-ordered).
    Outlier-robust pre-fit cleaning, then slope = plain lstsq on the cleaned vals."""
    if len(pts) < 3:
        return pts
    vals = [p[axis] for p in pts]
    out = list(pts)
    for i in range(len(pts)):
        lo, hi = max(0, i - k), min(len(pts), i + k + 1)
        med = float(np.median(vals[lo:hi]))
        out[i] = (pts[i][0], pts[i][1] if axis != 1 else med,
                  pts[i][2] if axis != 2 else med)
    return out


def _theilsen(pts, axis):
    """Median of pairwise slopes (frame-spaced). Fully outlier-robust."""
    if len(pts) < 2:
        return None
    ks = [p[0] for p in pts]
    vs = [p[axis] for p in pts]
    slopes = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dk = ks[j] - ks[i]
            if dk != 0:
                slopes.append((vs[j] - vs[i]) / dk)
    return float(np.median(slopes)) if slopes else None


def main(path):
    d = json.loads(Path(path).read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    is_real = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    hits = d["hits"]

    half = max(2, round(fps * 0.10))
    gap_thr = round(fps * 0.05)

    b_cands, _re, _nd = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = sorted(set(detect_turning_points(ac, fps, fh)) |
                   set(detect_sharp_turns(raw, is_real, fps, fw, fh)))
    hit_fr = [int(h["frame"]) for h in hits]
    allcand = sorted(set([int(f) for f, *_ in b_cands]) | set(int(t) for t in turns)
                     | set(hit_fr))

    print(f"\n=== vy_flip PHANTOM diag ({Path(path).name}) ===")
    print(f"{len(raw)} frames @ {fps}fps {fw}x{fh}; half={half} gap_thr={gap_thr}")
    print(f"candidates inspected: {len(allcand)}")
    print("\n  frame  near-GT  | lstsq vyflip | despike vyflip | theilsen vyflip "
          "| max|dy|pre max|dy|post | PHANTOM?")
    phantoms = []
    for f in allcand:
        if (_veto_max_gap(is_real, f - half, f) > gap_thr
                or _veto_max_gap(is_real, f, f + half) > gap_thr):
            continue
        pre, pre_tp = _veto_drop_teleports(
            _veto_collect_side(raw, is_real, f - half, f), fw)
        post, post_tp = _veto_drop_teleports(
            _veto_collect_side(raw, is_real, f, f + half), fw)
        if pre_tp or post_tp or len(pre) < 2 or len(post) < 2:
            continue

        def vyflip(pp, qq, slope):
            a = slope(pp, 2)
            b = slope(qq, 2)
            return (a is not None and b is not None and a > 0 and b < 0), a, b

        ls_flip, ls_a, ls_b = vyflip(pre, post, _lstsq)            # OLD non-robust
        ds_flip, ds_a, ds_b = vyflip(_despike_pts(pre, 2), _despike_pts(post, 2),
                                     _lstsq)
        ts_flip, ts_a, ts_b = vyflip(pre, post, _theilsen)         # NEW robust

        # max single-frame dy jump each side (outlier magnitude)
        def maxdy(pp):
            return max((abs(pp[i][2] - pp[i - 1][2])
                        for i in range(1, len(pp))
                        if pp[i][0] - pp[i - 1][0] == 1), default=0.0)
        mdp, mdq = maxdy(pre), maxdy(post)

        if not (ls_flip or ds_flip or ts_flip):
            continue
        ng = min([(abs(f - g), g, lb) for g, lb in
                  [(b, 'B') for b in GT_B] + [(h, 'H') for h in GT_H]])
        near_gt = f"{ng[2]}{ng[1]}@{ng[0]}" if ng[0] <= 8 else f"({ng[2]}{ng[1]}@{ng[0]})"
        phantom = ls_flip and not (ds_flip and ts_flip)
        if phantom:
            phantoms.append(f)
        print(f"  {f:>5}  {near_gt:<8} | {str(ls_flip):<5} {ls_a:+6.2f}/{ls_b:+6.2f} "
              f"| {str(ds_flip):<5} {ds_a:+6.2f}/{ds_b:+6.2f} "
              f"| {str(ts_flip):<5} {ts_a:+6.2f}/{ts_b:+6.2f} "
              f"| {mdp:5.1f} {mdq:5.1f} | {'<<< PHANTOM' if phantom else ''}")
    print(f"\nPHANTOM vy-flips (lstsq=True, robust=False): {phantoms}")


if __name__ == "__main__":
    main(sys.argv[1])
