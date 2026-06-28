"""
Court-homography evaluation — the "thermometer" for auto-court detection.

A PURE comparator (mirrors tools/bounce_eval/eval_bounces.py): it reads a
candidate homography dict + a hand-clicked ground-truth court file (same shared
format as calibrations/<name>.court.json, see tests/fixtures/court/) and reports
how accurate the homography is, IN METERS — the unit speed/depth actually depend
on. No video decode, no model inference, no GPU; runs in milliseconds.

It is the metric AND the gate-calibration oracle for models/court_auto.py:
- (a) corner reprojection error (px / %diag)
- (b) METRIC error — the PRIMARY number: project the GT clicks through H and
      compare KNOWN ITF inter-landmark distances (angle-invariant ground truth)
- (c) far/near split — grazing failures hide in one far corner; report the ratio
- a verdict PASS / MARGINAL / FAIL per the R&D thresholds (docs §8.2)

World frame (matches models/court_calibrator.WORLD_LANDMARKS): origin at court
center, +X toward the right doubles sideline, +Y toward the far baseline, meters.

ITF facts (documented geometric constants, NOT tuned to any video):
    doubles width   = 10.97 m   (sidelines 5.485 m either side of center)
    singles width   =  8.23 m
    court length    = 23.77 m   (baseline-to-baseline; ±11.885 m from net)
    service line    =  6.40 m   from the net

Usage
-----
    python tools/court_eval/eval_homography.py \\
        --truth tests/fixtures/court/felix.court.json \\
        --calibration               # score the committed sidecar homography
    # or score an explicit candidate dict dumped to JSON:
    python tools/court_eval/eval_homography.py --truth ... --pred cand.json
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.court_calibrator import (  # noqa: E402
    WORLD_LANDMARKS, homography_from_points,
)

# ── R&D §8.2 verdict thresholds (targets, not guarantees — see doc) ──
CORNER_RMS_PASS_PCT = 0.5      # <=0.5% diag corner reproj
CORNER_RMS_MARGINAL_PCT = 1.2
METRIC_PASS_MEAN_PCT = 3.0     # <=3% mean metric err (≈ ±5 km/h on a 160 km/h serve)
METRIC_PASS_MAX_PCT = 5.0      # AND <=5% max


# Known ITF inter-landmark distances (meters), keyed by the landmark pair.
# These are angle-INVARIANT ground truth: the camera angle cannot change the real
# distance between two court points, so projecting the GT clicks through H and
# comparing these is a clean metric check decoupled from the solve.
#   ⚠ BUG FIXED (R&D critique C1): net_*->far_svc_center is hypot(5.485,6.40)=8.43m,
#   NOT 6.40m (6.40 only holds from net_center (0,0), not a clicked landmark). We
#   use svc_to_svc=12.80m and the baselines/sidelines, all between real landmarks.
def _itf_dist(a: str, b: str) -> float:
    pa, pb = WORLD_LANDMARKS[a], WORLD_LANDMARKS[b]
    return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))


# Candidate distance checks, only used when BOTH landmarks were clicked in GT.
METRIC_PAIRS = [
    ("far_bl_left", "far_bl_right", "baseline_far"),       # 10.97 m
    ("near_bl_left", "near_bl_right", "baseline_near"),     # 10.97 m
    ("far_bl_left", "near_bl_left", "sideline_left"),       # 23.77 m
    ("far_bl_right", "near_bl_right", "sideline_right"),    # 23.77 m
    ("far_svc_center", "near_svc_center", "svc_to_svc"),    # 12.80 m
    ("far_sl_left", "far_sl_right", "singles_far"),         # 8.23 m
    ("near_sl_left", "near_sl_right", "singles_near"),      # 8.23 m
    ("net_left", "net_right", "net_width"),                 # 10.97 m
]


def _apply_H(H, x, y):
    p = np.asarray(H, dtype=np.float64) @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def _load_truth(path: str):
    data = json.loads(Path(path).read_text())
    pts = {k: (float(v[0]), float(v[1])) for k, v in data["points"].items()}
    wh = data.get("frame_wh") or [1920, 1080]
    diag = float(np.hypot(wh[0], wh[1]))
    return data, pts, wh, diag


def _resolve_candidate(args, truth_pts):
    """Return a homography dict {H, H_inv, confidence, ...} for the candidate."""
    if args.pred:
        d = json.loads(Path(args.pred).read_text())
        d["H"] = np.asarray(d["H"], dtype=np.float64)
        d["H_inv"] = np.asarray(d.get("H_inv") or np.linalg.inv(d["H"]),
                                dtype=np.float64)
        return d
    if args.calibration:
        # Score the committed sidecar/calibration homography for this clip.
        return homography_from_points(truth_pts)
    raise SystemExit("Provide --pred <cand.json> or --calibration")


def evaluate(truth_path: str, cand: dict, *, loo: bool = True) -> dict:
    """Score a candidate homography dict against the GT court file. Returns a
    report dict with corner/metric/far-near sub-metrics and a verdict."""
    data, gt_pts, wh, diag = _load_truth(truth_path)
    H = np.asarray(cand["H"], dtype=np.float64)
    H_inv = np.asarray(cand.get("H_inv") if cand.get("H_inv") is not None
                       else np.linalg.inv(H), dtype=np.float64)

    rep: dict = {
        "clip": data.get("video", Path(truth_path).stem),
        "difficulty": data.get("difficulty", "unknown"),
        "frame_wh": wh, "diag_px": diag,
        "confidence": cand.get("confidence", "unknown"),
        "source": cand.get("source", "unknown"),
        "n_gt_points": len(gt_pts),
    }

    # ── (a) corner reprojection: world corner -> image, vs GT click ──
    # Honest protocol (§8.2e): a 4-pt fit is EXACT at its control points, so
    # scoring the same points you solved gives a meaningless rms≈0. If >=5 GT
    # points, do leave-one-out (solve all-but-one, score the held-out point).
    corner_errs, corner_split = [], {"near": [], "far": []}
    if loo and len(gt_pts) >= 5:
        names = list(gt_pts)
        for held in names:
            train = {k: v for k, v in gt_pts.items() if k != held}
            try:
                hh = homography_from_points(train)
            except Exception:
                continue
            px, py = _apply_H(hh["H_inv"], *WORLD_LANDMARKS[held])
            e = float(np.hypot(px - gt_pts[held][0], py - gt_pts[held][1]))
            corner_errs.append(e)
            (corner_split["far"] if WORLD_LANDMARKS[held][1] >= 0
             else corner_split["near"]).append(e)
        rep["corner_protocol"] = "leave-one-out"
    else:
        # exactly-4 case: corner-rms is non-informative (rms≈0). Report it but
        # the metric check (b) carries the verdict.
        for name, (gx, gy) in gt_pts.items():
            px, py = _apply_H(H_inv, *WORLD_LANDMARKS[name])
            e = float(np.hypot(px - gx, py - gy))
            corner_errs.append(e)
            (corner_split["far"] if WORLD_LANDMARKS[name][1] >= 0
             else corner_split["near"]).append(e)
        rep["corner_protocol"] = "control-points (non-informative for n=4)"

    if corner_errs:
        rep["corner_rms_px"] = float(np.sqrt(np.mean(np.square(corner_errs))))
        rep["corner_max_px"] = float(np.max(corner_errs))
        rep["corner_rms_pct"] = rep["corner_rms_px"] / diag * 100.0
        rep["corner_max_pct"] = rep["corner_max_px"] / diag * 100.0
        rep["far_near_ratio"] = (
            float(np.mean(corner_split["far"]) / (np.mean(corner_split["near"]) + 1e-9))
            if corner_split["near"] and corner_split["far"] else None)

    # ── (b) METRIC error — the PRIMARY number ──
    # Project GT clicks -> meters via H, compare known ITF inter-landmark dists.
    metric_rows, metric_errs = [], []
    for a, b, label in METRIC_PAIRS:
        if a in gt_pts and b in gt_pts:
            ax, ay = _apply_H(H, *gt_pts[a])
            bx, by = _apply_H(H, *gt_pts[b])
            measured = float(np.hypot(ax - bx, ay - by))
            truth = _itf_dist(a, b)
            err_pct = abs(measured - truth) / truth * 100.0
            metric_rows.append({"pair": label, "measured_m": round(measured, 3),
                                "truth_m": round(truth, 3), "err_pct": round(err_pct, 2)})
            metric_errs.append(err_pct)
    rep["metric_checks"] = metric_rows
    if metric_errs:
        rep["metric_err_mean_pct"] = float(np.mean(metric_errs))
        rep["metric_err_max_pct"] = float(np.max(metric_errs))
    else:
        rep["metric_err_mean_pct"] = None
        rep["metric_err_max_pct"] = None

    # ── verdict (metric is primary; corner is corroborating when LOO) ──
    rep["verdict"] = _verdict(rep)
    return rep


def _verdict(rep: dict) -> str:
    m_mean = rep.get("metric_err_mean_pct")
    m_max = rep.get("metric_err_max_pct")
    if m_mean is None:
        return "NO_METRIC_CHECK"
    if m_mean <= METRIC_PASS_MEAN_PCT and (m_max or 0) <= METRIC_PASS_MAX_PCT:
        return "PASS"
    if m_mean <= 2 * METRIC_PASS_MEAN_PCT:
        return "MARGINAL"
    return "FAIL"


def _print(rep: dict):
    print(f"\n=== court homography eval: {rep['clip']} "
          f"({rep['difficulty']}, {rep['source']}, conf={rep['confidence']}) ===")
    print(f"  GT points: {rep['n_gt_points']}  frame {rep['frame_wh']}  "
          f"diag {rep['diag_px']:.0f}px")
    if "corner_rms_px" in rep:
        print(f"  [a] corner ({rep['corner_protocol']}): "
              f"rms={rep['corner_rms_px']:.1f}px ({rep['corner_rms_pct']:.2f}%), "
              f"max={rep['corner_max_px']:.1f}px ({rep['corner_max_pct']:.2f}%)"
              + (f", far/near={rep['far_near_ratio']:.2f}"
                 if rep.get("far_near_ratio") is not None else ""))
    if rep.get("metric_err_mean_pct") is not None:
        print(f"  [b] METRIC: mean={rep['metric_err_mean_pct']:.2f}%  "
              f"max={rep['metric_err_max_pct']:.2f}%  (PASS <= {METRIC_PASS_MEAN_PCT}% mean)")
        for r in rep["metric_checks"]:
            print(f"        {r['pair']:>14}: {r['measured_m']:.2f}m vs "
                  f"{r['truth_m']:.2f}m  ({r['err_pct']:+.1f}%)")
    print(f"  VERDICT: {rep['verdict']}")


def main():
    ap = argparse.ArgumentParser(description="Score a court homography vs GT (meters).")
    ap.add_argument("--truth", required=True, help="GT court file (tests/fixtures/court/*.court.json)")
    ap.add_argument("--pred", help="Candidate homography dumped to JSON {H, H_inv, confidence, source}")
    ap.add_argument("--calibration", action="store_true",
                    help="Score the homography solved from the GT clicks themselves "
                         "(the committed sidecar) — a sanity/positive-control baseline.")
    ap.add_argument("--no-loo", action="store_true", help="Disable leave-one-out corner scoring")
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = ap.parse_args()

    _, truth_pts, _, _ = _load_truth(args.truth)
    cand = _resolve_candidate(args, truth_pts)
    rep = evaluate(args.truth, cand, loo=not args.no_loo)
    if args.json:
        print(json.dumps(rep, indent=2, default=float))
    else:
        _print(rep)


if __name__ == "__main__":
    main()
