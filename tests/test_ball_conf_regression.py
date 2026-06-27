"""Guardrail for the SHIPPED --ball-conf default (fast, no GPU / no video).

test_bounce_regression replays a frozen felix cache at conf 0.15, so it is BLIND
to the actual --ball-conf default the pipeline ships. This test closes that hole:
it reads the live argparse default for --ball-conf and scores felix bounce F1 at
THAT conf, on a committed felix detection cache (floor 0.03 → any conf >= 0.03 is
replayable). It guards the FELIX BOUNCE-F1 FLOOR only (not demo3 ball CORRECT):
on the committed cache it trips at conf 0.05/0.08/0.11/0.12/0.13 (all <0.72) and
passes at 0.14/0.16/0.18. (Note 0.10 happens to score 0.774 here, so this test
does NOT by itself reject 0.10 — 0.10 was rejected for the SEPARATE reason that
it dips demo3 ball CORRECT; see docs/research/ball-track-density-CR.md.)

Run:  venv/bin/python tests/test_ball_conf_regression.py
"""
import sys
import json
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.bounce import detect_bounces_from_trajectory, smooth_ball_trajectory  # noqa: E402
# reuse the prod-faithful Kalman tracker + matcher
_spec = importlib.util.spec_from_file_location(
    "_tbr", str(ROOT / "tests" / "test_bounce_regression.py"))
_tbr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tbr)
_track, _match = _tbr._track, _tbr._match

CACHE = ROOT / "tests" / "fixtures" / "ball_gt" / "felix_det_cache_1280.json"
GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"
F1_FLOOR = 0.72  # same floor as test_bounce_regression


def shipped_ball_conf():
    """Read the LIVE --ball-conf default straight from run_pipeline_8s.py's parser
    so this test can never drift from what the pipeline actually ships."""
    src = (ROOT / "run_pipeline_8s.py").read_text()
    # the arg is declared as: parser.add_argument("--ball-conf", type=float, default=X,
    import re
    m = re.search(r'--ball-conf"[^)]*?default=([0-9.]+)', src, re.S)
    if not m:
        raise AssertionError("could not find --ball-conf default in run_pipeline_8s.py")
    return float(m.group(1))


def felix_f1_at(conf):
    cache = json.loads(CACHE.read_text())
    fw, fh, fps = cache["width"], cache["height"], cache["fps"]
    raw = [[(int(c["cx"]), int(c["cy"]), c["score"])
            for c in fr if c["score"] >= conf] for fr in cache["frames"]]
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
    return f1, p, r, tp, fp, fn


def test_shipped_ball_conf_no_felix_regression():
    conf = shipped_ball_conf()
    f1, p, r, tp, fp, fn = felix_f1_at(conf)
    assert f1 >= F1_FLOOR, (
        f"Shipped --ball-conf={conf} REGRESSES felix bounce F1 to {f1:.3f} "
        f"< floor {F1_FLOOR} (P={p:.3f} R={r:.3f} TP={tp} FP={fp} FN={fn}). "
        f"felix bounce F1 is noise-limited (16-bounce GT); conf <=0.08 and the "
        f"0.11/0.12/0.13 region drop below the floor — keep the default at/above "
        f"0.14 (0.16 is the shipped value, with margin from the 0.13 cliff).")


if __name__ == "__main__":
    conf = shipped_ball_conf()
    f1, p, r, tp, fp, fn = felix_f1_at(conf)
    print(f"shipped --ball-conf={conf}: felix bounce F1={f1:.3f} "
          f"P={p:.3f} R={r:.3f} (TP={tp} FP={fp} FN={fn}); floor={F1_FLOOR}")
    print("PASS" if f1 >= F1_FLOOR else "FAIL")
    sys.exit(0 if f1 >= F1_FLOOR else 1)
