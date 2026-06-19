"""
WASB-path bounce regression test (fast, no GPU / no video / no model).

Runs the PRODUCTION robust bounce detector (vision.bounce.detect_bounces_robust)
on a committed fixture: the WASB heatmap trajectory for felix.mp4 (precomputed)
+ the bounce ground truth. Asserts the bounce F1 on the WASB path does not
regress. This guards the --ball-tracker wasb path.

Run:  venv/bin/python tests/test_bounce_wasb_regression.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.bounce import detect_bounces_robust  # noqa: E402

WASB = ROOT / "tests" / "fixtures" / "cache" / "felix_wasb_centers.json"
GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"

# WASB + robust detector measured at F1 0.833 end-to-end / 0.865 on the saved
# trajectory; floor set with margin.
F1_FLOOR = 0.80


def _match(pred, truth, tol):
    cand = sorted((abs(p - t), p, t) for p in pred for t in truth if abs(p - t) <= tol)
    up, ut, m = set(), set(), 0
    for _, p, t in cand:
        if p in up or t in ut:
            continue
        up.add(p)
        ut.add(t)
        m += 1
    return m, len(pred) - m, len(truth) - m


def compute_f1():
    w = json.loads(WASB.read_text())
    gt = json.loads(GT.read_text())
    fps = w["fps"]
    fw, fh = 1920, 1080
    n = w["n_frames"]
    start = w.get("start_frame", 0)
    traj = [None] * n
    for e in w["centers"]:
        f = e["frame"] - start
        if 0 <= f < n and e.get("x") is not None:
            traj[f] = (e["x"], e["y"])

    gt_frames = [b["frame"] for b in gt["bounces"]]
    res = detect_bounces_robust(traj, [0.0] * n, fps, fh, fw)
    chosen = res[0] if isinstance(res, tuple) else res
    pred = [b[0] for b in chosen]

    tol = round(0.15 * fps)
    tp, fp, fn = _match(pred, gt_frames, tol)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r, tp, fp, fn


def test_wasb_bounce_f1_no_regression():
    f1, p, r, tp, fp, fn = compute_f1()
    assert f1 >= F1_FLOOR, (
        f"WASB bounce F1 regressed: {f1:.3f} < floor {F1_FLOOR} "
        f"(P={p:.3f} R={r:.3f} TP={tp} FP={fp} FN={fn})")


if __name__ == "__main__":
    f1, p, r, tp, fp, fn = compute_f1()
    print(f"WASB bounce regression (felix): F1={f1:.3f} P={p:.3f} R={r:.3f} "
          f"(TP={tp} FP={fp} FN={fn}); floor={F1_FLOOR}")
    print("PASS" if f1 >= F1_FLOOR else "FAIL")
    sys.exit(0 if f1 >= F1_FLOOR else 1)
