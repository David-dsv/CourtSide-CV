"""
Bounce-detection regression test (fast, no GPU / no video decode).

Runs the PRODUCTION bounce detector (vision.bounce) on a committed fixture: the
cached raw YOLO ball detections for felix.mp4 + the hand-annotated bounce ground
truth. Tracks the ball with the same Kalman config as the pipeline, then asserts
the bounce F1 against GT does not regress below a floor.

This is the guardrail for the accuracy work: any change that drops bounce F1 on
felix fails CI. Run:  venv/bin/python -m pytest tests/test_bounce_regression.py -q
or standalone:        venv/bin/python tests/test_bounce_regression.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

from vision.bounce import detect_bounces_from_trajectory, smooth_ball_trajectory  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "felix_dets.json"
GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"

# Floor below which the test fails. Current measured (cache replay): F1 ~0.80.
# Set with margin so noise doesn't flake the test, but a real regression trips it.
F1_FLOOR = 0.72


# ── minimal port of the pipeline Kalman tracking (production config) ──
class _KF:
    def __init__(self, pn=200.0, mn=8.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * pn
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * mn
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
        self.initialized = False
        self.fsu = 0

    def init(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.initialized = True
        self.fsu = 0

    def predict(self):
        if not self.initialized:
            return None
        p = self.kf.predict()
        self.fsu += 1
        return float(p[0][0]), float(p[1][0])

    def update(self, x, y):
        self.kf.correct(np.array([[x], [y]], np.float32))
        self.fsu = 0

    def gate(self, x, y):
        p = self.kf.statePost
        return float(np.hypot(x - p[0][0], y - p[1][0]))


def _track(raw, fw, fh, fps):
    """Production tracking config (matches run_pipeline_8s.py)."""
    from collections import Counter
    diag = float(np.hypot(fw, fh))
    # static filter (same params as pipeline)
    pts = [(cx // 20, cy // 20) for cands in raw for cx, cy, _ in cands]
    cnt = Counter(pts)
    thr = max(10, int(len(raw) * 0.08))
    static = {k for k, v in cnt.items() if v >= thr}

    def is_static(cx, cy):
        gx, gy = cx // 20, cy // 20
        return any((gx + dx, gy + dy) in static for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    kf = _KF()
    base, growth = 150.0, 80.0
    gmax = diag * 0.35
    coast = int(fps * 0.12)
    reinit = diag * 0.50
    last = None
    centers = []
    for cands in raw:
        moving = [(cx, cy, s) for cx, cy, s in cands if not is_static(cx, cy)]
        c = None
        pred = kf.predict()
        gate = min(gmax, base + growth * kf.fsu)
        if moving:
            if not kf.initialized:
                b = max(moving, key=lambda z: z[2])
                kf.init(b[0], b[1])
                c = (b[0], b[1])
            else:
                bd, bdist = None, float("inf")
                for cx, cy, s in moving:
                    d = kf.gate(cx, cy)
                    if d < gate and d < bdist:
                        bdist, bd = d, (cx, cy)
                if bd is not None:
                    kf.update(*bd)
                    st = kf.kf.statePost
                    c = (int(st[0][0]), int(st[1][0]))
                elif kf.fsu <= coast and pred:
                    c = (int(pred[0]), int(pred[1]))
                elif last is not None:
                    nb, nd = None, float("inf")
                    for cx, cy, s in moving:
                        d = np.hypot(cx - last[0], cy - last[1])
                        if d < reinit and d < nd:
                            nd, nb = d, (cx, cy)
                    if nb is not None:
                        kf.init(*nb)
                        c = nb
        if c is None and kf.initialized and kf.fsu <= coast and pred:
            c = (int(pred[0]), int(pred[1]))
        if c is not None:
            last = c
        centers.append(c)
    return centers


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


def compute_f1(thread_raw=False):
    """thread_raw=True replicates the LEGACY PROD call (run_pipeline_8s.py threads
    raw_centers + is_real so the turn-angle apex pass runs). thread_raw=False is the
    original no-arg call. Both must clear the floor — felix is the cross-clip
    guardrail for the turn-angle bounce-recovery branch (feat/legacy-bounce-shot-sep)."""
    cache = json.loads(CACHE.read_text())
    gt = json.loads(GT.read_text())
    fw, fh, fps = cache["width"], cache["height"], cache["fps"]
    gt_frames = [b["frame"] for b in gt["bounces"]]

    centers = _track(cache["detections"], fw, fh, fps)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(smoothed)  # curve-fit ignores speeds
    if thread_raw:
        is_real = [c is not None for c in centers]
        bounces = detect_bounces_from_trajectory(
            smoothed, speeds, fps, fh, fw, raw_centers=centers, is_real=is_real)
    else:
        bounces = detect_bounces_from_trajectory(smoothed, speeds, fps, fh, fw)
    pred = [b[0] for b in bounces]

    tol = round(0.15 * fps)
    tp, fp, fn = _match(pred, gt_frames, tol)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return f1, p, r, tp, fp, fn


def test_bounce_f1_no_regression():
    f1, p, r, tp, fp, fn = compute_f1()
    assert f1 >= F1_FLOOR, (
        f"Bounce F1 regressed: {f1:.3f} < floor {F1_FLOOR} "
        f"(P={p:.3f} R={r:.3f} TP={tp} FP={fp} FN={fn})")


def test_bounce_f1_prod_path_no_regression():
    """The LEGACY PROD path threads raw_centers/is_real → the turn-angle apex pass
    runs on felix too. It must stay above the floor (measured F1 0.774: the turn
    pass adds one mid-court FP; it is GAP-FILLER-only so it never displaces a y-V
    bounce). This is the test the no-arg call above does NOT cover."""
    f1, p, r, tp, fp, fn = compute_f1(thread_raw=True)
    assert f1 >= F1_FLOOR, (
        f"Bounce F1 (prod path, turn pass on) regressed: {f1:.3f} < floor {F1_FLOOR} "
        f"(P={p:.3f} R={r:.3f} TP={tp} FP={fp} FN={fn})")


if __name__ == "__main__":
    f1, p, r, tp, fp, fn = compute_f1()
    print(f"Bounce regression (felix cache, no-arg): F1={f1:.3f} P={p:.3f} R={r:.3f} "
          f"(TP={tp} FP={fp} FN={fn}); floor={F1_FLOOR}")
    f1p, pp, rp, tpp, fpp, fnp = compute_f1(thread_raw=True)
    print(f"Bounce regression (felix cache, PROD turn-pass): F1={f1p:.3f} P={pp:.3f} "
          f"R={rp:.3f} (TP={tpp} FP={fpp} FN={fnp}); floor={F1_FLOOR}")
    ok = f1 >= F1_FLOOR and f1p >= F1_FLOOR
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
