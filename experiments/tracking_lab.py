"""
Tracking + bounce experimentation lab.

Replays the run_pipeline_8s.py ball-tracking and bounce-detection logic from a
cached detection file (see cache_detections.py), with every knob exposed, so we
can measure the effect of each change against the bounce ground truth in seconds.

The tracking / smoothing code here is a faithful port of run_pipeline_8s.py
(BallKalmanFilter, the Pass-1b tracking loop, smooth_ball_trajectory) so that
improvements measured here transfer back to the pipeline 1:1.

Run the built-in A/B harness:
    venv/bin/python experiments/tracking_lab.py
"""
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402


# ─────────────────────────── Kalman (port) ───────────────────────────
class BallKalmanFilter:
    def __init__(self, process_noise=50.0, measurement_noise=8.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
        self.initialized = False
        self.frames_since_update = 0

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.frames_since_update = 0

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        self.frames_since_update += 1
        return float(pred[0][0]), float(pred[1][0])

    def update(self, x, y):
        if not self.initialized:
            self.init_state(x, y)
            return
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.frames_since_update = 0

    def gating_distance(self, x, y):
        if not self.initialized:
            return 0.0
        pred = self.kf.statePost
        return float(np.hypot(x - pred[0][0], y - pred[1][0]))


# ─────────────────────── smoothing (port) ───────────────────────
def smooth_ball_trajectory(raw_centers, max_gap=10):
    n = len(raw_centers)
    if n < 4:
        return raw_centers
    known_idx, known_x, known_y = [], [], []
    for i, c in enumerate(raw_centers):
        if c is not None:
            known_idx.append(i)
            known_x.append(c[0])
            known_y.append(c[1])
    if len(known_idx) < 4:
        return raw_centers
    t = np.array(known_idx, dtype=float)
    cs_x = CubicSpline(t, known_x, bc_type='natural')
    cs_y = CubicSpline(t, known_y, bc_type='natural')
    result = list(raw_centers)
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        gap_start = i
        while i < n and result[i] is None:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start
        if gap_len > max_gap:
            continue
        if gap_start == 0 or gap_end >= n:
            continue
        if raw_centers[gap_start - 1] is None or (gap_end < n and raw_centers[gap_end] is None):
            continue
        for j in range(gap_start, gap_end):
            result[j] = (int(cs_x(j)), int(cs_y(j)))
    return result


# ─────────────────────── static-FP filter (port, parameterizable) ───────────────────────
def build_static_filter(raw_detections, frame_count, static_radius=20, static_pct=0.08, enabled=True):
    """Returns an is_static(cx, cy) predicate. If enabled=False, never static."""
    if not enabled:
        return lambda cx, cy: False
    all_points = []
    for cands in raw_detections:
        for cx, cy, _ in cands:
            all_points.append((cx // static_radius, cy // static_radius))
    counts = Counter(all_points)
    threshold = max(10, int(frame_count * static_pct))
    static_zones = {k for k, v in counts.items() if v >= threshold}

    def is_static(cx, cy):
        gx, gy = cx // static_radius, cy // static_radius
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (gx + dx, gy + dy) in static_zones:
                    return True
        return False
    return is_static


# ─────────────────────── tracking loop (port, parameterizable) ───────────────────────
def track(raw_detections, frame_diag, fps, cfg):
    """
    Replays Pass-1b tracking. cfg is a dict of knobs. Returns
    (raw_ball_centers, ball_speeds_px).
    """
    is_static = build_static_filter(
        raw_detections, len(raw_detections),
        static_radius=cfg.get("static_radius", 20),
        static_pct=cfg.get("static_pct", 0.08),
        enabled=cfg.get("static_enabled", True),
    )

    kf = BallKalmanFilter(
        process_noise=cfg.get("process_noise", 50.0),
        measurement_noise=cfg.get("measurement_noise", 8.0),
    )
    kf_gate_base = cfg.get("kf_gate_base", 80.0)
    kf_gate_growth = cfg.get("kf_gate_growth", 40.0)
    kf_gate_max = frame_diag * cfg.get("kf_gate_max_ratio", 0.20)
    kf_max_coast = int(fps * cfg.get("kf_max_coast_s", 0.5))
    reinit_ratio = cfg.get("reinit_gate_ratio", 0.15)
    last_known_pos = None

    raw_ball_centers = []
    ball_speeds_px = []

    for frame_idx in range(len(raw_detections)):
        candidates = raw_detections[frame_idx]
        moving = [(cx, cy, sc) for cx, cy, sc in candidates if not is_static(cx, cy)]

        ball_center = None
        predicted = kf.predict()
        current_gate = min(kf_gate_max, kf_gate_base + kf_gate_growth * kf.frames_since_update)

        if moving:
            if not kf.initialized:
                best = max(moving, key=lambda c: c[2])
                kf.init_state(best[0], best[1])
                ball_center = (best[0], best[1])
            else:
                best_det, best_dist = None, float('inf')
                for cx, cy, sc in moving:
                    dist = kf.gating_distance(cx, cy)
                    if dist < current_gate and dist < best_dist:
                        best_dist, best_det = dist, (cx, cy)
                if best_det is not None:
                    kf.update(best_det[0], best_det[1])
                    state = kf.kf.statePost
                    ball_center = (int(state[0][0]), int(state[1][0]))
                elif kf.frames_since_update <= kf_max_coast and predicted:
                    ball_center = (int(predicted[0]), int(predicted[1]))
                elif last_known_pos is not None:
                    nearest, nearest_dist = None, float('inf')
                    reinit_gate = frame_diag * reinit_ratio
                    for cx, cy, sc in moving:
                        d = np.hypot(cx - last_known_pos[0], cy - last_known_pos[1])
                        if d < reinit_gate and d < nearest_dist:
                            nearest_dist, nearest = d, (cx, cy)
                    if nearest is not None:
                        kf.init_state(nearest[0], nearest[1])
                        ball_center = nearest

        if ball_center is None and kf.initialized:
            if kf.frames_since_update <= kf_max_coast and predicted:
                ball_center = (int(predicted[0]), int(predicted[1]))

        if ball_center is not None:
            last_known_pos = ball_center

        if kf.initialized:
            vx = float(kf.kf.statePost[2][0])
            vy = float(kf.kf.statePost[3][0])
            speed_px = np.hypot(vx, vy)
        else:
            speed_px = 0.0
        ball_speeds_px.append(speed_px)
        raw_ball_centers.append(ball_center)

    return raw_ball_centers, ball_speeds_px


# ─────────────────────── loaders / metrics ───────────────────────
def load_cache(path):
    return json.load(open(path))


def load_gt(path):
    d = json.load(open(path))
    return [b["frame"] for b in d["bounces"]], d.get("fps", 60.0)


def match_bounces(pred_frames, truth_frames, tol):
    cand = []
    for p in pred_frames:
        for t in truth_frames:
            dd = abs(p - t)
            if dd <= tol:
                cand.append((dd, p, t))
    cand.sort(key=lambda c: (c[0], c[1], c[2]))
    up, ut, m = set(), set(), []
    for dd, p, t in cand:
        if p in up or t in ut:
            continue
        up.add(p)
        ut.add(t)
        m.append((p, t))
    fp = sorted(p for p in pred_frames if p not in up)
    fn = sorted(t for t in truth_frames if t not in ut)
    return m, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate(pred_frames, gt_frames, fps, tol=None):
    tol = tol if tol is not None else round(0.15 * fps)
    m, fp, fn = match_bounces(pred_frames, gt_frames, tol)
    tp = len(m)
    p, r, f = prf(tp, len(fp), len(fn))
    return {"tp": tp, "fp": len(fp), "fn": len(fn), "precision": p, "recall": r,
            "f1": f, "tol": tol, "fp_frames": fp, "fn_frames": fn,
            "n_pred": len(pred_frames)}
