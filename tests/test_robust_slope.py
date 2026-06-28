"""
Robust direction-slope unit test (fast, no GPU / no video decode).

`vision.bounce._veto_slope` is the direction estimator behind the bounce-vs-hit
firewall: it feeds vx_flip / vy_flip in BOTH the legacy `vx_flip_veto` and the
`--event-methodo` `vision.events._direction_features`. A plain least-squares fit
has a 0% breakdown point — a SINGLE one-frame tracker jitter spike on an
otherwise clean ballistic arc dominates the slope and can flip its SIGN, faking
a vy-flip → a phantom mandatory BOUNCE anchor (the S3 bug). This test pins the
robust (Theil-Sen) behavior so the regression can't silently come back:

  1. A CLEAN monotone arc reads the exact analytic slope (robustness is a no-op
     on clean data — same answer as least-squares).
  2. A clean DESCENDING arc with ONE injected +Δ outlier still reads a POSITIVE
     (descending, y-growing) slope — the spike does NOT flip the sign. A plain
     least-squares fit on the SAME points DOES flip it (the failure being fixed).
  3. The 2-point degenerate side is byte-identical to least-squares (Theil-Sen of
     two points == their single connecting slope), so the load-bearing felix
     direction features on dense windows are unchanged where support is minimal.

Run:  venv/bin/python tests/test_robust_slope.py
or:   venv/bin/python -m pytest tests/test_robust_slope.py -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vision.bounce import _veto_slope  # noqa: E402


def _lstsq_slope(pts, axis):
    """The OLD non-robust implementation, kept here only to prove the contrast."""
    if len(pts) < 2:
        return None
    ks = np.array([p[0] for p in pts], float)
    vs = np.array([p[axis] for p in pts], float)
    A = np.vstack([ks, np.ones(len(ks))]).T
    (a, _b), *_ = np.linalg.lstsq(A, vs, rcond=None)
    return float(a)


def _arc(frames, x_slope, y_slope, x0=900.0, y0=300.0):
    """Build (frame, x, y) points with exact linear slopes."""
    return [(f, x0 + x_slope * (f - frames[0]), y0 + y_slope * (f - frames[0]))
            for f in frames]


def test_clean_arc_matches_analytic_slope():
    """On clean data the robust slope == the analytic (and least-squares) slope."""
    pts = _arc(range(100, 106), x_slope=-3.0, y_slope=+8.0)
    assert abs(_veto_slope(pts, 1) - (-3.0)) < 1e-9   # vx
    assert abs(_veto_slope(pts, 2) - (+8.0)) < 1e-9   # vy
    # and it agrees with plain least-squares on clean data
    assert abs(_veto_slope(pts, 2) - _lstsq_slope(pts, 2)) < 1e-9


def test_single_outlier_does_not_flip_vy_sign():
    """THE BUG: a clean DESCENDING arc (vy>0, y growing in image coords) + ONE
    upward jitter spike. Robust slope stays POSITIVE (still descending); plain
    least-squares is dragged NEGATIVE by the spike (the faked vy-flip)."""
    # descending side: y grows by ~+4 px/frame over 5 frames (a normal gentle fall
    # near the apex — small total trend, exactly where one spike can dominate).
    pts = _arc(range(106, 111), x_slope=-2.0, y_slope=+4.0)
    # inject a single big upward spike (y jumps -57 px) at the last point — exactly
    # the one-frame +Δ jitter shape from the PM counterfactual (sign flips here).
    f, x, y = pts[-1]
    pts[-1] = (f, x, y - 57.0)

    robust = _veto_slope(pts, 2)
    naive = _lstsq_slope(pts, 2)
    assert robust > 0, (
        f"robust vy slope must stay POSITIVE (descending) despite one spike; "
        f"got {robust:.2f}")
    assert naive < 0, (
        f"sanity: the OLD least-squares fit SHOULD be flipped negative by the "
        f"spike (that is the bug being fixed); got {naive:.2f}")


def test_two_point_side_equals_least_squares():
    """A 2-point side has exactly one pairwise slope == the least-squares slope:
    the degenerate case is byte-identical to the old fit (felix-safety)."""
    pts = [(200, 880.0, 410.0), (203, 868.0, 446.0)]
    assert abs(_veto_slope(pts, 1) - _lstsq_slope(pts, 1)) < 1e-9
    assert abs(_veto_slope(pts, 2) - _lstsq_slope(pts, 2)) < 1e-9


def test_handles_sparse_and_empty():
    """Contract preserved: <2 points -> None (abstain upstream)."""
    assert _veto_slope([], 1) is None
    assert _veto_slope([(5, 1.0, 2.0)], 1) is None


if __name__ == "__main__":
    test_clean_arc_matches_analytic_slope()
    test_single_outlier_does_not_flip_vy_sign()
    test_two_point_side_equals_least_squares()
    test_handles_sparse_and_empty()
    print("robust-slope unit tests: ALL PASS")
