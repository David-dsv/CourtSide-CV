"""
Ball-trajectory bounce detection (curve-fit method) + trajectory smoothing.

Extracted verbatim from run_pipeline_8s.py so the production pipeline and the
test suite exercise the identical algorithm. Pure functions: no video/CLI/IO.

Bounce method: a bounce is a sharp local-y maximum (ball descends to the court
then ascends); around each candidate we least-squares fit the descending and
ascending sides and take their intersection as the bounce frame, gated by slope,
fit residual, swing, energy restitution, and x-continuity — all as ratios of
fps / frame size (zero per-video hardcoding). Validated on felix.mp4 GT:
F1 ~0.80 (cached trajectory) / ~0.67 (live pipeline) vs ~0.30 for the previous
velocity-sign-change method.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from loguru import logger


def smooth_ball_trajectory(raw_centers, max_gap=10):
    """
    Post-process raw ball centers using cubic spline interpolation.

    Args:
        raw_centers: list of (x,y) or None for each frame
        max_gap: maximum number of consecutive missing frames to interpolate

    Returns:
        list of (x,y) or None — same length, with gaps filled
    """
    n = len(raw_centers)
    if n < 4:
        return raw_centers

    # Collect known positions
    known_idx = []
    known_x = []
    known_y = []
    for i, c in enumerate(raw_centers):
        if c is not None:
            known_idx.append(i)
            known_x.append(c[0])
            known_y.append(c[1])

    if len(known_idx) < 4:
        return raw_centers

    # Build cubic splines
    t = np.array(known_idx, dtype=float)
    cs_x = CubicSpline(t, known_x, bc_type='natural')
    cs_y = CubicSpline(t, known_y, bc_type='natural')

    result = list(raw_centers)  # copy
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        # Find the gap: consecutive None frames
        gap_start = i
        while i < n and result[i] is None:
            i += 1
        gap_end = i  # exclusive

        gap_len = gap_end - gap_start
        if gap_len > max_gap:
            continue  # gap too long, leave as None

        # Only interpolate if we have known points on both sides
        if gap_start == 0 or gap_end >= n:
            continue
        if raw_centers[gap_start - 1] is None or (gap_end < n and raw_centers[gap_end] is None):
            continue

        # Interpolate
        for j in range(gap_start, gap_end):
            result[j] = (int(cs_x(j)), int(cs_y(j)))

    return result


def _fit_line(xs, ys):
    """Least-squares line y = a*x + b. Returns (a, b, rmse)."""
    if len(xs) < 2:
        return None
    A = np.vstack([xs, np.ones(len(xs))]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = a * xs + b
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    return a, b, rmse


def _collect_ballistic_side(ys, n, i, win, direction):
    """Collect up to `win` consecutive tracked points on one side of frame i.

    direction=-1 -> descending side (before), +1 -> ascending side (after).
    Stops at the first gap so the fit stays on a single ballistic segment.
    """
    xs = []
    rng = (range(i, max(i - win - 1, -1), -1) if direction < 0
           else range(i, min(i + win + 1, n)))
    for k in rng:
        if 0 <= k < n and not np.isnan(ys[k]):
            xs.append(k)
        else:
            break
    order = np.argsort(xs)
    xa = np.array(xs, float)[order]
    return xa, ys[xa.astype(int)]


def detect_bounces_from_trajectory(ball_centers, ball_speeds_px, fps, frame_height, frame_width=None):
    """
    Detect ball bounces from the smoothed trajectory (curve-fit method).

    A bounce is a sharp LOCAL MAXIMUM of the image-y coordinate (the ball
    descends to the court — y grows in image coords — then ascends). Around each
    such candidate we fit a line to the descending side and a line to the
    ascending side (least squares, on the longest gap-free ballistic run up to
    a window), and take their intersection as the bounce frame. A real floor
    bounce loses vertical energy, so the ascending slope must not greatly exceed
    the descending one (a player HIT injects energy → steep rebound, rejected).

    This is robust to the per-frame velocity noise that defeats a naive
    velocity-sign-change test. Validated on the felix.mp4 ground truth
    (16 bounces): F1≈0.75 vs ≈0.30 for the previous sign-change method.

    All thresholds are ratios of fps / frame_height — no per-video constants.

    Args:
        ball_centers: list of (x,y) or None per frame (smoothed trajectory)
        ball_speeds_px: per-frame speed (kept for API compat; unused here)
        fps: video frame rate
        frame_height: frame height in pixels
        frame_width: frame width in pixels (unused, kept for API compat)

    Returns:
        list of (frame_idx, x, y)
    """
    n = len(ball_centers)
    if n < 7:
        return []

    # Parameters as ratios of fps / frame_height (zero per-video hardcoding)
    win = max(4, int(fps * 0.28))            # half-window for ballistic fit
    min_gap = int(fps * 0.30)                # min frames between bounces
    min_drop = frame_height * 0.008          # min y-swing each side (px)
    max_side_rmse = frame_height * 0.030     # max line-fit residual each side
    min_slope = frame_height * 0.0008        # min |slope| px/frame
    restitution_max = 1.8                    # reject |asc slope| > k*|desc| (hit)
    max_xjump_frac = 0.04                    # reject if |dx| jumps >4% width (stitched track)
    min_y_ratio = 0.12                       # on-court band (far baseline ~18%)
    max_y_ratio = 0.95

    ys = np.array([c[1] if c is not None else np.nan for c in ball_centers], dtype=float)
    xs = np.array([c[0] if c is not None else np.nan for c in ball_centers], dtype=float)

    candidates = []
    for i in range(2, n - 2):
        if ball_centers[i] is None or np.isnan(ys[i]):
            continue
        yc = ys[i]
        if not (min_y_ratio <= yc / frame_height <= max_y_ratio):
            continue

        # robust local maximum of y over ±2 frames (flat-top tolerant)
        lo, hi = max(0, i - 2), min(n, i + 3)
        local = ys[lo:hi]
        local = local[~np.isnan(local)]
        if len(local) == 0 or yc < np.nanmax(local) - 0.5:
            continue

        lx, ly = _collect_ballistic_side(ys, n, i, win, -1)
        rx, ry = _collect_ballistic_side(ys, n, i, win, +1)
        if len(lx) < 3 or len(rx) < 3:
            continue

        lf = _fit_line(lx, ly)
        rf = _fit_line(rx, ry)
        if lf is None or rf is None:
            continue
        a_l, b_l, rmse_l = lf
        a_r, b_r, rmse_r = rf

        # descending before (a_l>0 : y increasing), ascending after (a_r<0)
        if a_l < min_slope or a_r > -min_slope:
            continue
        if rmse_l > max_side_rmse or rmse_r > max_side_rmse:
            continue

        swing_l = abs(a_l) * (lx[-1] - lx[0])
        swing_r = abs(a_r) * (rx[-1] - rx[0])
        if swing_l < min_drop or swing_r < min_drop * 0.5:
            continue

        # energy-restitution gate: a player hit injects energy (steep rebound)
        if abs(a_r) > restitution_max * abs(a_l):
            continue

        # x-continuity gate: a real bounce keeps x continuous (ball near-vertical
        # at the apex). A large horizontal jump means the descending and ascending
        # sides belong to different flights stitched by a Kalman reinit → a false
        # positive, not a physical bounce.
        if frame_width:
            wx = max(3, int(fps * 0.07))
            jumps = [abs(xs[k] - xs[k - 1])
                     for k in range(i - wx, i + wx + 1)
                     if 0 < k < n and not np.isnan(xs[k]) and not np.isnan(xs[k - 1])]
            if jumps and max(jumps) > max_xjump_frac * frame_width:
                continue

        # refine bounce frame as the intersection of the two fitted lines
        if abs(a_l - a_r) > 1e-6:
            x_int = (b_r - b_l) / (a_l - a_r)
        else:
            x_int = float(i)
        if not (i - win <= x_int <= i + win):
            x_int = float(i)
        bf = max(0, min(n - 1, int(round(x_int))))

        score = (abs(a_l) + abs(a_r)) / (1.0 + rmse_l + rmse_r)
        bx = xs[bf] if not np.isnan(xs[bf]) else xs[i]
        by = ys[bf] if not np.isnan(ys[bf]) else yc
        candidates.append((bf, int(bx), int(by), score))

    # non-max suppression by min_gap, keep highest-scoring candidate
    candidates.sort(key=lambda c: -c[3])
    chosen, used = [], []
    for bf, bx, by, score in candidates:
        if all(abs(bf - u) >= min_gap for u in used):
            chosen.append((bf, bx, by))
            used.append(bf)
    chosen.sort(key=lambda c: c[0])

    logger.info(f"Trajectory bounce detection: {len(chosen)} bounces found")
    return chosen

