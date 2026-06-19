"""
SOTA-inspired bounce detection: local-y-extremum candidates validated by a
two-segment (descending / ascending) least-squares fit, taking the fit
intersection as the bounce frame (arXiv 2107.09255 approach, adapted to
operate on a 2D image-space trajectory rather than 3D).

Core signal (validated empirically on felix.mp4): a bounce is a sharp LOCAL
MAXIMUM of the image-y coordinate (ball descends to the ground = y grows in
image coords, then ascends = y shrinks). We:
  1. find local-y-maxima candidates on the smoothed trajectory,
  2. around each candidate, fit a line to the descending side and a line to the
     ascending side; a real bounce has both slopes present and opposite-signed
     (down-then-up) with a good combined fit (low residual),
  3. refine the bounce frame as the intersection of the two fitted lines,
  4. apply physical gates (on-court y-band, min spacing between bounces).

This is robust to the noise that defeats the per-frame velocity-sign-change
method, because it fits over a window instead of trusting a single Δy.
"""
import numpy as np


def _fit_line(xs, ys):
    """Least-squares line y = a*x + b. Returns (a, b, residual_per_point)."""
    if len(xs) < 2:
        return None
    A = np.vstack([xs, np.ones(len(xs))]).T
    (a, b), res, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = a * xs + b
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    return a, b, rmse


def detect_bounces_curvefit(ball_centers, fps, frame_height, frame_width=None,
                            win=None, min_drop=None, min_gap=None,
                            min_y_ratio=0.12, max_y_ratio=0.95,
                            max_side_rmse=None, min_slope=None):
    """
    Returns list of (frame_idx, x, y).

    Parameters auto-scale with fps; defaults tuned on 60fps tennis but expressed
    as ratios / fps multiples (zero hardcoding to one video).
    """
    n = len(ball_centers)
    if n < 7:
        return []

    win = win if win is not None else max(4, int(fps * 0.13))          # ~130ms half-window
    min_gap = min_gap if min_gap is not None else int(fps * 0.30)       # min frames between bounces
    min_drop = min_drop if min_drop is not None else frame_height * 0.012  # min y swing each side (px)
    max_side_rmse = max_side_rmse if max_side_rmse is not None else frame_height * 0.012
    min_slope = min_slope if min_slope is not None else frame_height * 0.0010  # px/frame

    ys = np.array([c[1] if c is not None else np.nan for c in ball_centers], dtype=float)
    xs = np.array([c[0] if c is not None else np.nan for c in ball_centers], dtype=float)

    candidates = []
    for i in range(win, n - win):
        if ball_centers[i] is None:
            continue
        yc = ys[i]
        if np.isnan(yc):
            continue
        y_ratio = yc / frame_height
        if y_ratio < min_y_ratio or y_ratio > max_y_ratio:
            continue

        # local maximum of y over ±2 frames (ball at lowest point in image)
        lo = max(0, i - 2)
        hi = min(n, i + 3)
        local = ys[lo:hi]
        local = local[~np.isnan(local)]
        if len(local) == 0 or yc < np.nanmax(local) - 1e-6:
            continue

        # gather descending side (before) and ascending side (after)
        left_x, left_y, right_x, right_y = [], [], [], []
        for k in range(i - win, i + 1):
            if 0 <= k < n and not np.isnan(ys[k]):
                left_x.append(k)
                left_y.append(ys[k])
        for k in range(i, i + win + 1):
            if 0 <= k < n and not np.isnan(ys[k]):
                right_x.append(k)
                right_y.append(ys[k])
        if len(left_x) < 3 or len(right_x) < 3:
            continue

        lf = _fit_line(np.array(left_x, float), np.array(left_y, float))
        rf = _fit_line(np.array(right_x, float), np.array(right_y, float))
        if lf is None or rf is None:
            continue
        a_l, b_l, rmse_l = lf
        a_r, b_r, rmse_r = rf

        # descending before (a_l > 0 : y increasing), ascending after (a_r < 0)
        if a_l < min_slope or a_r > -min_slope:
            continue
        if rmse_l > max_side_rmse or rmse_r > max_side_rmse:
            continue

        # vertical swing each side must be significant
        swing_l = abs(a_l) * (left_x[-1] - left_x[0])
        swing_r = abs(a_r) * (right_x[-1] - right_x[0])
        if swing_l < min_drop or swing_r < min_drop:
            continue

        # refine bounce frame as intersection of the two fitted lines
        if abs(a_l - a_r) > 1e-6:
            x_int = (b_r - b_l) / (a_l - a_r)
        else:
            x_int = float(i)
        # clamp intersection near the candidate
        if not (i - win <= x_int <= i + win):
            x_int = float(i)
        bf = int(round(x_int))
        bf = max(0, min(n - 1, bf))

        # score = sharpness of the V (sum of |slopes|) / fit error
        score = (abs(a_l) + abs(a_r)) / (1.0 + rmse_l + rmse_r)
        bx = xs[bf] if not np.isnan(xs[bf]) else xs[i]
        by = ys[bf] if not np.isnan(ys[bf]) else yc
        candidates.append((bf, int(bx), int(by), score))

    # non-max suppression by min_gap, keep highest score
    candidates.sort(key=lambda c: -c[3])
    chosen = []
    used = []
    for bf, bx, by, score in candidates:
        if all(abs(bf - u) >= min_gap for u in used):
            chosen.append((bf, bx, by))
            used.append(bf)
    chosen.sort(key=lambda c: c[0])
    return chosen
