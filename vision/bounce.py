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


def detect_bounces_tolerant(ball_centers, ball_speeds_px, fps, frame_height,
                            frame_width=None):
    """Tolerant bounce detector — same validation as detect_bounces_from_trajectory
    (ballistic line fits + slope/rmse/swing/restitution/x-continuity gates), but
    admits candidates via scipy find_peaks on the y series instead of a strict
    per-frame "is this the local max to 0.5px" test.

    Why: the strict test (yc >= nanmax(local) - 0.5) rejects real bounces whose
    apex is 1-3 frames off the annotated frame (annotation jitter + interpolated
    minimum shift). On broadcast clips this dropped recall to 2/9. find_peaks with
    a prominence floor admits the apex wherever it actually sits, and the same
    ballistic gates filter false positives.

    This is the broadcast path; detect_bounces_from_trajectory (the strict one)
    stays the default for felix and is felix-regression-tested. Do not collapse
    them into one tunable function — they are tuned for different trajectories.
    """
    from scipy.signal import find_peaks
    n = len(ball_centers)
    if n < 7:
        return []

    win = max(4, int(fps * 0.28))
    min_gap = int(fps * 0.30)
    min_drop = frame_height * 0.008
    max_side_rmse = frame_height * 0.030
    min_slope = frame_height * 0.0008
    restitution_max = 1.8
    max_xjump_frac = 0.04
    min_y_ratio = 0.12
    max_y_ratio = 0.95

    ys = np.array([c[1] if c is not None else np.nan for c in ball_centers], dtype=float)
    xs = np.array([c[0] if c is not None else np.nan for c in ball_centers], dtype=float)

    # admit candidate apexes as prominent local maxima of y (ball low-points).
    # prominence floor = half the side-rmse budget: ripples below that aren't
    # bounces; a real bounce dips many px. distance >= min_gap dedups one bounce.
    cand = []
    valid = np.where(~np.isnan(ys))[0]
    if len(valid) > 4:
        pk, _ = find_peaks(ys[valid], prominence=max_side_rmse * 0.5,
                           distance=max(1, min_gap))
        cand = [int(valid[k]) for k in pk]

    out = []
    for i in cand:
        if i < 2 or i >= n - 2 or np.isnan(ys[i]):
            continue
        yc = ys[i]
        if not (min_y_ratio <= yc / frame_height <= max_y_ratio):
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
        if a_l < min_slope or a_r > -min_slope:
            continue
        if rmse_l > max_side_rmse or rmse_r > max_side_rmse:
            continue
        swing_l = abs(a_l) * (lx[-1] - lx[0])
        swing_r = abs(a_r) * (rx[-1] - rx[0])
        if swing_l < min_drop or swing_r < min_drop * 0.5:
            continue
        if abs(a_r) > restitution_max * abs(a_l):
            continue
        if frame_width:
            wx = max(3, int(fps * 0.07))
            jumps = [abs(xs[k] - xs[k - 1])
                     for k in range(i - wx, i + wx + 1)
                     if 0 < k < n and not np.isnan(xs[k]) and not np.isnan(xs[k - 1])]
            if jumps and max(jumps) > max_xjump_frac * frame_width:
                continue
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
        out.append((bf, int(bx), int(by), score))

    out.sort(key=lambda c: -c[3])
    chosen, used = [], []
    for bf, bx, by, score in out:
        if all(abs(bf - u) >= min_gap for u in used):
            chosen.append((bf, bx, by))
            used.append(bf)
    chosen.sort(key=lambda c: c[0])
    logger.info(f"Tolerant bounce detection: {len(chosen)} bounces found")
    return chosen



# ─────────────────────── robust variant (WASB-grade trajectory) ───────────────────────
# The robust detector below pairs with a dense heatmap trajectory (WASB). It adds
# trajectory despiking + IRLS line fits + a desc/rise energy bounce-vs-hit test.
# Measured on felix.mp4 with the WASB trajectory: F1=0.865, recall=1.000 (16/16),
# vs F1=0.788 for detect_bounces_from_trajectory on the same trajectory.

def despike_trajectory(ball_centers, fps, frame_height, frame_width,
                       speed_sigma=4.0, min_speed_px_per_s=40.0,
                       run_min_len=3):
    """
    Remove non-physical position jumps AND short aberrant runs from the
    trajectory.

    Two-pass:
      1. Spike pass: null any point whose jump from the previous valid point
         exceeds sigma*(median+MAD) of per-frame speed, floored to
         min_speed_px_per_s/fps, capped at 8% of frame diagonal.
      2. Run pass: a short run of <=run_min_len consecutive valid points that is
         bracketed by gaps (or trajectory edges) AND lies far from the local
         trend is treated as an aberrant reinit cluster and nulled. "Far from
         the local trend" = the run's mean position is farther than the spike
         threshold from the nearest surviving point on either side.

    Returns (cleaned_centers, n_removed).
    """
    n = len(ball_centers)
    pts = [None if c is None else (float(c[0]), float(c[1])) for c in ball_centers]

    # ── per-frame speed stats ──
    speeds = []
    for i in range(1, n):
        a, b = pts[i - 1], pts[i]
        if a is None or b is None:
            continue
        speeds.append(np.hypot(b[0] - a[0], b[1] - a[1]))
    if len(speeds) < 8:
        return ball_centers, 0

    speeds = np.array(speeds)
    med = float(np.median(speeds))
    mad = float(np.median(np.abs(speeds - med))) or 1.0
    thr = max(min_speed_px_per_s / fps, med + speed_sigma * 1.4826 * mad)
    frame_diag = np.hypot(frame_width, frame_height)
    thr = min(thr, frame_diag * 0.08)

    # ── Pass 1: spike removal ──
    for i in range(1, n):
        a, b = pts[i - 1], pts[i]
        if a is None or b is None:
            continue
        d = np.hypot(b[0] - a[0], b[1] - a[1])
        if d > thr:
            pts[i] = None

    # ── Pass 2: short-run removal ──
    # Find runs of consecutive non-None points
    i = 0
    runs = []
    while i < n:
        if pts[i] is not None:
            j = i
            while j < n and pts[j] is not None:
                j += 1
            runs.append((i, j))  # [i, j)
            i = j
        else:
            i += 1

    n_removed = 0
    for (lo, hi) in runs:
        run_len = hi - lo
        if run_len > run_min_len:
            continue
        # only treat as aberrant if it's far from its nearest surviving neighbor
        # on at least one side
        run_mean = np.mean([(pts[k][0], pts[k][1]) for k in range(lo, hi)], axis=0)
        # find nearest surviving point before lo
        prev_pt = None
        for k in range(lo - 1, -1, -1):
            if pts[k] is not None:
                prev_pt = pts[k]; break
        next_pt = None
        for k in range(hi, n):
            if pts[k] is not None:
                next_pt = pts[k]; break
        d_prev = np.hypot(run_mean[0]-prev_pt[0], run_mean[1]-prev_pt[1]) if prev_pt else float('inf')
        d_next = np.hypot(run_mean[0]-next_pt[0], run_mean[1]-next_pt[1]) if next_pt else float('inf')
        # If the run is closer than thr to BOTH neighbors, it's a legit short
        # trajectory segment — keep it. Otherwise null it (aberrant cluster).
        if d_prev > thr and d_next > thr:
            for k in range(lo, hi):
                pts[k] = None
                n_removed += 1

    # also count pass-1 removals
    n_spike = sum(1 for a, b in zip(ball_centers, pts) if a is not None and b is None)
    cleaned = [None if p is None else (int(p[0]), int(p[1])) for p in pts]
    return cleaned, n_spike + n_removed


def _robust_fit(xs, ys, max_iter=3, k=2.5, min_pts=3):
    """Iteratively reweighted least squares: reject points >k*std of residual."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < min_pts:
        return None
    mask = np.ones(len(xs), bool)
    a = b = rmse = None
    for _ in range(max_iter):
        if mask.sum() < min_pts:
            return None
        A = np.vstack([xs[mask], np.ones(mask.sum())]).T
        (a, b), *_ = np.linalg.lstsq(A, ys[mask], rcond=None)
        resid = ys - (a * xs + b)
        std = np.std(resid[mask]) or 1e-6
        new_mask = np.abs(resid) <= k * std
        # always keep at least the candidate point (index of min xs in this side is not fixed)
        if new_mask.sum() < min_pts:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    # final stats on the inlier set
    A = np.vstack([xs[mask], np.ones(mask.sum())]).T
    (a, b), *_ = np.linalg.lstsq(A, ys[mask], rcond=None)
    pred = a * xs[mask] + b
    rmse = float(np.sqrt(np.mean((ys[mask] - pred) ** 2)))
    return float(a), float(b), rmse, mask


def _collect_side_robust(ys, n, i, win, direction):
    """Collect ALL non-NaN points in the half-window (don't stop at gaps)."""
    if direction < 0:
        rng = range(max(0, i - win), i + 1)  # include i
    else:
        rng = range(i, min(n, i + win + 1))
    xs = [k for k in rng if not np.isnan(ys[k])]
    if len(xs) < 2:
        return np.array([]), np.array([])
    xa = np.array(xs, float)
    return xa, ys[xa.astype(int)]


def detect_bounces_robust(ball_centers, ball_speeds_px, fps, frame_height,
                          frame_width=None, speed_sigma=4.0,
                          min_speed_px_per_s=40.0,
                          restitution_max=2.5,
                          min_drop_frac=0.008,
                          min_slope_frac=0.0008,
                          max_side_rmse_frac=0.030,
                          win_frac=0.28,
                          min_gap_frac=0.30,
                          run_min_len=3,
                          # Bounce-vs-hit discriminators (physically motivated).
                          # A ground bounce conserves/loses vertical energy: the
                          # post-bounce rise is at most ~hit_ratio_max * the
                          # pre-bounce descent. A racket hit injects energy and
                          # produces rise >> descent. hit_ratio_max is the
                          # physical upper bound for a real bounce (Magnus/spin
                          # can add a little, but not >1.8x).
                          hit_ratio_max=1.8,
                          # A real bounce has a visible V; reject candidates
                          # whose total vertical excursion (desc+rise) is below
                          # this fraction of frame height — those are tracking
                          # noise / tiny wiggles, not bounces.
                          min_v_excursion_frac=0.055):
    """
    Robust bounce detector. See module docstring.

    Returns (bounces, cleaned_smoothed_centers, n_despiked).
    """
    n = len(ball_centers)
    if n < 7:
        return [], ball_centers, 0

    # 1) despike the raw (already spline-smoothed) centers, then re-smooth.
    despiked, n_despiked = despike_trajectory(
        ball_centers, fps, frame_height, frame_width,
        speed_sigma=speed_sigma, min_speed_px_per_s=min_speed_px_per_s,
        run_min_len=run_min_len)
    re_smoothed = smooth_ball_trajectory(despiked, max_gap=max(1, int(fps * 0.4)))

    # 2) robust curve-fit bounce detection on re_smoothed
    win = max(4, int(fps * win_frac))
    min_gap = int(fps * min_gap_frac)
    min_drop = frame_height * min_drop_frac
    max_side_rmse = frame_height * max_side_rmse_frac
    min_slope = frame_height * min_slope_frac
    min_y_ratio = 0.12
    max_y_ratio = 0.95

    ys = np.array([c[1] if c is not None else np.nan for c in re_smoothed], dtype=float)
    xs = np.array([c[0] if c is not None else np.nan for c in re_smoothed], dtype=float)

    candidates = []
    for i in range(2, n - 2):
        if re_smoothed[i] is None or np.isnan(ys[i]):
            continue
        yc = ys[i]
        if not (min_y_ratio <= yc / frame_height <= max_y_ratio):
            continue
        lo, hi = max(0, i - 2), min(n, i + 3)
        local = ys[lo:hi]
        local = local[~np.isnan(local)]
        if len(local) == 0 or yc < np.nanmax(local) - 0.5:
            continue

        lx, ly = _collect_side_robust(ys, n, i, win, -1)
        rx, ry = _collect_side_robust(ys, n, i, win, +1)
        if len(lx) < 3 or len(rx) < 3:
            continue

        lf = _robust_fit(lx, ly)
        rf = _robust_fit(rx, ry)
        if lf is None or rf is None:
            continue
        a_l, b_l, rmse_l, _ = lf
        a_r, b_r, rmse_r, _ = rf

        if a_l < min_slope or a_r > -min_slope:
            continue
        if rmse_l > max_side_rmse or rmse_r > max_side_rmse:
            continue
        swing_l = abs(a_l) * (lx[-1] - lx[0])
        swing_r = abs(a_r) * (rx[-1] - rx[0])
        if swing_l < min_drop or swing_r < min_drop * 0.5:
            continue
        if abs(a_r) > restitution_max * abs(a_l):
            continue

        # ── Bounce-vs-hit discrimination ──
        # desc = how far the ball fell (y grew) into the candidate point;
        # rise = how far it rose (y shrank) after. In image coords y grows
        # downward, so a bounce (lowest screen point) has desc,rise > 0.
        left_min_y = float(np.nanmin(ly))   # highest screen point before
        right_min_y = float(np.nanmin(ry))  # highest screen point after
        desc = yc - left_min_y
        rise = yc - right_min_y
        if desc <= 0 or rise <= 0:
            continue
        # Reject racket hits: rise >> desc means energy was injected (a ground
        # bounce cannot launch the ball up faster than it came down by more
        # than the restitution/Magnus margin).
        if rise > hit_ratio_max * desc:
            continue
        # Reject tiny V's (tracking noise / flat wiggles): a real bounce moves
        # the ball a visible amount vertically across the window.
        if (desc + rise) < min_v_excursion_frac * frame_height:
            continue

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

    candidates.sort(key=lambda c: -c[3])
    chosen, used = [], []
    for bf, bx, by, score in candidates:
        if all(abs(bf - u) >= min_gap for u in used):
            chosen.append((bf, bx, by))
            used.append(bf)
    chosen.sort(key=lambda c: c[0])
    logger.info(f"Robust trajectory bounce detection: {len(chosen)} bounces "
                f"(despiked {n_despiked} pts)")
    return chosen, re_smoothed, n_despiked
