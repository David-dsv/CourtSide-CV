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


def _velocity_turn_candidates(raw_centers, is_real, fps, frame_width, frame_height,
                              angle_deg=70.0, win_frac=0.06, min_gap_frac=0.15,
                              min_disp_frac=0.002):
    """Candidate bounce frames from the ball's VELOCITY-VECTOR TURN ANGLE.

    The y-extremum candidate test in detect_bounces_from_trajectory needs a
    prominent local y-maximum (the ball descends then ascends). That MISSES
    far-court APEX bounces: when the ball bounces near the top of a flat arc its
    image-y barely dips (~1-6px on demo3) so no y-peak exists — yet its velocity
    VECTOR still turns sharply (the vertical component swings from flat to rising
    while the horizontal sign is preserved, the bounce signature). On the demo3
    cache only 2/9 GT bounces have a y-extremum but 8/9 have a >50° turn.

    For each frame with REAL (non-interpolated) neighbours k frames either side,
    we compute the angle between the incoming velocity (f-k → f) and the outgoing
    one (f → f+k) and admit the LOCAL PEAKS above ``angle_deg``. Read on the RAW
    pre-spline track + is_real mask (same gap-safety invariant as the vx-flip
    veto): a turn is trusted only when all three arms rest on real detections, so
    we never fabricate a turn across an interpolated gap.

    This is the SAME PRINCIPLE as vision.events.detect_sharp_turns, re-derived
    here so the LEGACY bounce path (which must not import vision.events — that is
    the --event-methodo path) recovers apex bounces on its own. The candidates it
    yields are still filtered by the caller's ballistic / direction gates, so it
    raises recall without flooding false positives.

    All thresholds are ratios of fps / frame size — no per-video constants.
    Returns a sorted list of candidate frame indices.
    """
    from scipy.signal import find_peaks
    n = len(raw_centers)
    if n < 7:
        return []
    k = max(2, round(fps * win_frac))
    diag = float(np.hypot(frame_width or frame_height, frame_height))
    min_disp = diag * min_disp_frac        # ignore a near-stationary ball (noise)
    ang = np.zeros(n, dtype=float)
    for f in range(k, n - k):
        if not (is_real[f - k] and is_real[f] and is_real[f + k]):
            continue
        c0, c1, c2 = raw_centers[f - k], raw_centers[f], raw_centers[f + k]
        if c0 is None or c1 is None or c2 is None:
            continue
        ax, ay = c1[0] - c0[0], c1[1] - c0[1]
        bx, by = c2[0] - c1[0], c2[1] - c1[1]
        na = (ax * ax + ay * ay) ** 0.5
        nb = (bx * bx + by * by) ** 0.5
        if na < min_disp or nb < min_disp:
            continue
        cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        ang[f] = np.degrees(np.arccos(cos))
    dist = max(1, int(fps * min_gap_frac))
    pk, _ = find_peaks(ang, height=angle_deg, distance=dist)
    return sorted(int(f) for f in pk)


def detect_bounces_from_trajectory(ball_centers, ball_speeds_px, fps, frame_height,
                                   frame_width=None, raw_centers=None, is_real=None,
                                   return_turn_frames=False):
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
        frame_width: frame width in pixels (used by the x-continuity + turn gates)
        raw_centers: OPTIONAL pre-spline tracker centers ((x,y) or None per frame).
                     When supplied with ``is_real`` the detector ALSO admits
                     far-court APEX bounce candidates from the velocity-vector turn
                     angle (see _velocity_turn_candidates) — these have no y-peak so
                     the strict y-max test alone misses them. Each turn candidate is
                     still validated by an on-court band + a vertical-reflection
                     (vy down→up) co-signal + a vx-not-flipped check (a vx flip is a
                     racket HIT, not a bounce), so recall rises without flooding FPs.
                     felix path passes None → behaviour byte-identical to before.
        is_real:     OPTIONAL bool-per-frame mask, True where raw_centers is a real
                     detection (not a spline fill). Required for the turn candidates.

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

    # ── 2nd pass: far-court APEX bounces via the velocity-vector turn angle ──
    # These have no prominent y-extremum (a flat far arc) so the loop above never
    # admits them, but they DO turn the velocity vector sharply. We add them ONLY
    # when the caller threaded the RAW pre-spline track + is_real mask (felix path
    # passes None → this whole block is skipped, behaviour byte-identical). Each
    # turn candidate is validated by an independent, lighter gate that is correct
    # for an apex (where the y-V test fails): on-court band + a vertical-reflection
    # co-signal (vy flips down→up across the turn) + a vx-not-flipped check so a
    # racket HIT (which inverts vx) is NOT promoted to a bounce. This keeps the
    # bounce-vs-hit firewall the rest of the pipeline relies on.
    #
    # SAFETY (felix-prod path): turn candidates are a pure GAP-FILLER for apex
    # bounces — they must NEVER compete with a y-extremum bounce. So (1) we DROP any
    # turn candidate within min_gap of a y-V candidate (the y-V detector already
    # owns that region; letting a raw turn frame win NMS there would replace the
    # line-intersection–refined frame and could inject an FP — measured felix prod
    # F1 0.800→0.774 before this guard), and (2) we score survivors strictly BELOW
    # every y-V score so NMS always prefers a real y-V bounce on any residual tie.
    if raw_centers is not None and is_real is not None and frame_width:
        rys = np.array([c[1] if c is not None else np.nan for c in raw_centers], float)
        rxs = np.array([c[0] if c is not None else np.nan for c in raw_centers], float)
        turn_win = max(2, round(fps * 0.06))   # ± arm for the velocity vectors
        yv_frames = [cd[0] for cd in candidates]               # y-extremum frames
        yv_min_score = min((cd[3] for cd in candidates), default=1.0)
        turn_score = min(0.5, yv_min_score * 0.5)              # strictly below y-V
        turn_admitted = []                                     # provenance tagging
        for tf in _velocity_turn_candidates(raw_centers, is_real, fps,
                                            frame_width, frame_height):
            if tf < turn_win or tf >= n - turn_win:
                continue
            # GAP-FILLER only: a y-V bounce already covers this region → defer to it
            # (never let a raw turn frame displace a refined y-V bounce in NMS).
            if any(abs(tf - yf) < min_gap for yf in yv_frames):
                continue
            yc = rys[tf]
            if np.isnan(yc) or not (min_y_ratio <= yc / frame_height <= max_y_ratio):
                continue
            c0 = raw_centers[tf - turn_win]
            c1 = raw_centers[tf]
            c2 = raw_centers[tf + turn_win]
            if c0 is None or c1 is None or c2 is None:
                continue
            # velocity components across the turn (pre = into tf, post = out of tf)
            vx_pre, vy_pre = c1[0] - c0[0], c1[1] - c0[1]
            vx_post, vy_post = c2[0] - c1[0], c2[1] - c1[1]
            # vertical reflection: a bounce descends (vy_pre >= ~0, y growing) then
            # ascends (vy_post < 0). Require the post side to rise meaningfully and
            # the pre side to NOT be strongly rising (that would be a pure ascent /
            # a hit launch). Scale-free floor on |vy_post|.
            vy_floor = frame_height * 0.0015 * (turn_win)   # ~few px over the arm
            if not (vy_post < -vy_floor and vy_pre > -vy_floor):
                continue
            # vx must be PRESERVED — a confident vx-sign flip is a racket hit, not a
            # bounce. Only enforce when both sides move horizontally enough to read
            # a sign (else abstain → keep, the apex may be near-vertical).
            netdx = frame_width * 0.004
            if abs(vx_pre) > netdx and abs(vx_post) > netdx:
                if np.sign(vx_pre) != np.sign(vx_post):
                    continue   # vx flip → hit, reject as a bounce
            # x-continuity: reject if the track teleports across the turn window
            wx = max(3, int(fps * 0.07))
            jumps = [abs(rxs[k] - rxs[k - 1])
                     for k in range(tf - wx, tf + wx + 1)
                     if 0 < k < n and not np.isnan(rxs[k]) and not np.isnan(rxs[k - 1])]
            if jumps and max(jumps) > max_xjump_frac * frame_width:
                continue
            bx = int(rxs[tf]) if not np.isnan(rxs[tf]) else int(c1[0])
            by = int(yc)
            candidates.append((tf, bx, by, turn_score))
            turn_admitted.append(tf)

    # non-max suppression by min_gap, keep highest-scoring candidate
    candidates.sort(key=lambda c: -c[3])
    chosen, used = [], []
    turn_chosen = set()   # which chosen bounces came ONLY from a turn (lower conf)
    _turn_set = set(turn_admitted) if (raw_centers is not None and is_real is not None
                                       and frame_width) else set()
    for bf, bx, by, score in candidates:
        if all(abs(bf - u) >= min_gap for u in used):
            chosen.append((bf, bx, by))
            used.append(bf)
            if bf in _turn_set:
                turn_chosen.add(bf)
    chosen.sort(key=lambda c: c[0])

    logger.info(f"Trajectory bounce detection: {len(chosen)} bounces found")
    if return_turn_frames:
        # frames of chosen bounces that were sourced ONLY from the turn-angle pass
        # (no y-extremum) — these are the LOWER-CONFIDENCE recoveries the legacy
        # bounce↔hit arbitration may demote on a collision with a confident hit.
        return chosen, sorted(turn_chosen)
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


# ─────────────────────── vx-flip veto (bounce vs hit by direction) ───────────────────────
# A floor bounce PRESERVES the ball's horizontal direction (it keeps crossing the
# same way) and FLIPS its vertical one (descends, then ascends). A racket HIT
# INVERTS the horizontal direction (the player sends it back). So a bounce
# candidate whose horizontal velocity sign *reverses* across the contact frame is
# really a hit emitted as a bounce by the detector — we veto it.
#
# Measured on the felix WASB native-1080p cache (16 GT bounces): at the 16 true
# bounces vx-sign is PRESERVED 15/15 (f424 is occluded → abstains), and the
# hit-FPs f1609 (vx -16.2 -> +10.9) / f820 (vx -11.6 -> +11.7) reverse ~180°.
# Vetoing them lifts bounce F1 0.889 -> 0.914 with 0 true bounce lost.
#
# SAFETY INVARIANT (the whole point): the veto is ONE-DIRECTIONAL. It only ever
# REJECTS a candidate, and only when the vx sign flip is confident AND measurable
# (enough real points, no gap/teleport in the window, both sides above the
# deadband). In every doubtful case it ABSTAINS (keeps the candidate, tagged
# "unconfirmed"). It therefore CANNOT lower recall — it can only add precision.
#
# It MUST read direction on the RAW pre-spline tracker centers + an is_real mask,
# never on the detector's re-smoothed array: the cubic spline rounds the
# reflection (degrades the GT signal 15/15 -> 14/15) and reads through filled
# gaps. See docs/research/bounce-vs-shot-direction.md (CHANTIER 0).

def _veto_collect_side(raw_centers, is_real, lo, hi):
    """Real (is_real) points (frame, x, y) with frame in [lo, hi] (inclusive)."""
    n = len(raw_centers)
    out = []
    for k in range(lo, hi + 1):
        if 0 <= k < n and is_real[k] and raw_centers[k] is not None:
            c = raw_centers[k]
            out.append((k, float(c[0]), float(c[1])))
    return out


def _veto_max_gap(is_real, lo, hi):
    """Longest run of consecutive interpolated/missing frames in [lo, hi]."""
    n = len(is_real)
    run = mx = 0
    for k in range(lo, hi + 1):
        if 0 <= k < n and not is_real[k]:
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    return mx


def _veto_drop_teleports(pts, frame_width):
    """Drop points that teleport (>8% frame-width per FRAME) from the running
    anchor — a track reinit to the far side of the court would otherwise blow up
    the slope and fake a vx flip (e.g. felix f1520's post-window jump to x≈255).

    pts is (frame, x, y); the divisor below is the FRAME gap (pts[i][0] is the
    frame index), so the threshold is a per-frame speed, not an x-span. Returns
    (kept_pts, dropped_any). The caller MUST abstain when dropped_any is True: a
    dropped teleport means the track is unreliable across this window, so fitting
    the surviving sparse points could itself fake a confident sign — exactly the
    case that must abstain, never reject (the one-directional safety invariant).
    A transient jump-and-return is caught here; a teleport-and-stay is caught by
    dropped_any=True at the call site."""
    if len(pts) < 3 or not frame_width:
        return pts, False
    thr = frame_width * 0.08
    kept = [pts[0]]
    dropped = False
    for i in range(1, len(pts)):
        dframe = max(1, pts[i][0] - kept[-1][0])  # frame gap, not x-span
        d = np.hypot(pts[i][1] - kept[-1][1], pts[i][2] - kept[-1][2]) / dframe
        if d <= thr:
            kept.append(pts[i])
        else:
            dropped = True
    return kept, dropped


def _veto_slope(pts, axis):
    """Least-squares slope of pts[:,axis] vs frame (axis 1=x, 2=y). >=2 pts."""
    if len(pts) < 2:
        return None
    ks = np.array([p[0] for p in pts], float)
    vs = np.array([p[axis] for p in pts], float)
    A = np.vstack([ks, np.ones(len(ks))]).T
    (a, _b), *_ = np.linalg.lstsq(A, vs, rcond=None)
    return float(a)


def vx_flip_veto(bounce_events, raw_centers, is_real, fps, frame_width,
                 half_frac=0.10, gap_frac=0.05,
                 eps_rel=0.15, eps_floor_frac=0.0015, min_netdx_frac=0.004):
    """One-directional vx-flip veto (A3 ⊕ A4) — bounce-vs-hit by direction.

    Post-filters a bounce-candidate list (from detect_bounces_robust) using the
    RAW pre-spline tracker centers + an is_real mask. For each candidate frame f
    it fits the horizontal velocity (slope of x(t)) on the real points just
    BEFORE and just AFTER f, and:

      - REJECT  only when BOTH slopes are confident AND their SIGN FLIPS → a
                racket hit, not a bounce. "Confident" requires, on each side:
                >= 2 real points, no interpolated gap, no dropped teleport, |vx|
                above the deadband, AND a net horizontal displacement above
                min_netdx_frac*frame_width.
      - ABSTAIN (keep) on every uncertainty: sparse / occluded, an interpolated
                gap in a side, a teleport dropped inside the window, near-still
                horizontal motion (drop-shot / near-vertical bounce / slow far
                bounce), or a sub-floor net Δx.
      - CONFIRM (keep) if the sign is preserved.

    The vy flip (descends then ascends) is a CONFIDENCE co-signal only: a CONFIRM
    with vy flip is tagged "metric", otherwise "unconfirmed". vy NEVER removes a
    candidate (it would touch recall — see the safety invariant).

    SAFETY NOTE (why the deadband has an ABSOLUTE floor + a net-Δx gate): a purely
    relative deadband (eps = eps_rel*speed_scale) collapses toward zero on a
    near-vertical bounce, where tracker quantization noise then produces opposite
    sign slopes that both clear it — a real bounce would be REJECTED. The absolute
    floor eps_floor_frac*frame_width (and the net-Δx requirement) keep the veto
    abstaining when the ball is barely moving horizontally, restoring the
    one-directional invariant. Likewise a teleport-and-stay is caught by
    _veto_drop_teleports' dropped flag → abstain (never fit through the hole).

    Args:
        bounce_events: list of (frame, x, y) candidates (kept order preserved).
        raw_centers:   pre-spline tracker centers, (x, y) or None per frame.
        is_real:       bool per frame, True where raw_centers is a real detection.
        fps, frame_width: scale parameters (no per-video hardcoding).
        half_frac:     half-window in seconds (0.10 = round(fps*0.10) frames;
                       0.08-0.12 is 15/15 stable on felix, 0.20 degrades).
        gap_frac:      max interpolated run per side, in seconds, before abstain.
        eps_rel:       relative deadband factor on the ball's own |dx| median.
        eps_floor_frac: ABSOLUTE deadband floor as a fraction of frame_width
                       (noise quantum; ~2.9 px/f at 1920). felix-provisional.
        min_netdx_frac: minimum net horizontal travel per side, as a fraction of
                       frame_width, before a sign is trusted. felix-provisional.

    Returns:
        (kept_events, direction_state) where kept_events is the filtered list of
        (frame, x, y) and direction_state maps frame -> {"metric",
        "unconfirmed", "rejected"} (the rejected entries are NOT in kept_events
        but are reported for diagnostics).
    """
    n = len(raw_centers)
    if n == 0 or not bounce_events:
        return list(bounce_events), {}

    half = max(2, round(fps * half_frac))
    gap_thr = round(fps * gap_frac)
    eps_floor = eps_floor_frac * frame_width
    min_netdx = min_netdx_frac * frame_width

    kept = []
    state = {}
    n_reject = 0
    for ev in bounce_events:
        f = int(ev[0])

        # gap guard PER SIDE: never read direction through a fabricated spline gap.
        pre_gap = _veto_max_gap(is_real, f - half, f)
        post_gap = _veto_max_gap(is_real, f, f + half)
        if pre_gap > gap_thr or post_gap > gap_thr:
            kept.append(ev)
            state[f] = "unconfirmed"
            continue

        pre, pre_tp = _veto_drop_teleports(
            _veto_collect_side(raw_centers, is_real, f - half, f), frame_width)
        post, post_tp = _veto_drop_teleports(
            _veto_collect_side(raw_centers, is_real, f, f + half), frame_width)

        # a teleport dropped inside either window → track unreliable → ABSTAIN
        # (don't fit through the hole; a teleport-and-stay would fake a flip).
        if pre_tp or post_tp:
            kept.append(ev)
            state[f] = "unconfirmed"
            continue

        vx_pre = _veto_slope(pre, 1)
        vx_post = _veto_slope(post, 1)
        if vx_pre is None or vx_post is None:
            kept.append(ev)            # too sparse (e.g. f424 occluded) → abstain
            state[f] = "unconfirmed"
            continue

        # deadband from the ball's own horizontal speed, with an ABSOLUTE floor so
        # tracker noise on a near-vertical bounce can't clear it (see SAFETY NOTE).
        allp, _allp_tp = _veto_drop_teleports(
            _veto_collect_side(raw_centers, is_real, f - half, f + half), frame_width)
        steps = [abs(allp[i][1] - allp[i - 1][1])
                 for i in range(1, len(allp))
                 if allp[i][0] - allp[i - 1][0] == 1]
        speed_scale = float(np.median(steps)) if steps else 0.0
        eps = max(eps_rel * speed_scale, eps_floor)

        # net horizontal travel each side (fit-window endpoints): a real sign needs
        # the ball to actually cross horizontally, not just jitter near vertical.
        netdx_pre = abs(pre[-1][1] - pre[0][1])
        netdx_post = abs(post[-1][1] - post[0][1])

        if (abs(vx_pre) <= eps or abs(vx_post) <= eps
                or netdx_pre < min_netdx or netdx_post < min_netdx):
            kept.append(ev)            # horizontal ~ still → can't judge → abstain
            state[f] = "unconfirmed"
            continue

        if np.sign(vx_pre) != np.sign(vx_post):
            n_reject += 1              # confident vx flip → a hit, not a bounce
            state[f] = "rejected"
            continue

        # sign preserved → a real bounce. vy flip (down then up) confirms "metric".
        vy_pre = _veto_slope(pre, 2)
        vy_post = _veto_slope(post, 2)
        vy_flip = (vy_pre is not None and vy_post is not None
                   and vy_pre > 0 and vy_post < 0)
        kept.append(ev)
        state[f] = "metric" if vy_flip else "unconfirmed"

    logger.info(f"vx-flip veto: {len(bounce_events)} candidates -> "
                f"{len(kept)} kept ({n_reject} rejected as hits)")
    return kept, state


# ─────────────────── bounce↔hit collision arbitration (legacy) ───────────────────
# The legacy path detects bounces (incl. recovered turn-angle apex candidates) and
# hits (wrist-speed peaks that passed the ball↔wrist proximity gate) INDEPENDENTLY.
# When a bounce candidate lands within a contact window of a confident hit, the two
# claim the same frame. The cardinal error to avoid is confusion_H→B — a real racket
# HIT shown as a BOUNCE (the user's #1 bug). A proximity-gated hit is a strong
# "contact-near-a-player" signal, so on a collision we KEEP THE HIT and DEMOTE THE
# BOUNCE. This is the direction+proximity arbitration the legacy path lacked (the old
# bounce_guard only went the other way — it suppressed hits near bounces, which on a
# mis-recovered turn bounce would DELETE the real hit and CREATE confusion_H→B).
#
# Measured motivation: on the LIVE Kalman track the turn pass recovered a bounce at
# f82 colliding with GT shot 81 — without this arbitration the bounce_guard would
# suppress the f81 hit and emit only the bounce → confusion_H→B=1. With it, the hit
# wins and the spurious bounce is dropped. Pure function, scale-free window.

def arbitrate_bounce_hit(bounce_events, hits, fps, turn_frames=None,
                         guard_frac=0.10):
    """CONFIDENCE-AWARE bounce↔hit collision resolution for the legacy path.

    Two independent detectors can claim the same frame. We resolve by CONFIDENCE,
    because a blind "hit always wins" loses real bounces (a clean y-V bounce with a
    follow-through wrist peak nearby) and a blind "bounce always wins" creates
    confusion_H→B (a mis-recovered turn bounce sitting on a real contact):

      - A bounce in ``turn_frames`` (recovered ONLY from the velocity-turn angle —
        no y-extremum → lower confidence) that collides with a hit is DROPPED: the
        proximity-gated hit (a contact near a player) outranks it. Keeps
        confusion_H→B at 0 when the apex recovery fires near a real swing.
      - A bounce NOT in turn_frames (a clean y-extremum V → high confidence) WINS:
        it is kept, and the colliding hit is SUPPRESSED (that wrist peak is the
        follow-through of the shot that produced the bounce, not a new contact).
      - Non-colliding bounces and hits pass through unchanged.

    With turn_frames=None every bounce is treated as high-confidence (the old
    bounce-wins behaviour), so callers that don't thread provenance are unaffected.

    Args:
        bounce_events: list of (frame, x, y) bounce candidates.
        hits:          list of hit dicts (each with a "frame"); proximity-gated.
        fps:           frame rate (scale-free window).
        turn_frames:   iterable of bounce frames sourced only from the turn pass
                       (lower confidence). None → treat all bounces as y-V.
        guard_frac:    collision window in seconds (default 0.10).

    Returns:
        (kept_bounces, kept_hits).
    """
    if not bounce_events:
        return list(bounce_events), list(hits)
    guard = max(1, round(fps * guard_frac))
    turn_set = set(int(f) for f in (turn_frames or []))
    hit_frames = [int(h["frame"]) for h in hits]

    kept_bounces = []
    suppressed_hit_frames = set()
    n_demoted = 0
    for ev in bounce_events:
        bf = int(ev[0])
        colliding = [hf for hf in hit_frames if abs(bf - hf) <= guard]
        if colliding and bf in turn_set:
            n_demoted += 1                    # low-conf turn bounce loses to the hit
            continue
        kept_bounces.append(ev)
        if colliding:                         # high-conf bounce wins → drop its hits
            suppressed_hit_frames.update(colliding)
    kept_hits = [h for h in hits if int(h["frame"]) not in suppressed_hit_frames]
    if n_demoted or suppressed_hit_frames:
        logger.info(f"bounce↔hit arbitration: {n_demoted} turn-bounce(s) demoted to "
                    f"hit, {len(suppressed_hit_frames)} hit(s) suppressed near a "
                    f"y-extremum bounce")
    return kept_bounces, kept_hits
