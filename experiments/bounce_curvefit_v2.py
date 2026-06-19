"""
Curve-fit bounce detector v2.

Improvements over v1 (driven by per-miss diagnosis on felix.mp4):
- Adaptive per-side fitting: each side (descending / ascending) is fit over the
  longest run of consecutive tracked frames up to `win`, so a late or steep
  return doesn't break the fit (fixes GT 721, 978).
- Energy-restitution gate (optional): a real floor bounce loses vertical energy,
  so the ascending slope magnitude should not greatly EXCEED the descending one;
  a player HIT injects energy (ascending much steeper). Gate on the ratio.
- Robust local-max detection with a small flat-top tolerance (fixes plateaus
  like GT 53).
- Asymmetry tolerance: require a strong V on at least the descending side and a
  *present* return on the ascending side, rather than symmetric strictness.

All thresholds are ratios of fps / frame_height (zero per-video hardcoding).
"""
import numpy as np


def _fit_line(xs, ys):
    if len(xs) < 2:
        return None
    A = np.vstack([xs, np.ones(len(xs))]).T
    (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = a * xs + b
    rmse = float(np.sqrt(np.mean((ys - pred) ** 2)))
    return a, b, rmse


def _collect_side(ys, n, i, win, direction):
    """Collect up to `win` consecutive tracked points on one side of i.
    direction=-1 -> left (before), +1 -> right (after). Includes i itself.
    Stops at the first gap to keep the fit on a single ballistic segment."""
    xs, vals = [], []
    if direction < 0:
        rng = range(i, max(i - win - 1, -1), -1)
    else:
        rng = range(i, min(i + win + 1, n))
    for k in rng:
        if 0 <= k < n and not np.isnan(ys[k]):
            xs.append(k)
            vals.append(ys[k])
        else:
            break  # stop at gap: keep a clean ballistic segment
    order = np.argsort(xs)
    return np.array(xs, float)[order], np.array(vals, float)[order]


def detect_bounces_curvefit_v2(ball_centers, fps, frame_height, frame_width=None,
                               win=None, min_drop=None, min_gap=None,
                               min_y_ratio=0.12, max_y_ratio=0.95,
                               max_side_rmse=None, min_slope=None,
                               restitution_max=2.2,
                               player_feet=None, player_gate_frac=0.0):
    """
    Returns list of (frame_idx, x, y).

    player_feet: optional list (per frame) of [(x,y), ...] player feet/contact
      points; if a candidate is within player_gate_frac * frame_height of any
      player point at that frame, it's rejected as a likely hit. Disabled when
      player_gate_frac<=0.
    restitution_max: reject if |ascending slope| > restitution_max * |descending|
      (player hit injects energy). Set high to disable.
    """
    n = len(ball_centers)
    if n < 7:
        return []

    win = win if win is not None else max(4, int(fps * 0.17))
    min_gap = min_gap if min_gap is not None else int(fps * 0.30)
    min_drop = min_drop if min_drop is not None else frame_height * 0.012
    max_side_rmse = max_side_rmse if max_side_rmse is not None else frame_height * 0.030
    min_slope = min_slope if min_slope is not None else frame_height * 0.0008

    ys = np.array([c[1] if c is not None else np.nan for c in ball_centers], dtype=float)
    xs = np.array([c[0] if c is not None else np.nan for c in ball_centers], dtype=float)

    candidates = []
    for i in range(2, n - 2):
        if ball_centers[i] is None or np.isnan(ys[i]):
            continue
        yc = ys[i]
        if not (min_y_ratio <= yc / frame_height <= max_y_ratio):
            continue

        # robust local maximum of y over ±2 (flat-top tolerant)
        lo, hi = max(0, i - 2), min(n, i + 3)
        local = ys[lo:hi]
        local = local[~np.isnan(local)]
        if len(local) == 0 or yc < np.nanmax(local) - 0.5:
            continue

        lx, ly = _collect_side(ys, n, i, win, -1)
        rx, ry = _collect_side(ys, n, i, win, +1)
        if len(lx) < 3 or len(rx) < 3:
            continue

        lf = _fit_line(lx, ly)
        rf = _fit_line(rx, ry)
        if lf is None or rf is None:
            continue
        a_l, b_l, rmse_l = lf
        a_r, b_r, rmse_r = rf

        # descending before (a_l>0), ascending after (a_r<0)
        if a_l < min_slope or a_r > -min_slope:
            continue
        if rmse_l > max_side_rmse or rmse_r > max_side_rmse:
            continue

        swing_l = abs(a_l) * (lx[-1] - lx[0])
        swing_r = abs(a_r) * (rx[-1] - rx[0])
        # descending side must show a real drop; ascending must be present (softer)
        if swing_l < min_drop or swing_r < min_drop * 0.5:
            continue

        # energy-restitution gate: player hit injects energy (steep rebound)
        if abs(a_r) > restitution_max * abs(a_l):
            continue

        # refine via line intersection, clamp near candidate
        if abs(a_l - a_r) > 1e-6:
            x_int = (b_r - b_l) / (a_l - a_r)
        else:
            x_int = float(i)
        if not (i - win <= x_int <= i + win):
            x_int = float(i)
        bf = max(0, min(n - 1, int(round(x_int))))

        # optional player-proximity gate
        if player_gate_frac > 0 and player_feet is not None and bf < len(player_feet):
            bx_ = xs[bf] if not np.isnan(xs[bf]) else xs[i]
            by_ = ys[bf] if not np.isnan(ys[bf]) else yc
            gate = player_gate_frac * frame_height
            too_close = False
            for (pxx, pyy) in (player_feet[bf] or []):
                if np.hypot(bx_ - pxx, by_ - pyy) < gate:
                    too_close = True
                    break
            if too_close:
                continue

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
    return chosen
