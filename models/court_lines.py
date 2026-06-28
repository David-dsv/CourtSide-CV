"""
Classical white-line extraction + vanishing-point pencils for automatic court
detection (no GPU, no learned model, SaaS-license-clean).

This is the front end of the classical auto-homography path (models/court_auto.py).
It reimplements the license-clean parts of the Farin / gchlebus
(BSD-3-Clause, github.com/gchlebus/tennis-court-detection) court-line pipeline,
plus the two reusable ideas from arXiv:2404.06977 (Agrawal et al., "Accurate
Tennis Court Line Detection on Amateur Recorded Matches"): the 7x7 court-color
line test and the warp-back self-score (the latter lives in court_auto.py).
Algorithms are not copyrightable; no GPL/AGPL/no-license code is used here.

Three stages:
  1. white_line_mask(frame)        -> binary mask of court-line pixels
  2. detect_line_candidates(mask)  -> merged maximal line segments
  3. estimate_vp_pencils(lines)    -> the (<=2) vanishing-point pencils + gate

Stage 3 is also the felix-grazing safety gate: a full court homography needs
TWO vanishing points (one per ITF line direction). On a grazing-oblique angle
(camera near-parallel to a sideline, e.g. felix.mp4) the depth-axis lines are
projected so flat they collapse into the single lateral pencil — only one VP is
recoverable, which cannot constrain an 8-DOF homography. estimate_vp_pencils
then reports `n_vp_capable_families < 2` and the caller ABSTAINS (the honest
answer; the semi-auto 4-corner sidecar remains felix's metric path).

ZERO-HARDCODING: every spatial threshold is a fraction of the frame diagonal or
a percentile. The only absolute constants are (a) the HSV achromatic-white
saturation ceiling (S<60/255 is the intrinsic colour definition of "white",
frame-size independent) and (b) angular bands in degrees (orientation is
scale-invariant). All ratios are the values MEASURED live on the three probe
clips (tennis / felix / djokovic) during the S4 grounding pass — see
docs/research/s4-auto-homography-CR.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2

# ── measured-grounded constants (fractions of diag / percentiles) ──
MASK_V_BRIGHT_PCTL = 80            # bright court paint: V above the 80th pctl
MASK_S_WHITE_CEIL = 60             # achromatic "white": S < 60/255 (intrinsic, not scaled)
TOPHAT_KERNELS_FRAC_DIAG = (0.001362, 0.00227, 0.003178)  # {3,5,7}px @ diag 2203
TOPHAT_RIDGE_PCTL = 95             # thin ridge above the 95th pctl of the top-hat
MASK_COVERAGE_SANE_BAND = (0.01, 0.35)   # abstain if <1% (starved) or >35% (clay flood)

HOUGH_THRESHOLD_FRAC_DIAG = 0.05   # accumulator votes
HOUGH_MIN_LEN_FRAC_DIAG = 0.10     # a court line spans a good chunk of the frame
HOUGH_MAX_GAP_FRAC_DIAG = 0.01
MERGE_DTHETA_DEG = 2.0             # near-duplicate merge: orientation
MERGE_DRHO_FRAC_DIAG = 0.005       # near-duplicate merge: offset

PENCIL_SEED_DTHETA_MAX_DEG = 45.0  # angle-coherent RANSAC seed pair window
PENCIL_INLIER_DTHETA_MAX_DEG = 50.0
PENCIL_MIN_LINES = 3               # a real pencil needs >=3 lines (2 define a VP trivially)
PENCIL_MIN_FRAC_OF_POOL = 0.05     # a PRINCIPAL pencil holds >=5% of merged lines
VP_RANSAC_INLIER_BAND_FRAC_DIAG = 0.02

# honest all-member spread bands (median perpendicular residual to the refit VP)
SPREAD_TIGHT_HIGH_FRAC_DIAG = 0.05   # both pencils below -> HIGH-eligible
SPREAD_LOOSE_LOW_FRAC_DIAG = 0.25    # above this -> abstain
VP_SEPARATION_MIN_FRAC_DIAG = 0.40   # the two principal VPs must be well separated
# The SECOND ITF direction (depth-axis sidelines) must carry real court-spanning
# length. On a grazing angle a few short parasitic oblique segments (net/fence/
# player edges) can fake a second pencil, but their total length is a small
# fraction of the dominant direction's. Measured second/first sum-of-length ratio:
# tennis 0.50, djokovic 0.49, felix 0.11 — a clean, scale-free felix discriminator.
SECOND_DIR_LENGTH_RATIO_MIN = 0.25   # second pencil >= 25% of the first's total line length


@dataclass
class Line:
    """A maximal merged line candidate in image space.

    Stored both as endpoints and as the homogeneous line coefficients (a,b,c)
    with a^2+b^2=1, so a point p=(x,y,1) lies on the line iff a*x+b*y+c=0 and
    the signed point-line distance is just a*x+b*y+c.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    support: int = 0           # mask pixels backing this line (the scoring currency)
    coef: tuple = (0.0, 0.0, 0.0)
    theta: float = 0.0         # orientation in [0,180) degrees

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def midpoint(self) -> tuple:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Pencil:
    """A set of image lines that (in a real court) concur at one vanishing point."""
    lines: list = field(default_factory=list)
    vp: Optional[tuple] = None          # (x,y) in image px, or None if at infinity
    spread_px: float = float("inf")     # HONEST all-member median perpendicular residual
    mean_theta: float = 0.0

    @property
    def total_length(self) -> float:
        """Sum of member-line lengths — court-spanning length of this direction."""
        return float(sum(ln.length for ln in self.lines))


# ───────────────────────────── stage 1: mask ──────────────────────────────

def white_line_mask(frame: np.ndarray, *, use_court_color: bool = True) -> np.ndarray:
    """Binary (uint8 0/255) mask of court-line pixels.

    Combines three license-clean detectors (OR), so a line survives if ANY fires:
      * bright achromatic paint  : V > p80(V) AND S < 60/255
      * thin ridge (morph top-hat at {3,5,7}px scales > p95)  — kills broad bright
        blobs (jerseys, sky, banners) while keeping pencil-thin lines
      * 7x7 court-color test (arXiv:2404.06977): the modal HSV of ~1000 random
        samples is the court colour; a pixel is a line iff it does NOT match the
        court colour AND >=4 of its 7x7 neighbours DO (a thin non-court streak on
        a court-coloured background) — robust to clay/grass/shadow with no
        per-video threshold.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    diag = float(np.hypot(w, h))

    # (a) bright achromatic paint
    v_thr = float(np.percentile(V, MASK_V_BRIGHT_PCTL))
    m_bright = (V > v_thr) & (S < MASK_S_WHITE_CEIL)

    # (b) multi-scale white top-hat ridge (bright thin structures only)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m_ridge = np.zeros((h, w), dtype=bool)
    for frac in TOPHAT_KERNELS_FRAC_DIAG:
        k = max(3, int(round(frac * diag)) | 1)   # odd kernel >=3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        if th.max() > 0:
            m_ridge |= th > float(np.percentile(th[th > 0], TOPHAT_RIDGE_PCTL))

    mask_bool = m_bright | m_ridge

    # (c) 7x7 court-color test (arXiv:2404.06977) — additive, OR-combined
    if use_court_color:
        mask_bool |= _court_color_line_mask(hsv)

    return (mask_bool.astype(np.uint8)) * 255


def _court_color_line_mask(hsv: np.ndarray, n_samples: int = 1000,
                           rng_seed: int = 0) -> np.ndarray:
    """arXiv:2404.06977 7x7 court-color line test.

    court_color = modal HSV over n_samples random pixels; a pixel is a line iff
    it is far from court_color while >=4 of its 7x7 neighbours are close to it.
    Implemented vectorised: a binary "is court colour" map, box-filtered over 7x7
    to count court-coloured neighbours, then (not-court center) AND (>=4 court
    neighbours).
    """
    h, w = hsv.shape[:2]
    rng = np.random.default_rng(rng_seed)
    ys = rng.integers(0, h, size=n_samples)
    xs = rng.integers(0, w, size=n_samples)
    samples = hsv[ys, xs].astype(np.int16)
    # modal court colour via a coarse HSV histogram (robust to noise)
    quant = (samples // np.array([16, 32, 32])).astype(np.int64)
    keys, counts = np.unique(quant, axis=0, return_counts=True)
    court_q = keys[int(np.argmax(counts))]
    court_color = (court_q * np.array([16, 32, 32]) + np.array([8, 16, 16])).astype(np.int16)

    # tolerance = robust spread of the dominant cluster (no magic px), floored
    diff_all = np.abs(samples - court_color)
    tau = np.maximum(np.percentile(diff_all, 60, axis=0), np.array([8, 24, 24]))

    hsv_i = hsv.astype(np.int16)
    is_court = np.all(np.abs(hsv_i - court_color) <= tau, axis=2).astype(np.float32)
    # count court-coloured neighbours in a 7x7 window (box filter * 49)
    neigh = cv2.boxFilter(is_court, ddepth=-1, ksize=(7, 7), normalize=False)
    center_is_court = is_court > 0.5
    return (~center_is_court) & (neigh >= 4.0)


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of pixels in the white-line mask (sanity-band gate input)."""
    return float((mask > 0).mean())


# ─────────────────────── stage 2: line candidates ─────────────────────────

def detect_line_candidates(mask: np.ndarray) -> list:
    """Merged maximal line segments from the white-line mask via HoughLinesP +
    near-duplicate fusion. Each surviving Line carries its mask support count
    (the currency the warp-back scorer trusts).
    """
    h, w = mask.shape[:2]
    diag = float(np.hypot(w, h))
    raw = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 180.0,
        threshold=max(20, int(HOUGH_THRESHOLD_FRAC_DIAG * diag)),
        minLineLength=int(HOUGH_MIN_LEN_FRAC_DIAG * diag),
        maxLineGap=max(2, int(HOUGH_MAX_GAP_FRAC_DIAG * diag)),
    )
    if raw is None:
        return []
    segs = [tuple(map(float, s[0])) for s in raw]
    lines = [_segment_to_line(s, mask) for s in segs]
    return _merge_collinear(lines, diag)


def _segment_to_line(seg, mask) -> Line:
    x1, y1, x2, y2 = seg
    a, b, c = _line_coef(x1, y1, x2, y2)
    theta = (np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
    ln = Line(x1, y1, x2, y2, coef=(a, b, c), theta=theta)
    ln.support = _line_support(ln, mask)
    return ln


def _line_coef(x1, y1, x2, y2):
    """Homogeneous (a,b,c) with a^2+b^2=1 from two endpoints."""
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    n = float(np.hypot(a, b)) + 1e-12
    return a / n, b / n, c / n


def _line_support(ln: Line, mask, band_frac_diag: float = 0.0015) -> int:
    """Count mask pixels within a thin band of the line over its extent."""
    h, w = mask.shape[:2]
    diag = float(np.hypot(w, h))
    band = max(1.0, band_frac_diag * diag)
    n = max(2, int(ln.length))
    xs = np.linspace(ln.x1, ln.x2, n)
    ys = np.linspace(ln.y1, ln.y2, n)
    cnt = 0
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            x0, x1 = max(0, xi - 1), min(w, xi + 2)
            y0, y1 = max(0, yi - 1), min(h, yi + 2)
            if mask[y0:y1, x0:x1].any():
                cnt += 1
    return cnt


def _merge_collinear(lines: list, diag: float) -> list:
    """Cluster near-duplicate segments (dtheta<=2deg, drho<=0.5%diag) and refit
    each cluster by total-least-squares over its endpoints. Returns one maximal
    Line per cluster, endpoints spanning the union extent."""
    drho = MERGE_DRHO_FRAC_DIAG * diag
    used = [False] * len(lines)
    merged = []
    for i, li in enumerate(lines):
        if used[i]:
            continue
        cluster = [li]
        used[i] = True
        rho_i = _rho(li)
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            lj = lines[j]
            if _angdiff(li.theta, lj.theta) <= MERGE_DTHETA_DEG and \
               abs(rho_i - _rho(lj)) <= drho:
                cluster.append(lj)
                used[j] = True
        merged.append(_refit_cluster(cluster))
    return merged


def _rho(ln: Line) -> float:
    """Signed distance from origin to the line (the Hough rho)."""
    return -ln.coef[2]


def _angdiff(a, b) -> float:
    """Smallest difference between two orientations in [0,180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _refit_cluster(cluster: list) -> Line:
    """Total-least-squares line through all cluster endpoints; endpoints =
    projections of the extreme endpoints onto the fitted direction."""
    pts = []
    support = 0
    for ln in cluster:
        pts.append((ln.x1, ln.y1))
        pts.append((ln.x2, ln.y2))
        support += ln.support
    pts = np.array(pts, dtype=np.float64)
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    # project endpoints onto the direction to find the extent
    t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    tmin, tmax = float(t.min()), float(t.max())
    x1, y1 = x0 + tmin * vx, y0 + tmin * vy
    x2, y2 = x0 + tmax * vx, y0 + tmax * vy
    a, b, c = _line_coef(x1, y1, x2, y2)
    theta = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
    return Line(float(x1), float(y1), float(x2), float(y2),
                support=support, coef=(a, b, c), theta=float(theta))


# ──────────────── stage 3: VP pencils + abstention gate ────────────────────

def _vanishing_point(lines: list):
    """VP = the point minimizing sum of squared signed point-line distances, i.e.
    the smallest right singular vector of the stacked [a b c] coefficient matrix.
    Returns (x,y) in image px, or None if at infinity (near-parallel)."""
    A = np.array([ln.coef for ln in lines], dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    v = Vt[-1]
    if abs(v[2]) < 1e-9:
        return None
    return (v[0] / v[2], v[1] / v[2])


def _all_member_spread(lines: list, vp) -> float:
    """HONEST spread = median perpendicular residual of EVERY member line to the
    VP (never the cherry-picked RANSAC-inlier subset — the documented djokovic
    optimism trap). For a line (a,b,c) and point (vx,vy,1) the perpendicular
    residual is |a*vx + b*vy + c| (coeffs are unit-normalized)."""
    if vp is None:
        return float("inf")
    vx, vy = vp
    res = [abs(a * vx + b * vy + c) for (a, b, c) in (ln.coef for ln in lines)]
    return float(np.median(res)) if res else float("inf")


def _intersect(l1: Line, l2: Line):
    """Image-space intersection of two lines (their candidate shared VP), via the
    cross product of homogeneous coefficients. None if (near-)parallel."""
    a1, b1, c1 = l1.coef
    a2, b2, c2 = l2.coef
    x = b1 * c2 - b2 * c1
    y = c1 * a2 - c2 * a1
    w = a1 * b2 - a2 * b1
    if abs(w) < 1e-9:
        return None
    return (x / w, y / w)


def estimate_vp_pencils(lines: list, frame_wh: tuple, *, rng_seed: int = 0) -> dict:
    """Group lines into VP-coherent pencils by sequential vanishing-point RANSAC,
    then refit each pencil's VP and measure its HONEST all-member spread.

    A pencil is a set of image lines that concur at one vanishing point (NOT a
    set of similarly-oriented lines — perspective fans a single ITF direction's
    lines across a wide orientation range, and two lines of equal orientation may
    belong to different ITF directions). So pencils are found by VP consensus:
    repeatedly sample line pairs, take their intersection as a candidate VP,
    count lines whose perpendicular residual to that VP is within the inlier
    band, keep the best-consensus VP, peel its inliers off, and repeat. This is
    the geometrically-correct discriminator the live probes converged on.

    Returns {pencils, principal, n_vp_capable_families, ...}. A pencil is
    VP-CAPABLE only with >=PENCIL_MIN_LINES members; "principal" if it also holds
    >=PENCIL_MIN_FRAC_OF_POOL of all merged lines (so a stray singleton/pair
    can't masquerade as a second family — the felix mis-count trap).
    """
    w, h = frame_wh
    diag = float(np.hypot(w, h))
    band = VP_RANSAC_INLIER_BAND_FRAC_DIAG * diag
    rng = np.random.default_rng(rng_seed)
    pool = sorted(lines, key=lambda L: -L.support)   # best-supported first
    remaining = list(range(len(pool)))
    pencils = []

    while len(remaining) >= PENCIL_MIN_LINES:
        best_inliers = []
        best_vp = None
        rem = remaining
        n = len(rem)
        # try all pairs if few, else a capped random sample of pairs
        pairs = []
        if n <= 24:
            pairs = [(rem[i], rem[j]) for i in range(n) for j in range(i + 1, n)]
        else:
            for _ in range(400):
                i, j = rng.choice(n, size=2, replace=False)
                pairs.append((rem[i], rem[j]))
        for (pi, pj) in pairs:
            li, lj = pool[pi], pool[pj]
            # the two seed lines must share an orientation neighbourhood (a real
            # pencil's lines do not span arbitrary angles) — cheap pre-filter
            if _angdiff(li.theta, lj.theta) > PENCIL_SEED_DTHETA_MAX_DEG:
                continue
            vp = _intersect(li, lj)
            if vp is None:
                continue
            inl = [k for k in rem
                   if abs(pool[k].coef[0] * vp[0] + pool[k].coef[1] * vp[1]
                          + pool[k].coef[2]) <= band]
            if len(inl) > len(best_inliers):
                best_inliers, best_vp = inl, vp
        if len(best_inliers) < PENCIL_MIN_LINES:
            break
        members = [pool[k] for k in best_inliers]
        vp = _vanishing_point(members)            # least-squares refit on inliers
        pencils.append(Pencil(
            lines=members, vp=vp,
            spread_px=_all_member_spread(members, vp),
            mean_theta=float(np.median([m.theta for m in members])),
        ))
        inlier_set = set(best_inliers)
        remaining = [k for k in remaining if k not in inlier_set]

    n_pool = max(1, len(pool))
    vp_capable = [p for p in pencils if len(p.lines) >= PENCIL_MIN_LINES]
    candidates = sorted(
        [p for p in vp_capable if len(p.lines) >= PENCIL_MIN_FRAC_OF_POOL * n_pool],
        key=lambda p: -len(p.lines))
    # The two PRINCIPAL pencils must be the court's two distinct ITF line
    # DIRECTIONS — not two sub-consensuses of a single collapsed direction. On a
    # grazing angle (felix) every line is near-horizontal, so VP-RANSAC can split
    # that one orientation band into two same-orientation pencils; that is NOT two
    # vanishing directions. So direction B is the largest candidate whose mean
    # orientation differs from direction A by more than PENCIL_SEED_DTHETA_MAX_DEG
    # (the same window two seed lines of ONE pencil are allowed to span). felix's
    # second "pencil" is ~1deg from the first -> rejected -> 1 distinct family.
    principal = []
    if candidates:
        a = candidates[0]
        principal = [a]
        for p in candidates[1:]:
            if _angdiff(a.mean_theta, p.mean_theta) > PENCIL_SEED_DTHETA_MAX_DEG:
                principal.append(p)
                break

    vp_sep = orient_sep = second_dir_len_ratio = None
    if len(principal) == 2 and principal[0].vp and principal[1].vp:
        vp_sep = float(np.hypot(principal[0].vp[0] - principal[1].vp[0],
                                principal[0].vp[1] - principal[1].vp[1]))
        orient_sep = _angdiff(principal[0].mean_theta, principal[1].mean_theta)
        # second direction's court-spanning length relative to the first's — the
        # felix catcher (a real depth axis carries real length; parasites don't)
        l0 = principal[0].total_length
        second_dir_len_ratio = (principal[1].total_length / l0) if l0 > 0 else None

    return {
        "pencils": pencils,
        "principal": principal,
        # n_vp_capable_families now counts DISTINCT-direction principal families
        # (the gate's primary signal), not every RANSAC sub-pencil.
        "n_vp_capable_families": len(vp_capable),
        "n_principal_families": len(principal),
        "orientation_separation_deg": orient_sep,
        "vp_separation_px": vp_sep,
        "second_dir_length_ratio": second_dir_len_ratio,
        "diag_px": diag,
        "n_merged_lines": len(lines),
    }


def abstention_verdict(vp_info: dict) -> dict:
    """Four ORTHOGONAL vetoes; ALL must pass to even attempt a homography
    (failing any -> ABSTAIN). felix is caught by (1) AND (2) independently.
    Returns {abstain, reason, gate_features} for logging + the CR.

    (1) two distinct directions : 2 PRINCIPAL pencils whose mean orientations
        differ (the two ITF line directions exist, not one collapsed band).
    (2) second-direction LENGTH : the depth-axis pencil carries >=25% of the
        dominant direction's total line length [PRIMARY felix catcher — felix's
        spurious 2nd pencil is short parasitic segments, ratio 0.11 << 0.25].
    (3) VP-separation           : the two VPs are well separated (>=0.40*diag),
        neither at-infinity (a near-parallel pair = degenerate perspective).
    (4) honest spread           : both principal spreads < SPREAD_LOOSE_LOW
        (0.25*diag), using the ALL-MEMBER spread (never the inlier-subset).
    A 5th independent confirmation (projected-quad aspect) runs in court_auto on
    the solved H — it needs a homography, so it can't live here.
    """
    diag = vp_info["diag_px"]
    principal = vp_info["principal"]
    feats = {
        "n_vp_capable_families": vp_info["n_vp_capable_families"],
        "n_principal_families": vp_info["n_principal_families"],
        "orientation_separation_deg": vp_info.get("orientation_separation_deg"),
        "second_dir_length_ratio": vp_info.get("second_dir_length_ratio"),
        "vp_separation_px": vp_info["vp_separation_px"],
        "vp_separation_frac_diag": (vp_info["vp_separation_px"] / diag
                                    if vp_info["vp_separation_px"] else None),
        "principal_spreads_frac_diag": [round(p.spread_px / diag, 4) for p in principal],
    }

    # (1) two distinct ITF directions
    if vp_info["n_principal_families"] < 2:
        return {"abstain": True, "reason":
                f"only {vp_info['n_principal_families']} distinct-direction "
                f"principal pencil(s) (need 2) — depth axis collapsed into the "
                f"lateral pencil (grazing/oblique)", "gate_features": feats}

    # (2) the second direction must carry real court-spanning length [felix catcher]
    ratio = vp_info.get("second_dir_length_ratio")
    if ratio is None or ratio < SECOND_DIR_LENGTH_RATIO_MIN:
        return {"abstain": True, "reason":
                f"second-direction line length ratio "
                f"{ratio if ratio is None else round(ratio,3)} < "
                f"{SECOND_DIR_LENGTH_RATIO_MIN} — the apparent depth axis is short "
                f"parasitic segments, not court-spanning sidelines (grazing)",
                "gate_features": feats}

    # (3) VP separation / conditioning
    if vp_info["vp_separation_px"] is None or \
       vp_info["vp_separation_px"] < VP_SEPARATION_MIN_FRAC_DIAG * diag:
        return {"abstain": True, "reason":
                f"vanishing points too close / near-parallel "
                f"({feats['vp_separation_frac_diag']:.2f} diag < "
                f"{VP_SEPARATION_MIN_FRAC_DIAG}) — degenerate perspective",
                "gate_features": feats}

    # (4) honest all-member spread
    worst = max(p.spread_px for p in principal)
    if worst >= SPREAD_LOOSE_LOW_FRAC_DIAG * diag:
        return {"abstain": True, "reason":
                f"pencil spread {worst/diag:.3f} diag >= "
                f"{SPREAD_LOOSE_LOW_FRAC_DIAG} — lines disagree on the VP",
                "gate_features": feats}

    return {"abstain": False, "reason": "two well-conditioned VP pencils",
            "gate_features": feats}
