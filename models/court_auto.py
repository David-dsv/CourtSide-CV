"""
Automatic court homography from a single frame — classical, no-GPU, SaaS-clean.

This is the deployable replacement for the no-LICENSE keypoint model
(yastrebksv/TennisCourtDetector) on oblique / amateur / court-level framings,
where that model domain-shifts to <4 keypoints. It needs no learned weights, no
GPU, and no labelled data — only OpenCV + numpy + the rigid ITF court geometry.

PIPELINE (all stages in models/court_lines.py except the fit, which is here):
  1. white_line_mask          — court-line pixels (bright + ridge + 7x7 court-colour)
  2. detect_line_candidates   — merged maximal Hough segments
  3. estimate_vp_pencils      — the 2 vanishing-point pencils (the 2 ITF directions)
  4. abstention_verdict       — 4 orthogonal vetoes; felix ABSTAINS (returns None)
  5. AutoCourtDetector._fit   — COMBINATORIAL WARP-BACK SELF-SCORE (this file)

The line->ITF-name assignment (the genuinely-hard core the R&D doc flagged) is
solved by the Farin / gchlebus (BSD-3-Clause) recipe, reimplemented in Python:
enumerate (detected line-pair in direction A) x (named ITF line-pair in family X)
x (detected line-pair in direction B) x (named ITF line-pair in family Y); each
4-line choice gives 4 image<->world point correspondences -> a 4-point
homography; warp ALL named ITF lines back into the image and SCORE by how many
sampled points land on the white-line mask (+1 on, -0.5 off, distance-transform
softened); keep the max-score hypothesis. The ITF NAMES come for free from the
winning pairing. VP-ordering of each pencil (monotonic far->near / left->right by
signed distance from the VP) prunes the combinatorics from a blind Cartesian
product to a small set of order-consistent hypotheses.

Output: the EXACT dict contract produced by court_calibrator.homography_from_points
and consumed by run_pipeline_8s.py — {H, H_inv, rms_px, n_used, landmarks,
landmark_px, source, confidence in {high|low|fallback}} — so it is a drop-in for
the existing cascade with zero downstream changes.

License: Farin's method + gchlebus's BSD-3 structure + arXiv:2404.06977's
warp-back score are all reimplemented from the algorithm (algorithms are not
copyrightable); attribution in the CR. No GPL/AGPL/no-license code is used.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import cv2
from loguru import logger

from models.court_lines import (
    white_line_mask, mask_coverage, detect_line_candidates,
    estimate_vp_pencils, abstention_verdict, _angdiff,
    MASK_COVERAGE_SANE_BAND, SPREAD_TIGHT_HIGH_FRAC_DIAG,
)
from models.court_calibrator import (
    WORLD_LANDMARKS, homography_from_points, _assess_conditioning,
    DOUBLES_HALF_W, SINGLES_HALF_W, HALF_LEN, SERVICE_FROM_NET,
)

# ── the ITF line model: each named line as two world-meter endpoints ──
# Two families, matching the two image pencils:
#   WIDTH lines (run left<->right, span X): baselines, service lines, net
#   LENGTH lines (run far<->near, span Y): sidelines, centre service line
# (world frame: origin court centre, +X right, +Y far baseline; meters)
D, S, H, SN = DOUBLES_HALF_W, SINGLES_HALF_W, HALF_LEN, SERVICE_FROM_NET

WIDTH_LINES = {   # ordered far(+Y) -> near(-Y); value = the shared world Y
    "baseline_far":  +H,
    "service_far":   +SN,
    "net":            0.0,
    "service_near":  -SN,
    "baseline_near": -H,
}
LENGTH_LINES = {  # ordered left(-X) -> right(+X); value = the shared world X
    "sideline_left":  -D,
    "singles_left":   -S,
    "center_service":  0.0,
    "singles_right":  +S,
    "sideline_right": +D,
}

# All named ITF line SEGMENTS (world endpoints) for the warp-back self-score.
# Width lines span the full doubles width; the singles/centre lines are clipped
# to their real extent (centre service line only between the two service lines).
ITF_MODEL_LINES = {
    "baseline_far":   ((-D, +H), (+D, +H)),
    "baseline_near":  ((-D, -H), (+D, -H)),
    "service_far":    ((-S, +SN), (+S, +SN)),
    "service_near":   ((-S, -SN), (+S, -SN)),
    "net":            ((-D, 0.0), (+D, 0.0)),
    "sideline_left":  ((-D, -H), (-D, +H)),
    "sideline_right": ((+D, -H), (+D, +H)),
    "singles_left":   ((-S, -H), (-S, +H)),
    "singles_right":  ((+S, -H), (+S, +H)),
    "center_service": ((0.0, -SN), (0.0, +SN)),
}

# The 4 doubles corners (the convexity / aspect conditioning check + emit anchors)
CORNERS_WORLD = {
    "far_bl_left":   (-D, +H), "far_bl_right":  (+D, +H),
    "near_bl_left":  (-D, -H), "near_bl_right": (+D, -H),
}

# warp-back scoring (gchlebus/Farin line-integral, normalized over the WHOLE
# model so a tiny mislocated sliver — most of its model off-frame — scores low)
WARP_SAMPLE_SPACING_M = 0.25       # sample the model lines every 25 cm
WARP_ON_MASK = 1.0                 # in-frame sample landing on a detected line
WARP_OFF_MASK = -0.5               # in-frame sample off any line (off-frame = 0)
WARP_DT_TOL_FRAC_DIAG = 0.004      # a sampled point "hits" within ~0.4% diag of a line
WARP_MIN_INFRAME_FRAC = 0.55       # >=55% of the model court must project in-frame
WARP_MIN_COURT_EXTENT_FRAC_DIAG = 0.35   # the in-frame court must span >=35% of diag
ASPECT_MIN = 0.18                  # projected doubles-quad aspect floor (grazing veto)
ASPECT_MAX = 2.5                   # and a ceiling: a real court projects WIDER than
                                   # tall (elevated view, aspect ~0.3-0.6) up to
                                   # ~1.5 on a steep angle; >2.5x taller-than-wide
                                   # is unphysical (a fence/banner grid, not a court)
# the projected doubles-CORNER quad must span >=this fraction of the frame diag.
# This is the load-bearing anti-collapse gate: a court fitted from too-close lines
# extrapolates its corners to a near-coincident point (spans ~0px) — its aspect
# RATIO can still look sane, but its image EXTENT cannot. A real elevated tennis
# view fills a large fraction of the frame.
QUAD_MIN_EXTENT_FRAC_DIAG = 0.35
# a projected doubles corner may sit at most this far (frac of diag) outside the
# frame. Generous, because a correct elevated view often has the NEAR baseline
# just off the bottom edge; but it rejects the wild off-frame blow-ups (x=-4665)
# of an over-extrapolated / mirrored fit.
CORNER_FRAME_MARGIN_FRAC = 0.55
# perspective sanity: with the camera above the court the NEAR baseline images at
# least this fraction of the FAR baseline's image width (it's actually WIDER, but
# allow slack for near-level angles). A fence/banner fitted as a court inverts
# this (near collapses to a point) -> rejected.
PERSP_NEAR_FAR_MIN_RATIO = 0.85
# line-integral score thresholds. The score is in (-0.5, 1]: +1 per on-line
# in-frame sample, -0.5 per off-line in-frame sample, 0 off-frame, /total. A
# correct full court that lands on lines scores ~1; partial/noisy fits lower.
SCORE_HIGH_FRAC = 0.50             # >= -> HIGH-eligible (least-grounded; see CR risk)
SCORE_LOW_FRAC = 0.20              # below -> abstain

# ── order-preserving correspondence (the Farin / gchlebus core) ──
# Each principal pencil is ONE court direction; its lines, ordered projectively
# (by ray angle around the VP), map to a CONTIGUOUS-or-not but ORDER-PRESERVING
# run of the 5 named ITF lines, in either orientation (the near<->far flip). We
# enumerate every order-preserving injection (an increasing index map into the 5
# named lines) x 2 orientations per pencil — tiny — instead of an arbitrary
# pairing, which is what killed the degenerate net-band collapse.
PENCIL_CORE_DTHETA_DEG = 22.0      # keep only pencil lines within this of the median
                                   # orientation (drops the cross-orientation
                                   # parasites that share a VP by coincidence)
# After re-estimating a direction's VP from its orientation-core, gather ALL
# detected lines converging to it within this perpendicular-residual band — this
# unites the left+right sidelines (parallel in world, one shared VP, but very
# different image orientations) that the RANSAC pencil split across pencils.
VP_GATHER_BAND_FRAC_DIAG = 0.02
VP_RANSAC_BAND_FRAC_DIAG = 0.015   # depth-VP RANSAC inlier band
# a LENGTH (sideline) line is one whose orientation differs from the WIDTH
# direction by more than this — the perpendicular court direction. Below it the
# line is a width line, not a sideline. 35deg is the same window the front-end
# pencil stage uses to call two directions "distinct".
PERP_ORIENT_MIN_DEG = 35.0
DEDUP_POS_TOL_FRAC_DIAG = 0.01     # transversal-position gap below which two
                                   # gathered lines are the same court line
MAX_DETECTED_PER_PENCIL = 5        # cap the injected detected lines (top-K longest)
MIN_NAMED_SPAN_M = 4.0             # the assigned named lines must span >=4 m real
                                   # distance in BOTH directions (else ill-conditioned)
# cross-ratio prune: with >=4 detected lines in a pencil, the cross-ratio of any 4
# (a projective invariant of the parallel ITF lines) must match the ITF cross-ratio
# of the assigned names within this relative tolerance. This nails which detected
# line is which named line even with missing/occluded lines.
CROSS_RATIO_REL_TOL = 0.18
# with only 3 detected lines, fall back to the weaker spacing-ratio order signal:
# the spacing ratio (d01/d12) of the 3 consecutive lines must match the named one.
SPACING_RATIO_REL_TOL = 0.30
MIN_LINES_PER_FAMILY = 2           # >=2 lines per family (2x2 = 4 intersections)
MIN_CORRESPONDENCES = 4            # findHomography needs >=4 image<->world points


class AutoCourtDetector:
    """Classical automatic court homography. Stateless; one call per frame.

    Usage mirrors CourtKeypointDetector:
        det = AutoCourtDetector()
        homo = det.compute_homography(first_frame, frame_diag)   # or None
    """

    def __init__(self, max_lines_per_dir: int = 6):
        # cap candidate lines per direction (top-K by length) so the combinatorial
        # search stays bounded on a 150-line frame (djokovic); the longest lines
        # are the real court lines. K=6 -> C(6,2)^2 * 20 * 20 * 2 ~= 180k vectorised
        # warp-backs, a few seconds for a once-per-clip static-camera solve.
        self.max_lines_per_dir = max_lines_per_dir

    def compute_homography(self, frame, frame_diag=None, *, debug=False):
        h, w = frame.shape[:2]
        diag = float(frame_diag or np.hypot(w, h))

        mask = white_line_mask(frame)
        cov = mask_coverage(mask)
        if not (MASK_COVERAGE_SANE_BAND[0] <= cov <= MASK_COVERAGE_SANE_BAND[1]):
            return self._fallback(f"mask coverage {cov*100:.1f}% out of sane band "
                                  f"{[b*100 for b in MASK_COVERAGE_SANE_BAND]}%")

        lines = detect_line_candidates(mask)
        if len(lines) < 4:
            return self._fallback(f"only {len(lines)} line candidates (<4)")

        vp_info = estimate_vp_pencils(lines, (w, h))
        verdict = abstention_verdict(vp_info)
        if verdict["abstain"]:
            return self._fallback(verdict["reason"], gate=verdict["gate_features"])

        # distance transform of the mask, for the soft warp-back score
        dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

        best = self._fit(vp_info["principal"], lines, dt, diag, (w, h))
        if best is None:
            return self._fallback("no court hypothesis projected onto the line mask",
                                  gate=verdict["gate_features"])

        H, score, landmark_px = best
        H_inv = np.linalg.inv(H)

        # 5th independent confirmation: the solved H must project the doubles
        # rectangle to a convex, non-grazing image quad (reuse the calibrator's
        # conditioning logic verbatim — same scale as the manual sidecar).
        aspect_ok, aspect = self._aspect_quad_ok(H_inv, diag)
        if not aspect_ok:
            return self._fallback(f"projected court aspect {aspect:.2f} < {ASPECT_MIN} "
                                  f"(grazing) — H rejected post-fit",
                                  gate=verdict["gate_features"])

        confidence = self._confidence(score, vp_info, diag)
        if confidence == "fallback":
            return self._fallback(f"warp-back score {score:.2f} < {SCORE_LOW_FRAC} "
                                  f"— court model does not align to the lines",
                                  gate=verdict["gate_features"])

        # package via homography_from_points so RMS / dict shape are IDENTICAL to
        # the manual calibration path (names came from the winning correspondence)
        homo = homography_from_points(landmark_px)
        homo["source"] = "autoline"
        homo["confidence"] = confidence       # our gate, not the 4-pt rms heuristic
        homo["warpback_score"] = float(score)
        homo["gate_features"] = verdict["gate_features"]
        logger.info(f"Auto court (classical): confidence={confidence}, "
                    f"warp-back score={score:.2f}, n_used={homo['n_used']}, "
                    f"aspect={aspect:.2f}")
        if debug:
            homo["_mask"], homo["_lines"], homo["_vp_info"] = mask, lines, vp_info
        return homo

    # ── stage 5: order-preserving correspondence + warp-back self-score ──
    def _fit(self, principal, all_lines, dt, diag, frame_wh):
        """The Farin / gchlebus correspondence core.

        Each of the two principal pencils marks ONE court line DIRECTION. For each
        direction we (re)gather its full membership — ALL detected lines that
        converge to that direction's vanishing point — then order them
        projectively. This is the key robustness step: the RANSAC pencil that
        seeds a direction often holds only the lines on ONE side of the court
        (e.g. only the right sidelines), so a fit from it alone collapses; the two
        real sidelines are PARALLEL in the world (one shared VP) but can have very
        different IMAGE orientations (left sloping up-left, right up-right), so
        orientation alone can't unite them — only VP convergence does.

        The K ordered detected lines of each family then map to an ORDER-
        PRESERVING injection of the 5 named ITF lines, in either orientation (the
        near<->far / left<->right flip) — NOT an arbitrary pairing. A cross-ratio
        prune (a projective invariant of the parallel ITF lines) rejects
        assignments whose detected spacing pattern doesn't match the named one.
        For each surviving (width-assign x length-assign) we build EVERY detected
        width x detected length intersection (>=4 image<->world correspondences)
        and solve with cv2.findHomography. Warp-back self-score tie-breaks among
        the geometrically valid candidates. Returns (H, score, lp) or None.
        """
        self._model_samples = _build_model_samples()   # cached for the run
        self._diag = diag                               # for the extent gate
        self._frame_wh = frame_wh                       # for the corner-sanity gate
        best = None

        # which principal direction is the WIDTH family (ITF lines spanning world
        # X — baselines/service/net) vs the LENGTH family (sidelines, spanning
        # world Y) is unknown a priori -> try BOTH choices. For each, the WIDTH
        # family is that direction's orientation-core; the LENGTH family is
        # RE-DISCOVERED by a vanishing-point RANSAC over every line that ISN'T
        # width-oriented — this unites the left + right sidelines (parallel in
        # world, one shared VP, but very different image orientations) that the
        # seed RANSAC split across pencils.
        best_key = None
        for wseed, lseed in ((principal[0], principal[1]),
                             (principal[1], principal[0])):
            det_w, det_l, tw, tl = self._families(wseed, lseed, all_lines, diag)
            if det_w is None:
                continue
            cand = self._score_assignment(det_w, det_l, tw, tl, dt, diag)
            if cand is None:
                continue
            H, score, landmark_px, key = cand
            if best_key is None or key > best_key:
                best_key = key
                best = (H, score, landmark_px)
        return best

    def _families(self, wseed, lseed, all_lines, diag):
        """Build the (WIDTH, LENGTH) detected-line families for one choice of
        which principal direction is the width axis.

        The two families are gathered ASYMMETRICALLY, because their geometries
        differ fundamentally:

          WIDTH  (baselines / service lines / net): these run across the court and
                 are mutually near-PARALLEL in the image, so their vanishing point
                 sits near horizontal infinity and is unreliable to estimate. So
                 the width family is gathered purely by ORIENTATION — every line
                 within PENCIL_CORE_DTHETA_DEG of the width direction. This spans
                 the full court DEPTH (far baseline high in the frame, near
                 baseline low), which a VP-convergence gather would miss (the near
                 baseline does not converge to any finite width VP).

          LENGTH (the two sidelines + centre service): these CONVERGE to a real
                 finite vanishing point, but the left and right sidelines have very
                 different image orientations, so orientation can't unite them —
                 only the shared VP does. So the length family is RE-DISCOVERED by
                 a VP-RANSAC over every line NOT aligned with the width direction,
                 then all lines converging to that VP are gathered.

        Both families are then de-duplicated (drop doubled Hough detections) and
        ordered projectively. Returns (det_w, det_l), or (None, None)."""
        wcore = self._orientation_core(wseed.lines)
        if len(wcore) < 2:
            return None, None, None, None
        wtheta = _orientation_median([L.theta for L in wcore])

        # WIDTH: orientation gather (the width VP is at ~infinity — unreliable)
        width_lines = [L for L in all_lines
                       if _angdiff(L.theta, wtheta) <= PENCIL_CORE_DTHETA_DEG]
        if len(width_lines) < 2:
            width_lines = wcore

        # LENGTH: RANSAC the depth VP over lines NOT aligned with width, then
        # gather every NON-width-oriented line converging there (both sidelines +
        # centre service). The non-width filter is kept on the GATHER too, so a
        # horizontal parasite passing near the depth VP can't enter the sideline
        # family (it would wreck the cross-ratio with a wild transversal position).
        non_width = [L for L in all_lines
                     if _angdiff(L.theta, wtheta) > PERP_ORIENT_MIN_DEG]
        lvp = self._ransac_vp(non_width, diag)
        length_lines = self._gather_from(lvp, non_width,
                                         [L for L in lseed.lines
                                          if _angdiff(L.theta, wtheta) > PERP_ORIENT_MIN_DEG]
                                         or lseed.lines, diag)

        # transversals for the cross-ratio / ordering: each family is measured
        # along a line of the OTHER family (a baseline crosses all sidelines, and
        # vice versa) — the only transversal that orders both opposite-sloping
        # sidelines correctly. Use the longest line of the other family.
        trans_for_w = max(length_lines, key=lambda L: L.length) if length_lines else None
        trans_for_l = max(width_lines, key=lambda L: L.length) if width_lines else None

        det_w = self._order_projectively(
            self._cap(self._dedup(width_lines, diag, trans_for_w)), trans_for_w)
        det_l = self._order_projectively(
            self._cap(self._dedup(length_lines, diag, trans_for_l)), trans_for_l)
        return det_w, det_l, trans_for_w, trans_for_l

    @staticmethod
    def _cap(lines):
        """Keep the K longest lines (bounds the combinatorial search)."""
        return sorted(lines, key=lambda L: -L.length)[:MAX_DETECTED_PER_PENCIL]

    @staticmethod
    def _gather_from(vp, pool, fallback, diag):
        """All lines in `pool` converging to `vp` within the gather band. Falls
        back to `fallback` if the VP is degenerate / too few lines. (Dedup + cap
        are applied by the caller, after gathering.)"""
        if vp is None or not np.all(np.isfinite(vp)):
            return list(fallback)
        band = VP_GATHER_BAND_FRAC_DIAG * diag
        conv = [L for L in pool
                if abs(L.coef[0] * vp[0] + L.coef[1] * vp[1] + L.coef[2]) <= band]
        return conv if len(conv) >= 2 else list(fallback)

    @staticmethod
    def _ransac_vp(cands, diag):
        """Max-consensus vanishing point over candidate lines (all pairs if few,
        else a capped random sample), refit by SVD on the inliers. Returns (x,y)
        or None."""
        n = len(cands)
        if n < 2:
            return None
        band = VP_RANSAC_BAND_FRAC_DIAG * diag
        best_inl, best_vp = [], None
        rng = np.random.default_rng(0)
        if n <= 30:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs = [tuple(rng.choice(n, size=2, replace=False)) for _ in range(500)]
        for i, j in pairs:
            vp = _line_intersection(cands[i], cands[j])
            if vp is None:
                continue
            inl = [L for L in cands
                   if abs(L.coef[0] * vp[0] + L.coef[1] * vp[1] + L.coef[2]) <= band]
            if len(inl) > len(best_inl):
                best_inl, best_vp = inl, vp
        if best_vp is None or len(best_inl) < 2:
            return None
        return _vp_from_lines(best_inl)

    @staticmethod
    def _dedup(lines, diag, transversal=None):
        """Drop near-coincident lines (same transversal position) — a doubled
        Hough detection of one court line would otherwise fake a second sideline
        and corrupt the cross-ratio. Keeps the longest of each coincident group."""
        if len(lines) < 2:
            return list(lines)
        tol = DEDUP_POS_TOL_FRAC_DIAG * diag
        pos = _transversal_positions(lines, transversal)
        order = sorted(range(len(lines)), key=lambda i: -lines[i].length)
        kept, kept_pos = [], []
        for i in order:
            p = pos[i]
            if p is None:
                kept.append(lines[i]); kept_pos.append(p); continue
            if any(kp is not None and abs(kp - p) < tol for kp in kept_pos):
                continue
            kept.append(lines[i]); kept_pos.append(p)
        return kept

    @staticmethod
    def _orientation_core(lines):
        """The orientation-coherent subset of a pencil — the members within
        PENCIL_CORE_DTHETA_DEG of the median orientation. A VP-RANSAC pencil can
        absorb cross-orientation parasites that pass near its VP by coincidence
        (a scoreboard/baseline line landing in the sideline pencil); those poison
        the VP re-estimate, so drop them first."""
        if not lines:
            return []
        med = _orientation_median([L.theta for L in lines])
        core = [L for L in lines if _angdiff(L.theta, med) <= PENCIL_CORE_DTHETA_DEG]
        return core if len(core) >= 2 else list(lines)

    @staticmethod
    def _order_projectively(lines, transversal=None):
        """Order lines by their position along the (cross-family) transversal —
        the genuine cross-ratio order along the pencil (ray angle around a noisy
        VP is unreliable; the transversal coordinate is exactly what the
        cross-ratio is computed on, so ordering by it keeps the order-preserving
        map and the shape prune in agreement)."""
        if len(lines) < 2:
            return list(lines)
        pos = _transversal_positions(lines, transversal)
        order = sorted(range(len(lines)),
                       key=lambda i: (pos[i] is None,
                                      pos[i] if pos[i] is not None else 0.0))
        return [lines[i] for i in order]

    def _score_assignment(self, det_w, det_l, trans_w, trans_l, dt, diag):
        """For ONE pencil->family assignment (det_w = WIDTH family ordered, det_l
        = LENGTH family ordered), enumerate every ORDER-PRESERVING injection of
        the detected lines into the 5 named ITF lines (both orientations) for each
        family, cross-ratio prune, solve via findHomography over ALL width x
        length intersections, warp-back score. Return best (H, score, lp, key) or
        None. trans_w / trans_l are the cross-family transversals for the
        cross-ratio (a baseline for the sidelines and vice versa)."""
        width_names = list(WIDTH_LINES)      # ordered far->near, value = world Y
        length_names = list(LENGTH_LINES)    # ordered left->right, value = world X
        if len(det_w) < 2 or len(det_l) < 2:
            return None

        # order-preserving (detected-subset -> named-subset) assignments per
        # family (both flips), cross-ratio / spacing pruned. Enumerating detected
        # SUBSETS (not all K) lets a single parasite line be dropped instead of
        # poisoning the whole family's cross-ratio.
        w_assigns = _ordered_assignments(det_w, width_names, WIDTH_LINES, trans_w)
        l_assigns = _ordered_assignments(det_l, length_names, LENGTH_LINES, trans_l)
        if not w_assigns or not l_assigns:
            return None

        best = None        # (H, warpback, landmark_px)
        best_key = None    # (coverage_bucket, warpback) — the selection objective
        for (w_sub, w_idx) in w_assigns:
            wl = [det_w[i] for i in w_sub]
            wn = [width_names[i] for i in w_idx]
            for (l_sub, l_idx) in l_assigns:
                ll = [det_l[i] for i in l_sub]
                ln = [length_names[i] for i in l_idx]
                corr = self._all_correspondences(wl, wn, ll, ln)
                if corr is None:
                    continue
                img_pts, world_pts = corr
                H = self._solve_homography(img_pts, world_pts)
                if H is None:
                    continue
                H_inv = np.linalg.inv(H)
                # ── all CHEAP geometric gates BEFORE the expensive warp-back ──
                # geometric validity: a degenerate sliver / wrong-aspect court is
                # rejected here so it can never out-score the true court.
                ok, _aspect = self._aspect_quad_ok(H_inv)
                if not ok:
                    continue
                landmark_px = self._emit_landmarks(H_inv)
                if landmark_px is None:
                    continue
                # reject mirror / self-intersecting / off-frame fits: the projected
                # doubles corners must keep the correct convex ordering AND sit
                # mostly in-frame. A symmetric court has a left<->right flip the
                # warp-back can't see (a mirrored court still lands on lines); this
                # corner-sanity test is what rejects it (and it's cheap).
                if not self._corners_sane(landmark_px, diag):
                    continue
                # SELECTION OBJECTIVE = (court COVERAGE bucket, warp-back). The
                # warp-back alone is fooled by a depth-collapsed court: when lines
                # are dense near the net a fit squashing the whole court into that
                # band still lands ~all its samples on lines (score ~1). So prefer
                # the assignment whose CONTROL LINES bracket the most court (full
                # depth via far+near baselines, full width via both sidelines) — a
                # well-conditioned full-court solve — with warp-back tie-breaking
                # within a coverage tier. Bucketing (not multiplying) stops a
                # slightly-lower-warp-back full court losing to a collapsed sliver.
                cov = round(_named_coverage(wn, ln), 1)
                # PRUNE: if this candidate's coverage bucket can't beat the best
                # already found, even a perfect warp-back (1.0) won't win -> skip
                # the expensive warp-back entirely.
                if best_key is not None and (cov, 1.0) <= best_key:
                    continue
                score = self._warpback_score(H, dt, diag)
                if score is None or score < SCORE_LOW_FRAC:
                    continue
                key = (cov, score)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (H, score, landmark_px, key)
        return best

    def _all_correspondences(self, det_w, names_w, det_l, names_l):
        """Every detected-width x detected-length intersection -> one image point
        whose world coords are known (X from the length-line name, Y from the
        width-line name). Returns (img_pts, world_pts) with >=4 points, or None.

        Requires the assigned named lines to span real distance in BOTH families
        (a solve from near-coincident world lines is ill-conditioned)."""
        ys = [WIDTH_LINES[n] for n in names_w]
        xs = [LENGTH_LINES[n] for n in names_l]
        if (max(ys) - min(ys)) < MIN_NAMED_SPAN_M:
            return None
        if (max(xs) - min(xs)) < MIN_NAMED_SPAN_M:
            return None
        img_pts, world_pts = [], []
        for dw, ym in zip(det_w, ys):
            for dl, xm in zip(det_l, xs):
                ip = _line_intersection(dw, dl)
                if ip is None:
                    continue
                img_pts.append(ip)
                world_pts.append((xm, ym))
        if len(img_pts) < MIN_CORRESPONDENCES:
            return None
        return img_pts, world_pts

    @staticmethod
    def _solve_homography(img_pts, world_pts):
        """image->world meters from >=4 correspondences via cv2.findHomography
        (least-squares / RANSAC over ALL points, so RMS is meaningful and the
        solve is robust to a single bad intersection)."""
        ip = np.array(img_pts, dtype=np.float32)
        wp = np.array(world_pts, dtype=np.float32)
        # reject degenerate (collinear / coincident) image control points
        if cv2.contourArea(cv2.convexHull(ip)) < 1.0:
            return None
        method = cv2.RANSAC if len(ip) > 4 else 0
        try:
            H, _ = cv2.findHomography(ip, wp, method=method,
                                      ransacReprojThreshold=0.20)
        except cv2.error:
            return None
        if H is None or not np.all(np.isfinite(H)):
            return None
        # reject (near-)singular H: it has no usable inverse (image<-world) and
        # would crash downstream. abs(det) tiny relative to the matrix scale.
        det = abs(np.linalg.det(H))
        if not np.isfinite(det) or det < 1e-12 * (np.abs(H).max() ** 3 + 1e-30):
            return None
        return H

    def _warpback_score(self, H, dt, diag):
        """Warp ALL named ITF model lines into the image via H_inv (one batched
        perspectiveTransform over the precomputed world samples) and score by the
        gchlebus/Farin line-integral, normalized over the WHOLE model:

            score = mean over ALL model samples of
                    (+1  if in-frame AND on a detected line,
                     -0.5 if in-frame but off any line,
                      0   if projected off-frame)

        Normalizing over the FULL model (not just in-frame samples) is what kills
        the degenerate sliver fit: a tiny mislocated court has most of its model
        off-frame (contributing 0), so even a perfect on-mask hit fraction over
        its few in-frame samples yields a low overall score. The true full court
        projects in-frame everywhere AND lands on lines -> score near 1. Returns
        the 0..1-ish score (can be slightly negative), or None if degenerate.

        Also enforces that the in-frame projected court SPANS a real fraction of
        the frame (a court crushed to a sliver, or off-frame, is rejected)."""
        H_inv = np.linalg.inv(H)
        hh, ww = dt.shape[:2]
        tol = WARP_DT_TOL_FRAC_DIAG * diag
        pts = cv2.perspectiveTransform(self._model_samples, H_inv).reshape(-1, 2)
        finite = np.all(np.isfinite(pts), axis=1)
        if not finite.any():
            return None
        pts = pts[finite]
        total = len(pts)
        x, y = pts[:, 0], pts[:, 1]
        inb = (x >= 0) & (x < ww) & (y >= 0) & (y < hh)
        inframe = int(inb.sum())
        if inframe < WARP_MIN_INFRAME_FRAC * total:
            return None   # court projects largely off-frame -> reject (sliver/wrong)
        xi = x[inb].astype(np.int32)
        yi = y[inb].astype(np.int32)
        # the in-frame court must span a real fraction of the frame extent
        extent = max(np.ptp(x[inb]), np.ptp(y[inb]))
        if extent < WARP_MIN_COURT_EXTENT_FRAC_DIAG * diag:
            return None
        on = dt[yi, xi] <= tol
        n_on = int(on.sum())
        n_off = inframe - n_on
        raw = WARP_ON_MASK * n_on + WARP_OFF_MASK * n_off   # off-frame contribute 0
        return raw / total

    def _emit_landmarks(self, H_inv):
        """Project the 4 doubles corners (+ service/net anchors when in-frame)
        to image px via H_inv, as the named-landmark dict homography_from_points
        consumes. Returns {name:(x,y)} or None if the corners are degenerate."""
        out = {}
        anchors = dict(CORNERS_WORLD)
        anchors.update({
            "far_svc_center":  (0.0, +SN), "near_svc_center": (0.0, -SN),
            "net_left": (-D, 0.0), "net_right": (+D, 0.0),
        })
        for name, (Xm, Ym) in anchors.items():
            p = H_inv @ np.array([Xm, Ym, 1.0])
            if abs(p[2]) < 1e-9:
                continue
            out[name] = (float(p[0] / p[2]), float(p[1] / p[2]))
        # need the 4 corners at minimum for a stable solve
        if not all(c in out for c in CORNERS_WORLD):
            return None
        return out

    def _corners_sane(self, lp, diag):
        """The 4 projected doubles corners must form a sane, correctly-oriented,
        mostly in-frame quad. Rejects the LEFT<->RIGHT mirror (which the warp-back
        cannot see on a symmetric court) and off-frame extrapolations.

          * left corners left of right corners (x_left < x_right) on BOTH baselines
          * far corners above near corners (y_far < y_near) on BOTH sidelines
          * the 4 corners are a convex quad
          * every corner sits within a modestly-expanded frame (no wild blow-up)
          * the court is ANCHORED in-frame: its centroid is inside the frame AND
            >=2 corners are strictly in-frame — rejects a "court" crammed into one
            corner / mostly off-frame (e.g. a fence/banner grid fitted on a frame
            that actually has no court).
        """
        need = ("far_bl_left", "far_bl_right", "near_bl_left", "near_bl_right")
        if not all(c in lp for c in need):
            return False
        fl, fr = np.array(lp["far_bl_left"]), np.array(lp["far_bl_right"])
        nl, nr = np.array(lp["near_bl_left"]), np.array(lp["near_bl_right"])
        # correct horizontal ordering (no mirror) on both baselines
        if not (fl[0] < fr[0] and nl[0] < nr[0]):
            return False
        # far above near (smaller image y) on both sidelines
        if not (fl[1] < nl[1] and fr[1] < nr[1]):
            return False
        # PERSPECTIVE consistency: the camera is above the court, so the NEAR
        # baseline (closer) must image at least as WIDE as the far one — never
        # dramatically narrower. The fence-on-a-no-court-frame false positive
        # collapses the near baseline to ~a point (far 438px vs near 39px); a real
        # court never inverts the perspective. Allow mild slack for level angles.
        far_w = abs(fr[0] - fl[0])
        near_w = abs(nr[0] - nl[0])
        if near_w < PERSP_NEAR_FAR_MIN_RATIO * far_w:
            return False
        quad = np.array([fl, fr, nr, nl], dtype=np.float32)
        if cv2.contourArea(cv2.convexHull(quad)) < 1.0:
            return False
        if len(cv2.convexHull(quad)) != 4:   # self-intersecting / non-convex
            return False
        # no corner may blow up far outside the frame (over-extrapolation tell)
        ww, hh = self._frame_wh
        marg = CORNER_FRAME_MARGIN_FRAC * diag
        for c in (fl, fr, nl, nr):
            if not (-marg <= c[0] <= ww + marg and -marg <= c[1] <= hh + marg):
                return False
        # the court must be ANCHORED in-frame, not crammed into a corner / off one
        # edge (the fence-on-a-no-court-frame false positive)
        cen = quad.mean(axis=0)
        if not (0 <= cen[0] <= ww and 0 <= cen[1] <= hh):
            return False
        n_inframe = sum(1 for c in (fl, fr, nl, nr)
                        if 0 <= c[0] <= ww and 0 <= c[1] <= hh)
        if n_inframe < 2:
            return False
        return True

    def _aspect_quad_ok(self, H_inv, diag=None):
        """Project the FULL doubles rectangle and require it to be a convex, sane
        image quad: aspect in [ASPECT_MIN, ASPECT_MAX] AND spanning a real fraction
        of the frame. The image-EXTENT gate is the load-bearing one — it rejects
        the collapse degeneracy (a homography fitted from only 2 width lines 6.4m
        apart extrapolates the ±11.885m baselines to a near-coincident point; its
        aspect ratio of two tiny numbers can still look fine, but its corner quad
        spans ~0px). Earlier ad-hoc area/aspect guards missed this because they
        didn't gate the projected-corner EXTENT in image pixels."""
        diag = diag if diag is not None else getattr(self, "_diag", None)
        corners = np.array([[-D, +H], [+D, +H], [+D, -H], [-D, -H]], dtype=np.float32)
        proj = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        if not np.all(np.isfinite(proj)):
            return False, 0.0
        hull = cv2.convexHull(proj.astype(np.float32))
        convex = len(hull) == 4
        cw = float(proj[:, 0].max() - proj[:, 0].min())
        ch = float(proj[:, 1].max() - proj[:, 1].min())
        aspect = ch / (cw + 1e-6)
        ok = convex and ASPECT_MIN <= aspect <= ASPECT_MAX
        if ok and diag is not None:
            extent = max(cw, ch)
            ok = extent >= QUAD_MIN_EXTENT_FRAC_DIAG * diag
        return ok, aspect

    @staticmethod
    def _confidence(score, vp_info, diag):
        spreads = [p.spread_px for p in vp_info["principal"]]
        tight = all(s < SPREAD_TIGHT_HIGH_FRAC_DIAG * diag for s in spreads)
        if score < SCORE_LOW_FRAC:
            return "fallback"
        if score >= SCORE_HIGH_FRAC and tight:
            return "high"
        return "low"

    @staticmethod
    def _fallback(reason, gate=None):
        logger.info(f"Auto court (classical): ABSTAIN — {reason}")
        return {"H": None, "H_inv": None, "confidence": "fallback",
                "rms_px": None, "n_used": 0, "source": "autoline",
                "abstain_reason": reason, "gate_features": gate}


def _build_model_samples():
    """All ITF model lines sampled every WARP_SAMPLE_SPACING_M, as a single
    (N,1,2) float32 world-point array ready for cv2.perspectiveTransform."""
    pts = []
    for (p0, p1) in ITF_MODEL_LINES.values():
        seg_m = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        n = max(2, int(seg_m / WARP_SAMPLE_SPACING_M))
        for t in np.linspace(0, 1, n):
            pts.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def _line_intersection(l1, l2):
    """Image intersection of two Line objects (homogeneous cross product)."""
    a1, b1, c1 = l1.coef
    a2, b2, c2 = l2.coef
    x = b1 * c2 - b2 * c1
    y = c1 * a2 - c2 * a1
    w = a1 * b2 - a2 * b1
    if abs(w) < 1e-9:
        return None
    return (x / w, y / w)


# ───────────── order-preserving correspondence + cross-ratio prune ─────────────

def _named_coverage(width_names, length_names):
    """How much of the real court the assigned named lines bracket, in [0,1]:
    (width-family world-Y span / full court depth) * (length-family world-X span /
    full court width). A full-court fit (far+near baselines x both sidelines) -> ~1;
    a net+service x half-width fit -> small. This is the selection objective that
    beats the depth-collapse degeneracy the warp-back alone can't see."""
    ys = [WIDTH_LINES[n] for n in width_names]
    xs = [LENGTH_LINES[n] for n in length_names]
    y_cov = (max(ys) - min(ys)) / (2.0 * H)
    x_cov = (max(xs) - min(xs)) / (2.0 * D)
    return float(y_cov * x_cov)


def _orientation_median(thetas):
    """Circular MEDIAN orientation in [0,180) — handles the 0/180 wrap (plain
    np.median of [0,1,179,180] gives a meaningless 90). Doubles the angle to map
    the 180-periodic orientation onto a full circle, takes the circular mean
    direction there, halves back."""
    a = np.deg2rad(np.asarray(thetas, dtype=np.float64) * 2.0)
    ang = np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 2.0
    return float(np.rad2deg(ang) % 180.0)


def _vp_from_lines(lines):
    """Least-squares vanishing point of a set of image lines: the right singular
    vector of the stacked unit [a b c] coefficients with the smallest singular
    value. Returns (x,y) in image px, or None if at infinity (near-parallel)."""
    if len(lines) < 2:
        return None
    A = np.array([L.coef for L in lines], dtype=np.float64)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    v = Vt[-1]
    if abs(v[2]) < 1e-9:
        return None
    return (v[0] / v[2], v[1] / v[2])


def _cross_ratio(p):
    """Cross-ratio of 4 collinear scalar positions p=[a,b,c,d] (a projective
    invariant). (a-c)(b-d) / ((a-d)(b-c)). None if degenerate."""
    a, b, c, d = p
    num = (a - c) * (b - d)
    den = (a - d) * (b - c)
    if abs(den) < 1e-9:
        return None
    return num / den


class _CoefLine:
    """Minimal Line-like wrapper carrying only homogeneous coefficients, for the
    intersection helper (a transversal we synthesise, not a detected segment)."""
    def __init__(self, coef):
        self.coef = coef


def _transversal_positions(det_lines, transversal=None):
    """1D position of each detected line along a common transversal — the basis
    for the projective cross-ratio of the pencil. Each detected line's
    intersection with the transversal, projected onto the transversal direction,
    is its scalar coordinate. Returns a list aligned with det_lines (None for
    lines parallel to the transversal).

    The transversal MATTERS for correctness: the cross-ratio of a pencil of lines
    equals the cross-ratio of their intersections with ANY transversal, but a poor
    transversal scrambles the ORDER. For the sideline family the two sidelines
    slope OPPOSITE ways in the image, so a centroid-perpendicular transversal
    orders them nonsensically; a line from the PERPENDICULAR family (a baseline,
    which every sideline crosses left->right) is the natural, correct transversal.
    Callers pass that line in `transversal`; otherwise we synthesise the
    centroid-perpendicular fallback."""
    if transversal is not None:
        a, b, c = transversal.coef
        # direction along the transversal (perp to its normal (a,b))
        tx, ty = -b, a
        # an origin point ON the transversal (its closest point to image origin)
        ox, oy = -a * c, -b * c
        tline = transversal
        cx, cy = ox, oy
    else:
        mids = np.array([L.midpoint for L in det_lines], dtype=np.float64)
        cx, cy = mids.mean(axis=0)
        th = np.deg2rad(_orientation_median([L.theta for L in det_lines]))
        dx, dy = np.cos(th), np.sin(th)            # line direction
        tx, ty = -dy, dx                           # transversal direction (perp)
        a, b = dx, dy
        c = -(a * cx + b * cy)
        tline = _CoefLine((a, b, c))
    pos = []
    for L in det_lines:
        ip = _line_intersection(L, tline)
        if ip is None:
            pos.append(None)
        else:
            pos.append((ip[0] - cx) * tx + (ip[1] - cy) * ty)
    return pos


def _order_consistent(det_pos, named_coords):
    """True if the detected lines' transversal positions vary monotonically in
    the SAME direction as the assigned named world coords — i.e. the order-
    preserving map holds. The detected lines arrive pre-ordered (ray angle around
    the VP); for the assignment to be order-preserving, their transversal
    coordinate must be monotone and increase iff the named coord increases."""
    valid = [(dp, nc) for dp, nc in zip(det_pos, named_coords) if dp is not None]
    if len(valid) < 2:
        return True
    dd = np.diff([v[0] for v in valid])    # detected position increments
    dn = np.diff([v[1] for v in valid])    # named world-coord increments
    if np.any(dd == 0) or np.any(dn == 0):
        return False
    # each sequence strictly monotone AND the two move together (sign-aligned)
    return bool(np.all(np.sign(dd) == np.sign(dd[0])) and
                np.all(np.sign(dn) == np.sign(dn[0])) and
                np.sign(dd[0]) == np.sign(dn[0]))


def _shape_ok(det_pos, named_coords):
    """Cross-ratio (>=4 lines) / spacing-ratio (3 lines) shape match between the
    detected transversal positions and the assigned named world coords. With 2
    lines there is no shape constraint (the span gate handles conditioning).

    With more than 4 valid lines, EVERY consecutive 4-window's cross-ratio must
    match — one parasite anywhere then fails the window it sits in (the caller
    instead drops it by trying a smaller detected subset)."""
    dps = [dp for dp, nc in zip(det_pos, named_coords) if dp is not None]
    ncs = [nc for dp, nc in zip(det_pos, named_coords) if dp is not None]
    n = len(dps)
    if n < 3:
        return True
    if n >= 4:
        for s in range(n - 3):
            cd = _cross_ratio(dps[s:s + 4])
            cn = _cross_ratio(ncs[s:s + 4])
            if cd is None or cn is None or abs(cn) < 1e-9:
                return False
            if abs(cd - cn) / abs(cn) > CROSS_RATIO_REL_TOL:
                return False
        return True
    # exactly 3: spacing ratio d01/d12
    sd = (dps[1] - dps[0]) / ((dps[2] - dps[1]) + 1e-9)
    sn = (ncs[1] - ncs[0]) / ((ncs[2] - ncs[1]) + 1e-9)
    if abs(sn) < 1e-9:
        return False
    return abs(sd - sn) / abs(sn) <= SPACING_RATIO_REL_TOL


def _ordered_assignments(det_lines, named_keys, coord_map, transversal=None):
    """Every ORDER-PRESERVING (detected-subset -> named-subset) correspondence,
    in BOTH orientations, surviving the cross-ratio / spacing shape prune.

    Enumerates: each ORDERED detected SUBSET of size m (MIN_LINES_PER_FAMILY <= m
    <= K) — dropping outliers/parasites — x each strictly-increasing injection of
    those m into the 5 named ITF lines x both named orientations (the flip). The
    detected lines are pre-ordered projectively, so a subset preserves order by
    construction. Returns a list of (det_subset_idx_tuple, named_idx_list). This
    is the combinatorial heart — small (sum_m C(K,m)*C(5,m)*2) and pruned hard by
    projective shape — NOT an arbitrary pairing. `transversal` is the cross-family
    line the cross-ratio is measured along."""
    k = len(det_lines)
    if k < 2:
        return []
    all_pos = _transversal_positions(det_lines, transversal)
    n_named = len(named_keys)
    out = []
    seen = set()
    m_min = min(max(MIN_LINES_PER_FAMILY, 2), k)
    for m in range(k, m_min - 1, -1):              # prefer using MORE lines first
        for det_sub in itertools.combinations(range(k), m):
            det_pos = [all_pos[i] for i in det_sub]
            if any(p is None for p in det_pos):
                continue
            for orient in (list(range(n_named)), list(range(n_named))[::-1]):
                for combo in itertools.combinations(range(n_named), m):
                    named_idx = [orient[i] for i in combo]
                    named_coords = [coord_map[named_keys[i]] for i in named_idx]
                    if not _order_consistent(det_pos, named_coords):
                        continue
                    if not _shape_ok(det_pos, named_coords):
                        continue
                    key = (det_sub, tuple(named_idx))
                    if key not in seen:
                        seen.add(key)
                        out.append((list(det_sub), named_idx))
    return out
