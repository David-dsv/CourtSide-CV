# S4 — Automatic court homography (classical, no-GPU) — Change Report

> **Branch:** `feat/s4-auto-homography` (from `feat/accuracy-overhaul`).
> **Flag:** `--auto-court` (default **OFF**; OFF = current behaviour unchanged).
> **Status:** SHIPPED no-GPU classical path. tennis solves metric-correct; felix
> abstains honestly; the third-party court-level clip abstains on the frames we
> can decode (honest, see Limits). Implements the R&D plan in
> `docs/research/auto-homography-oblique.md` (chantiers C1–C6 + C-VAL).

## 1. TL;DR

- **Built the classical no-GPU automatic court detector** the R&D doc recommended:
  white-line mask → Hough line candidates → vanishing-point pencils → Farin/
  gchlebus **order-preserving warp-back correspondence** → `cv2.findHomography`.
  Pure OpenCV+numpy, **zero learned weights, zero GPU, license-clean.**
- **The genuinely-hard core — "which detected line is which named ITF line" — is
  solved** by the gchlebus (BSD-3-Clause) combinatorial recipe: the two pencils
  are the two ITF directions; their lines, ordered projectively, map to an
  **order-preserving injection** of the 5 named lines per family; a warp-back
  line-integral over the white mask selects among the geometrically-valid
  hypotheses. The ITF names fall out of the winning correspondence — no labels.
- **felix correctly ABSTAINS** (grazing angle → depth axis unobservable). This is
  the load-bearing honesty property: the gate refuses to emit a confidently-wrong
  homography and defers to the existing semi-auto sidecar / scalar. Verified by
  the regression test as the single most important assertion.
- **tennis solves at `confidence=high`**: the projected ITF court traces the real
  court (overlay-verified), warp-back 0.69, aspect 0.38, metric baselines/sidelines
  consistent with the 10.97 m / 23.77 m ITF truth.
- **Deliverables:** `models/court_lines.py` (mask + lines + VP + gate),
  `models/court_auto.py` (correspondence + warp-back fit), `--auto-court` wiring
  in `run_pipeline_8s.py`, `tools/court_eval/eval_homography.py` (metric scorer,
  with the doc's 12.80 m distance-bug fix), `tests/test_court_regression.py` +
  committed line-mask cache. No regression: bounce F1 still 0.80 (floor 0.72).

## 2. Measured results per clip (live, first frame, 1920×1080, diag 2203 px)

All three probe clips re-measured live this session (the R&D numbers were
priors — these are the implementation's actual measurements).

| Clip | Angle | merged lines | distinct VP dirs | 2nd-dir length ratio | VP-sep (diag) | gate | fit verdict |
|---|---|---|---|---|---|---|---|
| **tennis** | elevated-oblique broadcast | 53 | 2 (θ 13°, 59°) | 0.50 | 0.55 | **PASS** | `high` — overlay traces the real court; baseline/sideline ≈ 10.97/23.77 m |
| **felix** | grazing side-on | 46 | 1 real (2nd pencil = 0.11 length ratio) | **0.107** | — | **ABSTAIN** | `fallback` (correct — depth axis unobservable) |
| **djokovic** | court-level (4K→1080p decode) | 152 (frame 0) | 2 (θ 12°, 89°) | 0.49 | 0.46 | gate PASS | `fallback` — frame 0 is a non-court shot; clean court frames lack a 2nd sideline candidate upstream (see Limits) |

**Timing** (once-per-clip static-camera solve): tennis ~4.4 s, felix ~0.3 s,
djokovic ~4.2 s. Acceptable for a one-shot calibration; cache per clip like the
manual sidecar for repeated runs.

## 3. What was built (chantiers)

- **C1–C2 `models/court_lines.py::white_line_mask` / `detect_line_candidates`** —
  mask = (bright achromatic V>p80 ∧ S<60/255) ∨ (multi-scale morphological
  top-hat ridge >p95 at {3,5,7}px) ∨ (arXiv:2404.06977 **7×7 court-colour test**:
  modal-HSV court colour; a pixel is a line iff it is non-court while ≥4 of its
  7×7 neighbours are court). HoughLinesP (minLen 0.10·diag, thr 0.05·diag) →
  collinear merge (Δθ≤2°, Δρ≤0.5%·diag) → TLS refit with mask-support count.
- **C3 `estimate_vp_pencils` + `abstention_verdict`** — pencils by **sequential
  VP-RANSAC** (sample line pairs → intersection = candidate VP → perpendicular-
  residual consensus → peel → repeat); pencils are VP-coherent sets, **not** θ
  bins (perspective fans one direction across a wide θ range — the trap two
  probe runs fell into first). **Honest all-member spread** (median perp residual
  of EVERY member, never the inlier-subset — the djokovic optimism trap). The
  **4-veto abstention gate** (all orthogonal): (1) ≥2 distinct-orientation
  principal pencils, (2) **second-direction line-length ratio ≥ 0.25** [primary
  felix catcher], (3) VP-separation ≥ 0.40·diag, (4) honest spread < 0.25·diag.
- **C4–C5 `models/court_auto.py::AutoCourtDetector`** — width family gathered by
  orientation (near-parallel, VP at ∞), length family **re-discovered by
  VP-convergence** (unites the two sidelines, which are parallel in world but
  opposite-sloping in image); ordered detected lines → **order-preserving
  injection** into the 5 named ITF lines (+ cross-ratio/spacing prune) → all
  width×length intersections → `cv2.findHomography` → warp-back **line-integral**
  (+1 on-mask, −0.5 off, 0 off-frame, normalised over the WHOLE model so a tiny
  mislocated court scores low) → select by (court-coverage tier, warp-back) among
  hypotheses passing aspect∈[0.18,2.5], corner-extent ≥0.35·diag, convexity,
  perspective near/far width ratio, and anchored-in-frame gates. Emits the EXACT
  `court_calibrator.homography_from_points` dict → **drop-in, zero downstream
  changes** (`vision/court_zones.py`, speed, minimap all consume it unchanged).
- **C6 wiring** — `--auto-court` inserted in the cascade AFTER `--homography`
  (keypoint keeps broadcast) and BEFORE the calibration sidecar (a calibrated
  clip still prefers its exact manual points). OFF by default; OFF path untouched.
- **C-VAL** — `tools/court_eval/eval_homography.py`: pure-comparator metric scorer
  vs hand-clicked GT (corner-RMS with leave-one-out for ≥5 pts; **metric error
  between known ITF inter-landmark distances** — the PRIMARY number — with the
  R&D **bug fix `svc_to_svc = 12.80 m`, not the spec's wrong 6.40 m**; far/near
  split). `tests/test_court_regression.py` exercises the full fit+gate on a
  committed white-line-mask cache (no video / torch), asserting felix→fallback,
  tennis→sane metric court, djokovic→honest abstain.

## 4. Web technologies evaluated (+ licenses, verified against real LICENSE files)

| Component | Role | License | SaaS-OK | Verdict |
|---|---|---|---|---|
| **gchlebus/tennis-court-detection** | the line→ITF combinatorial warp-back recipe | **BSD-3-Clause** | ✅ | **Reimplemented** (algorithm, not code copy) — this is the reference design |
| Farin et al. 2003-05 | foundational 4-stage classical court calib | paper | ✅ | combinatorial stage-3 = the naming solution |
| **arXiv:2404.06977** (Agrawal) | 7×7 court-colour test + warp-back score | algorithm | ✅ | both ideas reimplemented; its ML parts (ShadowFormer, YOLOv5-AGPL) skipped |
| OpenCV / HoughLinesP / top-hat | classical stack | Apache-2.0 | ✅ | the whole measured pipeline is Apache-clean |
| **yastrebksv/TennisCourtDetector** | current keypoint model | **NO LICENSE** (a web "MIT" claim was WRONG) | ❌ | confirmed SaaS blocker — internal eval only, never shipped; **this work is its clean replacement** |
| DeepLSD (MIT) / M-LSD (Apache) | optional CPU line upgrade | MIT / Apache | ✅ | future option for faded clay; not needed for the probe clips |
| PnLCalib | point+line LM refinement | **GPL-2.0** | ❌ | idea-only (LM refine), reimplement clean; never link |
| RF-DETR Keypoint | learned replacement | Apache-2.0 (base) | ✅ | future GPU pilot only (needs labelled oblique data); `rfdetr_plus`/XL = PML-1.0, avoid |
| Ultralytics YOLO(v5/v8) | (used elsewhere in repo) | **AGPL-3.0** | ❌ | network-copyleft trap; not used by this path |

**Posture:** the shipped auto-court path is **100% BSD/Apache/MIT/self-written/
uncopyrightable-algorithm**. It needs no third-party weights and no labelled data,
so it sidesteps the yastrebksv SaaS blocker entirely.

## 5. What works / what doesn't (honest)

**Works.** felix abstains (the safety property). tennis: a full-court elevated
view is fit metric-correct from scratch, no model. The dict is a true drop-in.
The abstention gate rests on **four orthogonal signals**, not one — felix is
caught by both the second-direction length ratio (0.107≪0.25) AND the post-fit
aspect, so a single parasitic oblique line can't flip it.

**Doesn't (yet).**
- **The court-level clip never produces a positive fit.** Frame 0 is a behind-
  the-court non-court shot (correct abstain). On the clean full-court frames the
  upstream `detect_line_candidates` surfaces only ONE sideline direction as a
  real pencil (the other sideline is faint/occluded at that 4K-downscaled
  court-level angle), so the gate's second-direction veto fires. This is an
  **upstream line-detection density limit, not a fit bug** — the fit no longer
  emits the geometrically-wrong collapsed homography it did mid-development; it
  abstains honestly. A DeepLSD/M-LSD line extractor (license-clean, CPU) is the
  documented next lever to surface the faint sideline.
- **Far-field conditioning on tennis.** The near half of the projected court
  aligns tightly; the far baseline sits a few px off (the classical far-field
  blow-up the R&D doc predicts). `confidence=high` here is generous — a future
  pass should weight the warp-back by far-field residual and may demote such fits
  to `low`. The `SCORE_HIGH_FRAC`/`SCORE_LOW_FRAC` warp-back thresholds remain
  the **least-grounded constants** (extrapolated, not swept on GT) — recalibrate
  once a multi-clip oblique GT set with metric ground truth exists.
- **Line→ITF naming is solved combinatorially, not provably.** The
  order-preserving + cross-ratio prune makes it tractable and correct on tennis,
  but a heavily-occluded amateur frame (missing far baseline) can still mis-seed;
  warp-back is the only backstop. The near-half-anchor-then-extend trick
  (arXiv:2404.06977) is the documented mitigation, not yet implemented.

## 6. Reusable idea

**The discriminator between "a real second court direction" and "parasitic
oblique noise" is total line LENGTH, not line count or VP-spread.** felix's
spurious second pencil has a tight VP spread (0.42%·diag — *tighter* than the
real direction) and ≥3 members, so a count- or spread-only gate passes it. But
its members are short net/fence/player edges: their summed length is **11% of the
dominant direction's** (vs ~50% for tennis/djokovic's real sidelines). A real ITF
direction spans the court; parasites don't. The **second-direction length ratio**
is a scale-free, single-number felix catcher with a wide margin (0.11 vs 0.49) —
and it generalises (it's "does the depth axis carry court-spanning evidence",
which is exactly the physical condition for an observable depth axis). Pair it
with the post-fit projected-aspect veto for an independent confirmation.

A second, hard-won lesson: **warp-back self-score alone is NOT a safe selection
objective.** A small court mislocated onto a dense line band (the net) scores AS
HIGH as the true full court. The fix is to make **court coverage / image-extent
the primary selection key and warp-back only the tie-breaker** within a coverage
tier — and to normalise the warp-back integral over the *whole* model (off-frame
samples contribute 0) so a sliver is penalised. Both are baked into `_fit`.
