# S1 — Ball tracking toward SOTA (CR)

**Branch:** `feat/s1-ball-sota` (from `feat/accuracy-overhaul`)
**Session:** autonomous ULTRACODE, measure-driven, web-research, zero hardcoding/cheat.
**Thermometer:** human ball GT `tests/fixtures/ball_gt/tennis_demo3.ball_gt.json` (311 visible pts + 39 occluded) scored by `tools/ball_gt/measure_ball_coverage.py` (coverage % + CORRECT % within 2%·W of GT, overall + cold-start first 4s). felix bounce GT (`tests/fixtures/bounces/felix.bounces.json`, 16 bounces) as the adversarial generalization guard via `test_bounce_regression` / `test_ball_conf_regression` (floor F1 ≥ 0.72).

> ⚠️ Every number below is measured LIVE on the real pipeline path or on a det-cache that was *separately re-validated LIVE* (the demo3 cache↔LIVE match is exact). We do not trust a frozen cache without a LIVE check (the project's "caches lied 4×" lesson).

---

## TL;DR — what shipped

**Density-adaptive ball resolution/conf** (`--ball-auto`, default ON). The pipeline probes the mean ball-candidate density on a cheap subsampled low-conf pass, then picks the ball detection path:

| Clip | measured density | path chosen | result |
|---|---|---|---|
| demo3 (clean broadcast) | **2.0** | 1920 imgsz + conf 0.05 | coverage **86.2% → 95.5%**, CORRECT **79.1% → 92.3%**, cold-start CORRECT **74.5% → 90.8%** |
| tennis (clean broadcast) | 2.5 | 1920 + 0.05 | (clean side, gets the win) |
| felix (parasite-heavy oblique) | 6.9 | 1280 + conf 0.16 | bounce F1 **0.750** = current prod, **byte-identical** (zero regression) |

The threshold (`--ball-density-thr 3.5`) is derived from a **detected feature** (candidate density), not a per-video constant: demo3 2.0 / tennis 2.5 / felix 6.9 — a 3.4× separation, **stable across thr ∈ [3.0, 5.0]** and across probe-conf/subsample. This is the CLAUDE.md "thresholds from detected features / generalization over precision" principle applied to the ball detector.

**Net:** clean broadcast (the SaaS target — amateur on a tripod, in-play balls) gets +9.3pt coverage, **+13.2pt CORRECT, +16.3pt cold-start CORRECT**; parasite-heavy clips keep the proven felix-safe path. All regression tests green.

---

## The problem (measured)

Baseline (LIVE, 1280 imgsz / conf 0.16): demo3 coverage 86.2% / CORRECT 79.1% / cold-start 85.3% / 74.5%. Event recall is plateaued by trajectory holes, worst at cold-start and on tiny/far/high balls.

## Journal (measure → change → re-measure)

### Step 0 — baselines (LIVE, prod-faithful)
- demo3 1280/0.16: **cov 86.2% / CORR 79.1% / START cov 85.3% / CORR 74.5%** (exactly the prompt's baseline).
- felix bounce F1 (prod turn-pass): 0.750–0.774 (≥0.72 floor).

### Step 1 — imgsz 1920 is a huge win on clean broadcast (LIVE-confirmed)
Cache sweep then **LIVE-validated**:
- demo3 **1920 / conf 0.05**: cov **95.5%** / CORR **92.3%** / START cov **95.1%** / CORR **90.8%**.
- The cache predicted it and the LIVE scorer reproduced it *exactly* — the gain is real, not a cache artifact.

**Why it works (deep finding):** the per-frame "oracle" (best raw candidate vs GT) at 1920/0.05 is only 75% CORRECT, yet the Kalman output is 92%. The accuracy comes from the **motion model**, not the raw box: 1920 gives the Kalman *denser, more consistent measurements* so its filtered state lands closer to truth. That's a robust mechanism (smoothing), not overfitting to the box centers.

### Step 2 — but 1920 is TOXIC on the parasite-heavy oblique clip (felix)
Built `felix_det_cache_1920.json` and swept. **Every** 1920 conf fails the felix floor:

| conf | felix 1280 F1 | felix 1920 F1 |
|---|---|---|
| 0.05 | 0.625 | **0.710** (FAIL) |
| 0.10 | 0.774 | 0.710 (FAIL) |
| 0.16 | 0.750 | 0.686 (FAIL) |

Diagnosis: at 1920, felix has **7238 candidates @0.16 with 100% multi-candidate frames** (vs demo3's clean ~390/650). felix is a grazing-oblique clip with crowd/line/far-field junk; 1920 surfaces *more parasites*, and the Kalman locks onto a parasite trajectory. Despike has **zero** effect (the FPs aren't teleport spikes — they're smooth-but-wrong bounce calls f89/f601/f820/f1608, several of which are the known hit/bounce-confusion frames that `--event-methodo` arbitration exists for, and which this session must not touch). The felix track at 1920 is actually *denser/better* (98.6% coverage); the F1 dip is a downstream bounce-classifier sensitivity, not a tracking regression — but it's a real LIVE degradation we won't ship.

### Step 3 — the generalizing fix: density-adaptive resolution
The candidate density at a low probe floor separates clean from parasite robustly and cheaply (no GT, no per-video constant):

| clip | mean cand/frame @probe 0.05 | stable to subsample 10× |
|---|---|---|
| demo3 | 2.0 | 2.1 |
| tennis | 2.5 | — |
| felix | 6.9 | 6.9 |

→ clean (≤3.5) gets 1920/0.05; parasite (>3.5) gets 1280/0.16 (the proven felix-safe = current prod). Parasite-path conf is **0.16, not 0.10**: felix F1 is knife-edge in conf (fails at 0.09/0.11/0.13, passes 0.14/0.15/0.16) — 0.16 sits on the stable plateau, so the parasite path is byte-identical to today's prod and `test_ball_conf_regression` stays valid.

LIVE end-to-end confirmation on demo3: `Ball-auto: density=2.12 (thr=3.5) → CLEAN broadcast → imgsz=1920, conf=0.05` then `Ball tracking: 578/650 detected, 31 interpolated, 609/650 total` — the 1920 path runs end-to-end in the real pipeline and the auto-probe fires correctly (density 2.12 matches the cache's 2.0).

### Step 4 — pistes that did NOT win (measured, not assumed)
- **WASB (heatmap-temporal)**: demo3 coverage **65.0% / CORRECT 62.4%** (frame_step=1) — far below 1920-Kalman's 95.5%. WASB is blind on this clip's tiny/far balls; not viable as primary. (Kept for the `--ball-tracker wasb` path it already serves on felix bounce recall.)
- **Tiling / upscale (SAHI-like)**: ran the YOLO26 ball model on a 2×-upscaled top-half crop at conf 0.05 over the 17 high-in-frame demo3 misses → **recovered 0/17**. The model finds *nothing* there; the residual gaps are a **model-capability ceiling** (serve-toss/blur balls outside the training distribution), not an inference-resolution gap. SAHI confirmed NO-GO for this footage. (These gaps are short and the 0.4s cubic spline already bridges them in the usable trajectory.)
- **WASB fusion (fill Kalman gaps)**: gated WASB gap-fill into the 1920-Kalman track filled **1 gap**, coverage unchanged (95.5% → 95.5%). WASB is blind exactly where YOLO is (tiny/far/high balls), so it has nothing to contribute where the Kalman has holes. NO-GO — pure complexity for zero gain. (WASB stays on its own `--ball-tracker wasb` path for felix bounce recall, where it already earns its keep at F1 0.865.)

---

## Web research — SOTA tennis ball trackers 2024–2026 (license-gated)

License is a hard SaaS gate: MIT/Apache OK; AGPL/no-license = blocker.

| # | Model | License | Tennis accuracy | CPU/MPS | Weights public | Verdict |
|---|---|---|---|---|---|---|
| 1 | **TOTNet** (occlusion-aware, 2025) | **MIT** | visible RMSE 9.59 vs WASB 16.58; full-occlusion RMSE 15.31 vs WASB 264.45 (same tennis benchmark) | yes, 8.65M / 28 FPS | ⚠️ `weights/` dir, no README link | **PILOT (highest ceiling)** — confirm/obtain weights or train on public TrackNet tennis set (MIT-clean) |
| 2 | **RacketVision / MS-TrackNetV3** (AAAI'26) | **MIT** | P 0.945 / R 0.880 / MDE 1.96px | yes (CUDA-first; verify MPS) | ✅ HF `linfeng302/RacketVision-Models` | **PILOT (lowest risk — weights exist + MIT)** |
| 3 | **SAHI** sliced inference | **MIT** | wrapper (+5–13% AP small obj elsewhere) | yes | uses our weights | **tested → NO-GO here** (0/17 recovered; model-ceiling, not resolution) |
| 4 | TrackNetV4 (motion attn) | MIT | +1.6 F1 over V2 | TF | ⚠️ placeholder links | NO-GO (marginal, no weights) |
| 5 | TrackNetV3 (qaz812345) | MIT | 98.56 F1 **badminton only** | yes | ✅ shuttlecock only | NO-GO (wrong sport) |
| 6 | yastrebksv/TrackNet (tennis) | **NO LICENSE** | tennis broadcast | yes | ✅ tennis | **NO-GO (blocker)** — same unlicensed author as the court detector already flagged |
| 7 | TrackNetV5 | proprietary | F1 0.986 | small | ❌ gated | **NO-GO (blocker)** |

**Recommendation (next session):** the two MIT pilots worth a measured trial are **RacketVision MS-TrackNetV3** (public MIT weights, lowest integration risk; verify it runs on MPS and score vs the 1920-Kalman baseline + felix) and **TOTNet** (highest occlusion ceiling; needs weights or a short MIT-clean train). Both are gated behind a flag and must beat 1920-Kalman *and* hold the felix floor before becoming default. Neither is a blocker on this session's shipped win.

(Full report with URLs in the research agent transcript; key links: TOTNet https://github.com/AugustRushG/TOTNet · RacketVision https://github.com/OrcustD/RacketVision + HF `linfeng302/RacketVision-Models` · SAHI https://github.com/obss/sahi · WASB https://github.com/nttcom/WASB-SBDT.)

---

## Felix-intact proof
- `test_bounce_regression` (felix cache, F1 ≥ 0.72): **PASS** (0.800 no-arg / 0.774 prod turn-pass).
- `test_ball_conf_regression` (reads the LIVE parasite-path conf 0.16): **PASS** (0.750).
- `test_bounce_wasb_regression` (≥ 0.80): **PASS** (0.865).
- `test_event_confusion_regression`: **PASS** (H→B 0, B→H 0, bounce F1 0.889).
- The parasite path (felix) is **byte-identical** to today's prod (1280/0.16); the auto block only changes the *clean* path. So felix's whole pipeline is untouched.

## Most reusable idea
**Key a per-clip detector knob on a measured candidate-density feature, not a global constant.** The same high-res/low-conf that wins on clean footage floods a parasite-heavy clip; the *raw candidate density at a low floor* is a cheap, GT-free, robust discriminator (3.4× separation, wide stable threshold) that lets one pipeline serve both geometries. This generalizes beyond imgsz: any aggressive-recall setting that risks parasites can be density-gated the same way. The deeper accuracy lesson: on this data **coverage comes from detections but CORRECT comes from the Kalman motion model** — denser consistent measurements make the *filtered* estimate accurate even when individual raw boxes are noisy; that's why 1920 helps and why it's robust rather than overfit.

## Files touched
- `models/yolo_detector.py` — `probe_ball_density()` (GT-free cleanliness metric).
- `run_pipeline_8s.py` — `--ball-auto`/`--no-ball-auto`/`--ball-density-thr`; density probe + adaptive imgsz/conf in Pass 1; `--ball-conf`/`--ball-imgsz` default None (auto) with explicit override precedence.
- `tests/test_ball_conf_regression.py` — `shipped_ball_conf()` reads the parasite-path auto_conf when the default is adaptive.
- `tools/ball_gt/strat_lab.py` — two-clip (demo3 ball-GT + felix bounce-GT) strategy thermometer that catches demo3-overfit instantly.
- `tests/fixtures/ball_gt/felix_det_cache_1920.json`, `tennis_det_cache_1280.json` — new caches for the felix 1920 measurement + the 3rd-clip threshold de-risk.
