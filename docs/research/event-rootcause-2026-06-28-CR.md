# CR — Event-detection root cause (2026-06-28 PM re-diagnosis, supersedes the stale handoff framing)

**Branch:** `feat/accuracy-overhaul` (PM artifacts only — no production code changed here).
**Method:** ULTRACODE, measure-driven. A multi-agent diagnostic workflow attributed every
live event error to a verified root cause and ranked fixes; each attribution was
adversarially verified and each fix measured offline on the faithful live dump.

## TL;DR — three findings that change the plan

1. **MEASUREMENT INTEGRITY (the big one).** The canonical demo3 test input is
   **`tennis.mp4 -s 73 -d 13`** (the user's actual FULLSTACK invocation), NOT the
   re-extracted **`data/output/tennis_demo3.mp4 -s 0`**. The extracted clip is an H.264
   **re-encode that degrades the tiny near-player pose**, collapsing hit F1 from **0.500
   (canonical) to 0.133 (extracted)** while bounce F1 stays 0.625. The prior handoff's
   trap-#2 fix (re-extract RAW to kill minimap pollution) silently introduced a
   re-encode-degradation trap. **All event benchmarks must run on the full video.**
   (CPU ≡ MPS on the same input — this is NOT a device-nondeterminism issue.)

2. **The ball track is GOOD, NOT the wall** (contradicts the handoff's "balle en
   cacahuète → fix the ball track" premise). Measured live: ~95.5% coverage, **92.6%
   CORRECT**, 87.5% cold-start. Only 3 of the canonical baseline's misses are true
   candidate-generation gaps (B174, B308, H394); the rest are in-pool, dropped/mislabeled
   by the classifier. The S3 "(90,772) teleport-frozen blob" is gone on the current track.

3. **The dominant, verified, generalizable HIT bug is a CIRCULAR proximity measurement
   in `vision/events.py`** (not the ball track, not device noise, not detect_hits FWHM).
   In PASS-0 (`events.py:356`) each `detect_hits` member is inserted with
   `xy=(h["x"],h["y"])` = the **self-reported wrist contact point**; then `dmin`
   (`events.py:375`) and the `hit@wrist` anchor (`events.py:419-421`, `dmin<near`) are
   computed AT that wrist point — so any near-court wrist-speed peak is trivially "near a
   wrist" and fires a HIT anchor the firewall cannot suppress. Measuring `dmin` from the
   **actual tracked ball position** (`_xy_at`) cleanly separates real from phantom:
   GT hits have ball-dmin ≤0.51; phantom anchors have ball-dmin ≥0.90.

## Canonical baseline (full-video = FULLSTACK = what the user complained about)

| metric | value |
|---|---|
| pool recall | 14/17 (NO CANDIDATE: B174 flat far apex, B308, H394) |
| confusion H→B / B→H | 1 / 1 |
| bounce F1 | 0.625 (R 0.56) |
| hit F1 | 0.500 (R 0.50) |
| pred BOUNCE | [122,231,253,271,369,456,510] |
| pred HIT | [65,108,147,196,292,333,487,530] |
| spurious | (108,HIT),(231,BOUNCE),(292,HIT),(487,HIT) |
| bounce misses | 174, 308, 569 |
| hit misses | 81, 394, 468 |

Faithful instrument (replays byte-for-byte vs `_stats.json`):
`venv/bin/python tools/event_eval/live_pool.py tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json`
The committed dumps:
- `tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json` — **canonical**, use this.
- `tests/fixtures/methodo/demo3_methodo_inputs_extracted_DEGRADED.json` — the degraded
  extract, kept ONLY to demonstrate the input-integrity effect. Do not optimize against it.

## Ranked fixes (measured, adversarially verified)

### ✅ S1 — Ball-anchored HIT proximity (the dominant verified win) — `vision/events.py`
Decouple the proximity `xy` used for `dmin`/`hit_like`/`hit@wrist` from the rep-frame `xy`:
for a `hit` member, measure proximity at `_xy_at(fr)` (the ball), not `(h["x"],h["y"])`.
- **Canonical full-video: bounce F1 0.625 → 0.706** (recovers B570), hit F1 0.500 held,
  confusion held — a clean +0.08 with no regression.
- Degraded extract: hit F1 0.133 → 0.500 (the big rescue), confirming the mechanism.
- Regression cache: 0/0 held, bounce F1 unchanged 0.889, **hit F1 0.632 → 0.737**.
- Scale-free: robust across `near=0.4-1.0`. No new constant. Removes a circular measurement.
- KNOWN residual (do NOT fix here): +1 confusion_B→H (B308 parity-filled as HIT@307) — an
  upstream/decoder artifact (the ball at 307 IS far from a wrist; the global DP overrides
  the feature). That is S2's territory.

### 🔶 S3 — Robust direction-slope to kill jitter-faked vy-flips — `vision/bounce.py`
Phantom BOUNCE@540 (on the degraded extract) is a NON-ROBUST least-squares slope artifact:
a single-frame +57px jitter on a clean descending arc fakes a vy-flip because `_veto_slope`
(`bounce.py:841`) has no outlier rejection. A robust/despiked slope (median / Theil-Sen)
flips it back. Counterfactual verified (despike → phantom gone, conf B→H 1→0, bounce F1
0.625→0.750). **Caveat:** this touches a load-bearing felix direction feature — the
implementer MUST keep `test_bounce_regression` (felix F1≥0.72) green. Disjoint file from
S1 → parallel-safe.

### 🔶 S2 — Less-coupled decoder + the residual B→H@308 (OPT-IN ONLY) — `vision/events.py`
`_anchor_parity_decode` (segment-local) exists but `classify_events` hardcodes
`decoder="global"` (`events.py:443`). The global single-STEP parity DP is brittle (it
parity-fills B308 as HIT@307 and propagates phantom anchors). **BLOCKED as a default
flip:** on the committed cache the anchor decoder regresses bounce F1 0.889→0.714 and hit
F1 0.632→0.308 (fails the floors). Run AFTER S1 lands; expose anchor decoder as opt-in or
do a targeted DP fix for @308; default-flip forbidden until a felix LIVE dump + a
re-baselined cache agree. Same file as S1 → must run SEQUENTIALLY after S1.

## ❌ DO NOT PURSUE (measured net-zero / regression)

- **wrist_prox_max=1.2 gate on the methodo detect_hits** (`run_pipeline_8s.py:1654`):
  re-introduces confusion_B→H and drops cache bounce F1 0.889→0.778 (fails 2 tests); on
  the live track it cuts REAL hits (263→268, 520→525, dmin>1.2) and keeps phantoms. The
  track is offset from the wrist AT contact — proximity-as-a-gate is the wrong layer; S1
  (ball-anchored proximity-as-a-feature, scale-free) is the right one.
- **Corroboration-gating the mandatory vy-flip anchor:** EXACT no-op (the phantom's track
  jitter makes turning_points + sharp_turns fire together — they read the same corrupted
  track, so they are not independent corroborators). Reproduces S3-CR lever #6b.
- **FWHM upper-bound / prominence inside detect_hits:** refuted as root cause AND computed
  on the wrong (raw, not locked) pose stream. The lever is pose-feed alignment / S1.
- **Hand BOUNCE@540 to an S1 static-FP teleport reject:** refuted — f540 is a 1-frame
  jitter on a clean arc, not a teleport-and-stay; it legitimately survives the teleport
  veto. The fix is robust slope (S3), not track deletion.
- **More ball-track work as the headline fix:** the ball track is 92.6% CORRECT; low
  leverage for this failure case.

## Device recommendation
Make the pipeline **device-robust, do NOT pin cpu**. CPU ≡ MPS on the same input; the pose
feed is dense and good on MPS (far-pose 100%). The cross-track sensitivity seen in S2 is the
cache/track candidate-set differing, not MPS pose noise. Keep BOTH validators permanently:
the LIVE full-video dump (accuracy gate) AND the committed cache tests (no-regression
floor); ship only when both agree.

## Reusable lessons
1. **Re-confirm the test INPUT before optimizing.** A re-encoded extract degraded hits 4×
   and would have sent a whole session chasing a phantom "ball-track" wall.
2. **The cache lies in both directions** — accuracy on a fresh LIVE full-video run; cache
   = guardrail only.
3. **A "near a wrist" proximity feature anchored at the wrist is circular** — anchor it at
   the independent signal (the ball).
