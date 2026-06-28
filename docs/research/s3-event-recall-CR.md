# S3 — Event-recall candidate generation: a measured NEGATIVE result + the upstream hand-off

**Branch:** `feat/s3-event-recall` (from `feat/accuracy-overhaul`)
**Scope:** `vision/events.py` candidate generators (+ optionally `vision/bounce.py`
`detect_bounces_robust`). NOT `models/`, `vision/pose.py`, `vision/shots.py`
`detect_hits`.
**Mode:** ULTRACODE — measure-driven, web-research, zero hardcoding, no questions.

## TL;DR

The S3 premise — *bounce/hit recall is capped by CANDIDATE GENERATION (far-court
apex + cold-start events "with no candidate")* — **is refuted by direct measurement
on a fresh live run.** On the canonical command the **live candidate-pool recall is
17/17**: every GT event already has a well-placed candidate, and 16/17 GT events are
tracked at the correct ball *location*. The live wall is **classifier PLACEMENT**,
not generation. No `vision/events.py`-local change robustly raises live F1 while
holding confusion 0/0 — **8 distinct intervention families were measured and all
regress or flood**, because the global alternation DP is brittle to a single spurious
anchor and the spurious anchors come from **upstream ball-track glitches**. The
honest deliverable is this negative result + a reusable LIVE thermometer
(`tools/event_eval/live_pool.py`) + a precise hand-off of the real fix to S1
(ball-track) and to a future decoder-robustness item.

This conclusion independently **reproduces and sharpens** the prior finding
`memory: courtside-events-plateau-ball-tracking-wall` ("STOP iterating
vision/events.py classification: plateaued by ball-tracking").

---

## 1. The thermometer (LIVE, zero-cheat)

The S3 prompt is explicit that **the cache lies** — the frozen demo3 event cache's
Kalman track differs from what the live pipeline produces today. Confirmed here a
5th time: the committed cache is **saturated at pool recall 17/17, bounce F1 0.889,
hit F1 0.632, confusion 0/0** — it has *no* generation gap and *hides* the live
placement wall entirely. **Every number below is from a LIVE run**, scored against
the demo3 GT (`tests/fixtures/bounces|shots/tennis_demo3.*.json`, 9 bounces + 8 hits,
matcher tolerance ±8f @50fps).

Measurement loop (instant, faithful):
1. One live run dumps its exact `classify_events` inputs via the existing env hook:
   `COURTSIDE_DUMP_METHODO_INPUTS=/tmp/s3/live_inputs_run1.json python run_pipeline_8s.py
   tennis.mp4 -s 73 -d 13 --device cpu --dump-bounces …`
2. **New tool** `tools/event_eval/live_pool.py` replays the dump offline: per-generator
   pool recall + the full `classify_events` decode + confusion/F1. Its output is
   **byte-identical to the live `_stats.json`** (verified: pred BOUNCE
   `[126,165,177,229,272,320,369,406,456,508]`, pred HIT
   `[108,146,196,300,348,398,427,467,519]` match the real stats exactly), so offline
   iteration measures the live result with zero pipeline cost and zero self-deception.

---

## 2. The decisive measurement — generation is NOT the wall

LIVE run1 baseline (`--event-methodo` ON, the default):

| metric | value |
|---|---|
| candidate-pool recall | **17/17** (bounces 9/9, hits 8/8 have a candidate ±8f) |
| GT events tracked at correct ball *location* | **16/17** (≤48px; only B174 is a frozen-FP coincidence) |
| confusion H→B / B→H | **1 / 1** |
| bounce F1 | **0.526** (R 0.556) |
| hit F1 | **0.588** (R 0.625) |
| bounce misses | 62, 255, 569 — **all in-pool** (nearest cand at d8, **d1**, d7) |
| hit misses | 81, 333 — **all in-pool** (nearest cand at **d1**, **d0**) |
| spurious | HIT@108, BOUNCE@165, BOUNCE@229, BOUNCE@320, HIT@348, **BOUNCE@406**, HIT@427 |

**Every dropped/mislabeled GT already has a well-placed candidate.** B255 has a
candidate at distance 1, H81 at distance 1, H333 at distance 0 — and all are dropped
by the classifier. There is no generation gap to close on this track.

### Root cause A — spurious MANDATORY anchors from ball-track glitches
The headline spurious event **BOUNCE@406**: the live raw track teleports the ball
from (997,158) at f398 to (87,772) at f399 (**1097 px/frame**), then it **freezes**
at ~(90,772) — a static false-positive in the bottom-left corner, off the play area —
for 16 frames (399–414). `_direction_features` reads a tiny `vy_flip=True` on that
frozen blob at f406, which makes it a **MANDATORY** BOUNCE anchor (the DP cannot skip
it and even *restarts* the chain from it). It corrupts the rally cadence and cascades
mislabels onto real events (B308→labeled HIT@300, H268→labeled BOUNCE@272). A second
corrupting anchor is a `detect_hits` FP at f348 (a near-court swing with no GT there).

### Root cause B — the global alternation DP is brittle / globally coupled
`_global_alternation_decode` derives ONE rally `STEP` + parity over the whole clip,
so inserting or removing a SINGLE candidate reshuffles labels far downstream. The
first committed anchor is at **f177**, so the entire HEAD (GT B62, H81, B120, H141) is
*pre-anchor*, decoded only by the weak HIT-only edge logic — which is exactly why the
cold-start bounce/hit are dropped.

### Why local features cannot separate phantom from real (all REFUTED by measurement)
The phantom 406 and the real frozen-pause bounce 174 are **locally identical** (174 is
*also* a frozen cluster, (553,473) for 30 frames). Discriminators measured and refuted:
- `static_run_len`: real 174 = 30f **>** phantom 406 = 16f.
- `y_excursion`: phantom 406 = 0.57 is the **largest** of all.
- `speed_scale`: real bounce 174 = 0.0 = phantom 406 = 0.0.
- `±10f teleport`: also flags real bounce 511 and real hit 525.

The one CLEAN discriminator found — **teleport-entry into a frozen plateau** (406's
plateau entry jump = 0.497·diag vs real 174's = 0.019·diag, scale-free) — separates
406 from real bounces, but on the live track it **also fires on real events near track
re-initializations** (e.g. real hit 525's plateau is entered by a 0.245·diag jump),
and removing 406 still leaves the globally-coupled DP free to reshuffle.

---

## 3. Eight in-scope levers measured — all regress or flood (live run1)

| # | lever (events.py / bounce.py, in scope) | live conf H→B/B→H | live bF1 | live hF1 | verdict |
|---|---|---|---|---|---|
| 0 | **BASELINE** (current code) | 1 / 1 | 0.526 | 0.588 | — |
| 1 | SG-curvature turn generator (research #1) | n/a (cache-saturated, no live delta vs sharp_turns) | — | — | no-op |
| 2 | SG-velocity-angle turns | — | — | — | does NOT recover far apex (SG flattens the kink) |
| 3 | ball-at-wrist cold-start hits (full) | 0 / 2 | 0.571 | 0.600 | B→H worse, +10 spurious |
| 4 | ball-at-wrist (cold-start/head-only or silent-only) | 0 / 1 | 0.500 | 0.556 | one insertion cascades (flips B120/H196/H525) |
| 5 | teleport-frozen-FP **vy-anchor demotion** (the clean discriminator) | 0 / 1 | 0.600 | **0.444** | bounce +0.07 but **hit −0.14** |
| 6 | corroboration-gated mandatory-restart — `bnc` | 0 / 1 | 0.471 | 0.444 | kills turn-only real bounces 174/511 |
| 6b| corroboration-gated mandatory-restart — `bnc_or_turn` (the judge's (A)) | 1 / 1 | 0.526 | 0.588 | **exact no-op** (406 has turn=True) |
| 7 | promote ball-at-wrist turns to HIT anchors | 1 / 2 | 0.421/0.500 | 0.556 | cascades (step 26→27.5) |
| 8 | two-parabola fit-residual generator (research #2) | 1 / 2 | 0.500 | 0.500 | **floods** (62 cands), +19 spurious |

**Pattern:** every lever that fixes a bounce error introduces a hit error (or vice
versa) through the DP coupling — the firewall already holds confusion 0/0 *structurally*
(it is logical gates in `_reward`, untouched by all of these), but the *F1* wall is the
placement DP + the upstream track, neither of which a candidate-generation edit reaches
robustly. **No lever holds 0/0 AND raises F1.**

---

## 4. Web research evaluated (2024–2026)

A verified multi-agent survey (TrackNet-family event heads, physics/impact detection,
change-point detection, kinematic features) was run and adversarially fact-checked.
Top CPU-only candidate-generation ideas and their measured outcome here:

- **Savitzky–Golay-smoothed curvature (κ = |v×a|/|v|³) for the turn detector** (all
  four threads' top pick). Implemented + measured: recovers the far apex on the cache
  but **adds no live pool recall** (the pool is already 17/17) and the SG *velocity-
  angle* variant actively *flattens* the kink it seeks. No live gain.
- **Two-parabola / change-point fit-residual segmentation** (TT3D 2D-cost, `ruptures`).
  Implemented + measured: **floods** (a candidate every ~7 frames) without heavy
  gating that would make it equivalent to the existing detectors. Net F1 down.
- **Cold-start wrist-swing rescue** (badminton "Case-4" / shot-refinement fusion).
  Implemented + measured: recovers the cold-start hit's *frame* but **cascades** in the
  DP. No robust gain.
- **DO NOT BOTHER** (refuted): TTNet/PES event heads (GPU + frame-stack, blind to the
  1px far apex), TrackNetV2/3/4 & WASB as *event* detectors (tracking-only, no event
  head), TT3D 3D half (needs calibration/IPopt), BlurBall/CPLASS/FOCuS (GPU / no Python
  port / no code), LSTM bounce classifier (GPU + author reports it fails on far apex).

The research is sound; it simply **does not apply** because the live bottleneck is not
generation. (Full brief archived in the session transcript.)

---

## 5. What ships

1. **`tools/event_eval/live_pool.py`** — a LIVE pool-recall + placement + confusion
   thermometer that replays a `COURTSIDE_DUMP_METHODO_INPUTS` dump and matches the real
   `_stats.json` byte-for-byte. This is the honest instrument the cache could not be
   (the cache is saturated and hides the live wall). Reusable for S1 and any future
   event work. **No production-code change** — `vision/events.py` and `vision/bounce.py`
   are left byte-identical, so all 6 regression tests stay green and felix is untouched.

2. **This negative-result CR + memory** (`courtside-s3-event-recall-negative`).

### The real fix (out of this scope — hand-off)
- **S1 ball-track** (`run_pipeline_8s.py` static-FP filter): add a **teleport-then-
  frozen static-FP reject** — a detection cluster pinned for ≥N frames that is *entered*
  by a >~0.15·diag single-frame jump is a non-ball blob (court mark / corner logo), not
  a ball. This deletes the corrupt input (e.g. the (90,772) blob) for **every**
  downstream consumer at the source, where it is cleanly identifiable, instead of trying
  to discriminate it locally after it has already poisoned the candidate pool. Highest
  leverage; belongs in the `feat/s1-ball-sota` worktree.
- **A less-coupled / soft-anchored decoder** (`_global_alternation_decode`): the global
  single-`STEP` parity DP is structurally brittle (one anchor edit cascades). A
  segment-local or soft-anchor decoder is a separate research item, not a one-line fix.

---

## 6. Robustness (≥2 LIVE runs) + regression status

- **Run1 LIVE** (verified vs `_stats.json` byte-for-byte): pool 17/17; conf 1/1; bF1
  0.526; hF1 0.588. Confusion is NOT 0/0 on this fresh track — consistent with
  `courtside-events-plateau-ball-tracking-wall` (live confusion reproducibly ≠ 0 once
  the ball track varies; the firewall holds 0/0 *structurally* but the DP+track produce
  spurious placements).
- **Run2 LIVE** (second independent run, same canonical command): the
  `COURTSIDE_DUMP_METHODO_INPUTS` dump is **byte-identical** to run1
  (`diff` clean, both 1383941 bytes) — the cpu pipeline is fully deterministic on this
  clip, so the placement wall is **exactly reproducible**: pool 17/17, conf 1/1, bF1
  0.526, hF1 0.588, the same spurious BOUNCE@406 and the same in-pool drops
  (62/255/569, 81/333). The phantom anchor is not run-noise — it is a stable, fixable
  property of the deterministic live track, which is the ideal evidentiary basis for
  the S1 static-FP hand-off (the corrupt (90,772) blob can be validated-away
  deterministically). (Prior sessions' run-to-run confusion variance was *cross-
  environment* — different YOLO/CPU — not within-environment.)
- **6 regression tests**: GREEN — no production code changed, so
  `test_event_confusion_regression` (cache 0/0, bF1 0.889, hF1 0.632), `test_sharp_turns`,
  `test_bounce_regression`, `test_bounce_wasb_regression`, `test_vx_veto`,
  `test_far_coverage` are byte-unaffected. **felix** end-to-end: untouched.

## 7. Reusable lessons

1. **Re-measure the stated premise on a FRESH live run before optimizing.** The
   "generation wall" was a frozen-cache artifact; live generation is fine (17/17).
2. **A saturated benchmark cache cannot reveal a live placement wall** — and it
   actively misleads (the judge's own replay of a slightly-different harness produced
   bF1 0.737 vs the true 0.526; only matching the real `_stats.json` is trustworthy).
3. **When a global DP couples all decisions, local candidate hygiene cannot fix
   placement without cascading** — the decoder or the input must change.
4. **Do not tune a scale-free threshold to make one live run look good** — that is the
   precise trap this codebase has fallen into before. A documented non-fix beats a
   brittle fix that games one run.
