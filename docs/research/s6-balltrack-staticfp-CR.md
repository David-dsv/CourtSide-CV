# CR — S6: ball-track determinism, teleport-then-frozen static-FP reject

**Branch:** `feat/s6-balltrack-staticfp` (from `feat/accuracy-overhaul` @fdcd350).
**Scope (as briefed):** `run_pipeline_8s.py` Pass 1 only — the static-FP filter +
Kalman tracking. `vision/events.py` and `vision/bounce.py` left untouched (the
classifier is solid; the briefed wall is UPSTREAM ball-track hygiene).
**Method:** ULTRACODE, measure-driven. Every claim below is a LIVE measurement on
the canonical input `tennis.mp4 -s 73 -d 13 --device mps --match-mode`, scored
against the committed human ball GT + the demo3 event GT. A design+adversarial
workflow (8 agents) pressure-tested the discriminator before a line was written.

---

## TL;DR

The live ball track locks onto **static-FP "teleport-then-frozen" blobs** — a
transient corner/field FP the trajectory jumps into and freezes on. Read by
`vision/events.py`, a frozen blob fabricates a `vy_flip` → a **MANDATORY
(un-skippable) BOUNCE anchor** that floods the alternation DP. These are the
"phantom bounces that come and go." Fix = delete the blobs **at the source** in
Pass 1, before the spline / is-real mask / events consume them — deterministic,
fixes every downstream consumer at once.

**Result (LIVE, 3 baseline + 3 post-fix runs, all byte-identical on this box):**

| metric | BASELINE (3/3) | POST-FIX (3/3) |
|---|---|---|
| teleport-then-frozen blobs (the determinism metric) | **2** | **0** |
| bounce F1 | 0.400 | **0.556** |
| hit F1 | 0.400 | **0.444** |
| spline CORRECT vs ball GT (end-to-end) | 55.3% | **55.3%** (0 real points lost) |
| `measure_ball_coverage.py` replica (the stated 95.5%/92.3% gate) | 95.5% / 92.3% | **95.5% / 92.3%** (untouched) |
| run-to-run track variance | 0 (identical) | 0 (identical) |

The fix removes **4 FP segments / 76 frames** (cold-start + corner + chain +
right-field) and **10 off-frame coast points** — **0 of which are the real ball**
(every deleted segment's track is 340–1032px from the GT ball; tolerance is 38px).
All 8 required felix/regression tests stay green; a new deterministic unit test
(`tests/test_static_fp_reject.py`, 10/10) pins the discriminator.

---

## The bug, measured

LIVE on the canonical input the raw track contains (deterministic on this box, but
the *shape* is what varies cross-env, per the brief):

| FP segment | frames | track centroid | entry | the real ball (GT) | min dist |
|---|---|---|---|---|---|
| **cold-start** | f17–51 (35f) | (538,473) mid-court | *first lock* (no entry) | top of frame (826,25) | 413px |
| **corner** | f399–414 (16f) | (89,772) bottom-left | 1098px = **0.498·diag** jump | occluded / top | 1007px |
| **chain** | f417–421 (5f) | (711,517) | 0.212·diag from pre-blob | climbing far court | 360px |
| **right-field** | f518–549 (32f) | (1266,664) | 539px = 0.245·diag jump | climbing (957,190)→(1171,336) | 340px |

Plus **10 off-frame coast points** (the Kalman coasting the prediction off the top
edge to y=−62..−359 — physically impossible).

Each frozen blob's tiny vy jitter reads as a `vy_flip` in
`events._direction_features` → a mandatory BOUNCE anchor. On the baseline the
spurious events `(406,BOUNCE)`, `(549,HIT)`, `(152,BOUNCE)`, `(52,BOUNCE)` map
1:1 onto these segments. Removing them removes those phantoms.

**Determinism note (honest):** on THIS machine MPS is byte-deterministic, so the
3 baseline runs were identical (cov 84.5%, 2 blobs, F1 0.40) — the ±0.3 swing the
brief cites is a *cross-environment* effect (different YOLO/CPU candidate sets give
different blobs). The fix removes the blobs **at the source**, so whichever track a
given environment samples, the teleport-frozen FP can no longer survive to fake an
anchor. The determinism metric (`blob_audit.py` count) goes 2→0 and stays 0.

---

## The discriminator (and the trap it must not fall into)

⚠️ **The trap (brief + memory):** a REAL ball at an arc apex / serve toss is ALSO
near-frozen (it decelerates at the top — the ball GT confirms ±5px frozen runs at
apexes). A "frozen" reject ALONE destroys real tracking. Verified live: the cache
serve-toss @ (826,25) is frozen ~25f and IS the real ball (GT ±7px). So a run is
rejected only on a **CONJUNCTION**, never on frozenness. Three reject paths, each
with a measured keep-case that escapes it:

1. **Tier-1 hard teleport.** Frozen ≥0.16s **AND** entry jump > **0.30·diag** — a
   physically-impossible single-frame jump (no real ball covers 30% of the diagonal
   in one frame). Catches the corner blob (0.498). The real toss (0.243) is below it.
2. **Tier-2 soft teleport.** Frozen **AND** entry jump > 0.15·diag **AND** the track
   was **healthily MOVING just before** (premotion ≥ ~0.0008·diag/f). Catches the
   right-field blob (0.245, premotion 8.6px/f). A real ball re-acquired after a
   COAST enters from a near-frozen decayed prediction (~0.4px/f, below the floor) →
   kept. This is the cache-toss escape (premotion 0.37px/f).
3. **Cold-start (exit-teleport).** The FIRST lock has no entry to test. Reject it iff
   it is the first lock **AND** a long freeze (≥0.30s) **AND** **exited by a
   teleport** (>0.15·diag from the centroid to the next real point). A real
   cold-start apex LEAVES on a smooth arc (small exit jump, toss = 0.068); an FP is
   abandoned by a teleport when the track finally finds the real ball (538-blob exit
   = 0.185). Restricted to the *first* lock so a mid-clip real ball that merely
   PRECEDES an FP (its teleport-out is caused by the *next* segment) is never
   deleted — verified on the live f380–398 real climb (kept).

**Episode chaining.** One FP capture is often compound: the track teleports into the
corner blob, freezes, then briefly latches a 5-frame frozen FP @ (711,517) on the
way back. The base rule misses it (run < min_run), and leaving it lets the spline
reconnect through it so the phantom BOUNCE just MOVES to the episode edge instead of
vanishing (measured: deleting only the 16f core moved BOUNCE@406→417). So after a
core blob we keep absorbing trailing SHORT frozen runs that are still teleports from
the pre-episode anchor, until a genuine ballistic re-acquisition ends the episode.

**Off-frame reject.** Any track point outside `[0,W)×[0,H)` is nulled — a real ball
is always in-frame (GT confirms), so this loses 0 GT points by construction and
kills the off-frame-coast phantom (B@152→gone).

### Why NOT the two approaches the workflow's adversary killed
- **Gate-tightening the reinit gate (0.50→0.25·diag):** UNSAFE — it false-rejects a
  real **redirect-after-occlusion** re-acquisition (ball occluded at contact,
  reappears the other way, ~0.34·diag from the stale prediction). Our post-pass
  keeps it because a redirected real ball **keeps moving** (never forms a ≥min_run
  freeze). Verified with a synthetic redirect: 0 deletions.
- **Recurrence (a 2nd, lower static threshold):** REFUTED by measurement — the live
  FP cells (89,772)/(1267,663) have **0–1** candidates clip-wide (transient, not
  recurring logos), so a recurrence filter would not fire on them; and the toss cell
  recurs across repeated serves, so it would false-reject the real toss. The
  teleport+frozen+exit conjunction sidesteps both.

All thresholds are fractions of `frame_diag`/`fps` — zero clip pixels. The unit test
re-runs the decisive cases at 2× resolution and asserts identical decisions.

---

## Proof that 0 real GT ball points are deleted

For every deleted segment the track is FAR from the real ball during the run
(tolerance = 0.02·W = 38px):

```
cold (538,473)  f17-51 : GT present 29f, mean dist 501px,  min 413px  -> FP
corner (90,772) f399-414: GT present  3f, mean dist 1032px, min 1007px -> FP
chain (711,517) f417-421: GT present  2f, mean dist 377px,  min 360px  -> FP
right (1266,664)f518-549: GT present  8f, mean dist 461px,  min 340px  -> FP
off-frame ×10           : every point outside [0,W)x[0,H) -> can never match GT
```

End-to-end confirmation: **spline CORRECT vs ball GT is 55.3% before AND after** —
the reject removed only never-correct points. (The 55.3% is the live MPS track's
pre-existing quality; the cache replica that the `measure_ball_coverage.py` gate
scores is 92.3% CORRECT and is untouched by this change — `replay_strat.py` is not
modified.)

---

## Where it lives

`run_pipeline_8s.py`:
- `reject_teleport_frozen_blobs(centers, frame_diag, fps, ...)` — the pure-numpy
  discriminator (top of file, next to `BallKalmanFilter`). Testable in isolation.
- Call site: inside `if not wasb_path:`, right after Pass 1b builds
  `raw_ball_centers` and BEFORE `ball_is_real` / `--dump-ball-track` / spline /
  events. **Gated to the CLEAN-broadcast density path** (`_clean_path`, same gate as
  the ROI fallback): on a parasite-heavy oblique clip (felix) the far-court ball
  genuinely hovers near the baseline and a big apparent jump is real perspective
  motion — running the reject there deletes real far-apex bounces (measured: 11
  felix deletions incl. GT bounce f53). felix path stays byte-identical.
- Off-frame reject: a 5-line loop in the same gated block.

New tooling: `tools/ball_gt/blob_audit.py` — the determinism thermometer (counts
teleport-frozen blobs in a `--dump-ball-track` dump; `--json` for scripting).

---

## Validation

- **8 required regression tests GREEN:** `test_bounce_regression` (felix F1 0.800,
  floor 0.72), `test_bounce_wasb_regression` (0.865), `test_vx_veto`,
  `test_event_confusion_regression` (0/0, bounce F1 0.889), `test_ball_conf_regression`
  (felix 0.750), `test_ball_roi_regression`, `test_roi_redetect`, `test_far_coverage`.
- **New deterministic test GREEN:** `tests/test_static_fp_reject.py` — 10/10
  (4 reject flavours + 5 keep-traps incl. redirect-after-occlusion & real-toss &
  real-ball-preceding-FP + 2 scale-free 2× invariance). Discriminating: with the
  reject disabled (no-op) the corner-blob assertion FAILS.
- **felix safe:** the reject is gated out of felix; the felix tests use their own
  ported tracker/cache and are unaffected.

---

## Honest limits / what S6 does NOT fix

- **confusion_B→H = 4 persists.** Of the live spurious events, the teleport-frozen +
  off-frame phantoms are killed (B@406, B@152, B@52, H@549 gone; bounce F1
  0.40→0.556). The residual spurious events (B@126, B@150, H@229, B@300, B@320,
  H@348) are **DP-placement errors on the REAL ball** + a deeper in-frame
  coast-divergence (f147–153 the track drifts up-and-back ON-screen) — these are the
  globally-coupled single-STEP alternation-DP wall documented in
  `courtside-s3-event-recall-negative` ("8 levers all regress"). They are OUT of S6
  scope (events/decoder territory, frozen here on purpose).
- The live track's overall CORRECT (55%) is below the cache replica's 92% — the live
  MPS YOLO/Kalman track is genuinely weaker than the frozen cache suggests. That is a
  separate ball-detection lever (cold-start recall, `courtside-ball-conf-density`),
  not the static-FP determinism this session owns.

## Reusable lessons
1. **Deleting the frozen CORE is not enough** — the phantom anchor moves to the
   episode edge unless the whole compound excursion (chain + off-frame) is removed.
   Validate the EVENT output, not just the blob count.
2. **The cold-start FP is the dominant live one** and has NO teleport-entry — the
   exit-teleport (a real apex exits ballistically, an FP is abandoned) is the only
   causal separator, and it must be restricted to the first lock.
3. **Validate on a LIVE `_stats.json`**, not the cache: the cache replica is 92%
   CORRECT and 1 (different) blob; the live track is 55% CORRECT with 2 different
   blobs. The cache HID both the live FPs and the live quality.
