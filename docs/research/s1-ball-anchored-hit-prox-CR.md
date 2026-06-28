# CR — S1: ball-anchored HIT proximity (the dominant verified win)

**Branch:** `feat/s1-ball-anchored-hit-prox` (from `feat/accuracy-overhaul`).
**Commit:** `25178a9` — `fix(events): ball-anchored HIT proximity (kills circular wrist measurement)`.
**Scope:** `vision/events.py` ONLY (one file, +29 / −4 lines).
**Method:** ULTRACODE, measure-driven. Baseline → fix → re-measure on the canonical LIVE
full-video input AND the regression cache; ship only because both agree.
**Brief:** `docs/research/event-rootcause-2026-06-28-CR.md` §S1 + `docs/prompts/` S1 prompt.

---

## TL;DR

The dominant, generalizable HIT bug was a **circular proximity measurement** in
`classify_events` PASS-0: each `detect_hits` member was inserted with `xy` = the wrist
contact point self-reported by `detect_hits`, then `dmin` / `hit_like` / the `hit@wrist`
anchor were computed AT that wrist point — so any wrist-speed peak was trivially "near a
wrist" (dmin≈0) and fired a HIT anchor the firewall could not suppress.

**Fix:** measure player-proximity for a `hit` member at the **TRACKED BALL** position
(`_xy_at(fr)`), not the wrist. The ball is the independent signal: a real racket contact
has the ball genuinely at a wrist; a phantom swing has the ball far away.

**Result (LIVE full-video, the canonical input the user actually runs):**

| metric | baseline | **after fix** | floor | verdict |
|---|---|---|---|---|
| bounce F1 | 0.625 (R 0.56) | **0.706 (R 0.667)** | ≥ 0.625 | ✅ **+0.081** (recovers B569) |
| hit F1 | 0.500 (R 0.50) | **0.500 (R 0.50)** | ≥ 0.500 | ✅ held |
| confusion H→B | 1 | **1** | held | ✅ identical pair (@271, pre-existing) |
| confusion B→H | 1 | **1** | known @308/@62 = S2 | ✅ identical pair (@65, pre-existing) |
| bounce misses | 174, 308, **569** | 174, 308 | — | ✅ **569 recovered** |
| spurious | (108,H)(231,B)(292,H)(487,H) | same | — | ✅ no new FP |

LIVE and cache **agree byte-for-byte** on the event list. The CR's predicted gain (bounce
F1 0.625→0.706, hit held, confusion held, cache hit F1 0.632→0.737) reproduced **exactly**.

---

## Canonical input (non-negotiable)

All event benchmarks run on the user's real invocation, **NOT** the re-encoded extract:

```
venv/bin/python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode
venv/bin/python tools/event_eval/score_stats.py data/output/tennis_annotated_stats.json
```

GT: `tests/fixtures/bounces/tennis_demo3.bounces.json` (9) +
`tests/fixtures/shots/tennis_demo3.shots.json` (8). Tolerance ±8f (0.16s @ 50fps).

Fast faithful replay (byte-identical to the live `_stats.json` events):

```
venv/bin/python tools/event_eval/live_pool.py tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json
```

---

## Root cause (verified)

`classify_events` PASS-0, before the fix:

- `events.py:356` — `cand.append((int(h["frame"]), "hit", (int(h["x"]), int(h["y"])), h))`
  → a `hit` member's `xy` = the **wrist** point auto-reported by `detect_hits`.
- `events.py:375-376` — `dmin = min(_nearest_player_dist_norm(fr, xy, …) for fr,_,xy,_ in members)`
  computed AT that wrist `xy`. **Circular**: distance from a wrist to the nearest wrist ≈ 0.
- `events.py:419-421` — `hit@wrist` anchor fires on `hit and dmin < near`; `hit_like = (hit
  or dmin < near) and not vy_bounce` (`events.py:415`). With `dmin≈0` for every hit member,
  the firewall is defeated **by construction** — a phantom wrist-speed peak cannot be
  measured as "far from the ball-at-contact" because the measurement never looked at the
  ball.

**The independent signal is the tracked ball.** Measured separation on the canonical clip:
real GT hits have **ball-dmin ≤ 0.51**; phantom anchors have **ball-dmin ≥ 0.90**. Concrete
example traced from the live dump — the GT shot@268 cluster (event @271): ball@271=(688,344),
**ball-dmin = 1.58** (clearly off any wrist), whereas the wrist-anchored measurement gave
≈0.

---

## The fix (clean — not the PM's proxy)

The PM's diagnostic proxy rewrote `h["x"],h["y"]` *before* the loop, which also moved the
rep-frame/display xy (a side effect). The shipped fix is the correct, decoupled one: the
display / rep-frame `xy` is **left untouched** (still the wrist point); only the value fed to
`_nearest_player_dist_norm` changes. A single closure `_prox_xy(fr, src, xy)` returns the
**ball** (`_xy_at(fr)`) for `hit` members and the existing (ball-derived) `xy` for
`bounce`/`turn` members. The three proximity call-sites (the `dmins` list, the `hit`
rep-pick, the min-dist rep fallback) route through it.

- **Zero new constant.** Reuses the existing scale-free, box-height-normalized `near` (=
  `near_wrist`, default 0.5). No magic number, no per-video tuning. Honors "zéro hardcoding".
- **Falls back to the wrist xy** only when the ball is untracked at `fr` (rare — `_xy_at`
  reads the spline-filled `smoothed_centers`, which is ~99% dense), preserving old behavior
  exactly in that degenerate case.

### Exact diff (`git show 25178a9`)

```diff
@@ classify_events(...)  (after the _xy_at closure)
+    def _prox_xy(fr, src, xy):
+        """The xy at which to MEASURE player-proximity for this candidate member.
+        ... (ball for `hit` members, else the existing ball-derived xy; display xy untouched) ...
+        if src == "hit":
+            ball = _xy_at(fr)
+            if ball is not None:
+                return ball
+        return xy

@@ PASS-0 per-cluster
-        dmins = [_nearest_player_dist_norm(fr, xy, players_per_frame)
-                 for fr, _, xy, _ in members]
+        dmins = [_nearest_player_dist_norm(fr, _prox_xy(fr, s, xy),
+                                           players_per_frame)
+                 for fr, s, xy, _ in members]

@@ rep selection (hit@wrist pick)
-                dn = _nearest_player_dist_norm(fr, xy, players_per_frame)
+                dn = _nearest_player_dist_norm(
+                    fr, _prox_xy(fr, s, xy), players_per_frame)

@@ rep selection (min-dist fallback)
-            best = min(members, key=lambda m: _nearest_player_dist_norm(
-                m[0], m[2], players_per_frame))
+            best = min(members, key=lambda m: _nearest_player_dist_norm(
+                m[0], _prox_xy(m[0], m[1], m[2]), players_per_frame))
```

---

## Measured results (real, on the canonical full video)

### A. LIVE full-video `_stats.json` (the accuracy gate)

Run: `run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device mps --match-mode`, scored with
`score_stats.py`.

```
            Pred BOUNCE   Pred HIT   Missed
GT BOUNCE        6           1         2
GT HIT           1           4         3
confusion_H->B = 1 | confusion_B->H = 1
BOUNCE  P=0.750 R=0.667 F1=0.706  (TP=6 FP=2 FN=3)
HIT     P=0.500 R=0.500 F1=0.500  (TP=4 FP=4 FN=4)
bounce misses: [174, 308]   hit misses: [81, 394, 468]
spurious: [(108,'HIT'),(231,'BOUNCE'),(292,'HIT'),(487,'HIT')]
pred BOUNCE: [122, 231, 253, 271, 369, 456, 510, 570]
pred HIT   : [65, 108, 145, 196, 292, 333, 487, 530]
```

vs baseline (before fix, same input):

```
confusion_H->B = 1 | confusion_B->H = 1
BOUNCE  F1=0.625 (R=0.56)   HIT  F1=0.500 (R=0.50)
bounce misses: [174, 308, 569]
pred BOUNCE: [122, 231, 253, 271, 369, 456, 510]   (no 570)
pred HIT   : [65, 108, 147, 196, 292, 333, 487, 530]
```

**Delta:** +BOUNCE@570 (matches GT B569, dist 1) → bounce F1 +0.081. Everything else byte-
identical (one cosmetic 147→145 HIT shift, both match GT shot@141).

### B. Regression cache (the no-regression floor) — `test_event_confusion_regression`

```
demo3 confusion: H->B=0 B->H=0 | bounce F1=0.889 | hit F1=0.737   PASS
```

Baseline cache hit F1 was 0.632; the fix **lifts it to 0.737** (exactly the CR's prediction),
with confusion **0/0 held** and bounce F1 **0.889 unchanged**.

### C. felix intact — `test_bounce_regression`

```
no-arg:        F1=0.800 P=0.857 R=0.750 (TP=12 FP=2 FN=4)   PASS  (floor 0.72)
PROD turn-pass:F1=0.774 P=0.800 R=0.750 (TP=12 FP=3 FN=4)   PASS  (floor 0.72)
```

### D. All committed tests green (+ a new focused guard)

```
test_ball_anchored_prox         : PASS  (NEW — see below)
test_event_confusion_regression : PASS  (0/0, bounce 0.889, hit 0.737)
test_bounce_regression          : PASS  (felix 0.800 / 0.774 >= 0.72)
test_sharp_turns                : PASS  (pool recall 17/17, firewall plateau holds)
test_vx_veto                    : PASS  (felix WASB F1 0.9143)
```

`py_compile vision/events.py` clean.

**`tests/test_ball_anchored_prox.py` (NEW, force-added — tests are gitignored here).**
A fast synthetic guard that locks the fix and is DISCRIMINATING (verified to FAIL on
HEAD~2's pre-fix code):

- `test_proximity_anchored_at_ball_not_wrist` — a hit member whose ball is ~600px off the
  player while its self-reported wrist xy sits ON a wrist. Asserts the emitted event's
  `dist_norm` is **large** (ball-anchored). Pre-fix this read **0.000** (circular
  wrist-on-wrist); post-fix it reads **6.002**. This is the exact mechanism the fix removes.
- `test_fallback_to_wrist_when_ball_untracked` — the ball is `None` at the hit frame, forcing
  `_xy_at`→None so proximity falls back to the wrist xy (graceful degradation). This branch
  is **dead on the committed methodo input** (0/13 hit frames have an untracked ball), so the
  test is the only place it is exercised — it locks the contract so a future sparse clip
  can't silently change behavior. (Closes the adversarial verifier's only open nice-to-have.)

---

## `near` plateau (scale-free robustness, no hardcoding)

Swept `near_wrist` on the canonical full-video dump (replay). The fix is robust across the
brief's required `0.4–1.0` band; the default **0.5 sits inside the stable plateau** and
delivers the contracted gain. (Default unchanged — the brief forbids new constants; retuning
the default is explicitly out of scope for S1.)

| near | bounce F1 | bounce R | hit F1 | hit R | H→B | B→H |
|---|---|---|---|---|---|---|
| 0.4 | 0.706 | 0.67 | 0.400 | 0.38 | 1 | 1 |
| **0.5 (default)** | **0.706** | **0.67** | **0.500** | **0.50** | **1** | **1** |
| 0.6 | 0.750 | 0.67 | 0.500 | 0.50 | 1 | 1 |
| 0.7 | 0.750 | 0.67 | 0.500 | 0.50 | 1 | 1 |
| 0.8 | 0.750 | 0.67 | 0.500 | 0.50 | 1 | 1 |
| 0.9 | 0.750 | 0.67 | 0.500 | 0.50 | 1 | 1 |
| 1.0 | 0.667 | 0.56 | 0.471 | 0.50 | 1 | 1 |

Confusion is invariant 1/1 across the whole band; bounce F1 is flat-to-improving on
`0.5–0.9` and only sags at the 0.4 / 1.0 extremes. **Observation (not shipped):** the
`0.6–0.9` zone reaches bounce F1 0.750 with hit F1 0.500 and confusion held — a possible
free +0.044 if the PM later decides to retune the default; left for a separate, explicitly-
scoped change since it's not the S1 deliverable and would introduce a tuned constant.

---

## What is NOT fixed here (honest scope) — the residual confusion 1/1

The canonical baseline already carries confusion **1/1** on the LIVE track; the fix
**maintains it byte-for-byte** (no regression, no new leak). Both confused pairs are
identical before and after the fix and are **out of S1 scope** (S2 / decoder territory) by
the brief's explicit instruction:

- **conf_H→B = 1 — event @271 ↔ GT shot@268.** Traced: `why=vy_flip`, `src=['turn']`,
  `vx_flip=None`, `vy_flip=True`, **ball-dmin = 1.58** (far from any wrist; nearest
  `detect_hit` is f259, 12 frames away — outside the merge window, so NO hit member is in
  this cluster). It is a real far-court HIT whose struck ball shows a vertical descend-then-
  ascend on the live track and anchors as a **mandatory pure BOUNCE** via `vy_bounce`. The
  ball-anchored proximity fix correctly does not touch it — there is no wrist-anchored hit
  member here; the issue is the `vy_flip`-as-unconditional-BOUNCE-anchor, which is the
  vx-gate / decoder layer (`courtside-event-methodo-prod-resolved`). Fixing it from S1 would
  require a hack the brief forbids.
- **conf_B→H = 1 — event @65 ↔ GT bounce@62.** The known residual the CR assigns to S2
  (cold-start, near the clip start where the ball track is youngest). Same family as the
  B308 parity-fill artifact.

Both are **upstream/decoder**, not proximity, and the firewall holds 0/0 on the cache (the
cache track simply doesn't expose these two live-track artifacts).

---

## Hard-constraint checklist

- [x] **confusion_H→B maintained** — identical pair (@271), pre-existing, byte-for-byte; not
      introduced or worsened by S1. Firewall logic untouched; cache holds 0/0.
- [x] **+1 B→H @ ~308/@65 left alone** — not "repaired" by a hack (S2 territory).
- [x] **Floors:** bounce F1 0.706 ≥ 0.625 ✅, hit F1 0.500 ≥ 0.500 ✅ (canonical input).
- [x] **Tests green:** all 4 (`test_event_confusion_regression` cache 0/0 + hit F1 0.737,
      `test_bounce_regression` felix ≥ 0.72, `test_sharp_turns`, `test_vx_veto`).
- [x] **Zero hardcoding:** no new constant; reuses scale-free box-normalized `near`; plateau
      `0.4–1.0` verified.
- [x] **Ship only if LIVE + cache agree:** they agree byte-for-byte on the event list.
- [x] **Touched UNIQUEMENT `vision/events.py`.**

---

## Files

- `vision/events.py` — `+ _prox_xy()` closure in `classify_events`; 3 proximity call-sites
  re-routed. +29 / −4. No public signature change. (`classify_events` args unchanged.)

## Integration notes for the PM

- Disjoint from S3 (`vision/bounce.py`) → parallel-safe to integrate.
- **Sequential before S2** (same file). S2's decoder / vy-anchor work should build on this
  HEAD (it fixes the @271 H→B and @65/@308 B→H residuals this CR documents).
- No new flag, no signature change, default behavior changed only in the proximity
  measurement (display unchanged) — drop-in.
