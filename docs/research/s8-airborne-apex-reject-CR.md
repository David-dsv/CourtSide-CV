# CR — S8: reject the AIRBORNE APEX (the phantom "rebond en haut / deep out compté")

**Branch:** `feat/s8-airborne-apex-reject` (from `feat/accuracy-overhaul` @65acdb7).
**Scope:** `vision/events.py` ONLY (70 insertions, 3 deletions, 1 file).
**Method:** ULTRACODE, measure-driven. Controlled WITH-vs-WITHOUT replays on the faithful
instrument (byte-matches `_stats.json`) + 2 LIVE full-video runs + scale-invariance proof.

## TL;DR

The user's hero render had a **phantom bounce at the top of the frame** ("double détection en
haut", "un deep qui est OUT mais quand même compté") = the **@231 phantom** (813,191), classified
`deep`, ball off-court. Measured: the ball **RISES** into f231 (y 224→219→…→191) then **comes back
down** (210→213→216…) — it is the **APEX of the ball's arc IN FLIGHT**, not a floor bounce. A
sharp-turn candidate fired there (a real velocity-vector turn exists at an apex) and the
alternation DP parity-filled it as a BOUNCE.

**Fix:** a sign-only **airborne-apex gate** in `vision/events.py` — the y-MAX mirror of the
existing y-MIN `vy_flip` floor-bounce signature. An apex (ball rises in, `vy_pre<0`, then settles/
descends, `vy_post>=0`) can never be labeled BOUNCE. Verified to fire on the phantom and on
**0/9 GT bounces** (every real bounce *descends into* the floor, `vy_pre>=0`).

**Result (canonical hero fixture, controlled replay):**

| metric | BEFORE (post-S7) | AFTER (S8) |
|---|---|---|
| pred BOUNCE | `[122,172,231,253,307,369,455,510,570]` | `[122,172,253,307,369,455,510,570]` |
| **@231 phantom** | **present** | **GONE** ✅ |
| spurious | `[(108,HIT),(231,BOUNCE),(487,HIT)]` (3) | `[(108,HIT),(487,HIT)]` (**2**) ✅ |
| **bounce F1** | **0.889** | **0.941** ✅ (P 0.89 → **1.00**) |
| hit F1 | 0.625 | 0.625 (held) ✅ |
| confusion H→B | 0 | **0** (firewall intact) ✅ |
| confusion B→H | 1 | 1 (unchanged — irreducible cold-start @65) |
| GT bounces detected | 9/9 | **9/9** (all preserved) ✅ |

The `(231,BOUNCE)` disappears, spurious drops to 2, and bounce F1 **rises** 0.889 → 0.941 (one FP
removed lifts precision to 1.00) — exactly the prompt's target, met and beaten.

## The diagnostic (verified by direct measurement — production slope reader)

At the rep frame f231 of the canonical post-S1 fixture, the production `_direction_features`
(half=5 @ 50fps, Theil-Sen slope, same gap/teleport guards as `vx_flip_veto`):

```
pre  y: 219 213 208 202 196 191   → vy_pre = -5.67   (ball RISING, image-y decreasing)
post y: 191 210 210 210 210 210   → vy_post = 0.0    (jumps down then PLATEAUS)
```

`vy_pre < 0` is the **unambiguous, robust discriminant**: the ball was rising INTO the candidate.
A floor bounce can never do this — it descends into the floor (`vy_pre > 0`) and then ascends
(`vy_post < 0`, = the existing `vy_flip`). The prompt's stated `vy_post = +6.3` came from a
different reader (k=3 median); under the production half-window Theil-Sen slope `vy_post` reads
~0 because of the post-apex 210-plateau — so the gate admits `vy_post >= 0` (settle OR descend),
and rests the decision on the rising-IN sign (`vy_pre < 0`).

### vy_pre/vy_post for ALL 9 predicted bounce rep frames (hero fixture)

| frame (GT) | vy_pre | vy_post | shape | apex? |
|---|---|---|---|---|
| 122 (b120) | 0.0 | −9.0 | descend/flat → ascend | no |
| 172 (b174) | +21.3 | +3.0 | descend → settle (knee) | no |
| **231 (PHANTOM)** | **−5.67** | **0.0** | **RISE → settle (APEX)** | **YES** |
| 253 (b255) | +4.5 | −66.7 | descend → ascend | no |
| 307 (b308) | +19.3 | +1.3 | descend → settle (knee) | no |
| 369 (b369) | +0.5 | −9.0 | descend → ascend | no |
| 455 (b456) | +21.3 | −5.5 | descend → ascend | no |
| 510 (b511) | +0.8 | −5.4 | descend → ascend | no |
| 570 (b569) | +20.8 | +2.0 | descend → settle (knee) | no |

**@231 is the ONLY predicted bounce with `vy_pre < 0`.** All 9 real GT bounces descend in
(`vy_pre >= 0`). The phantom is uniquely separable.

## The gate (exact)

`vision/events.py`, two parts:

**1. `_direction_features` — expose the signal (sign-mirror of `vy_flip`):**
```python
out["vy_pre"], out["vy_post"] = vy_pre, vy_post
out["vy_flip"]      = (... vy_pre > 0 and vy_post < 0)   # y-MIN floor bounce (creux)
out["airborne_apex"]= (vy_pre is not None and vy_post is not None
                       and vy_pre < 0 and vy_post >= 0)   # y-MAX arc apex (sommet)
```
Same half-window / gap-guard / teleport-drop / RAW-track+is_real path as `vx_flip_veto` and
`vy_flip` — so it inherits the exact safety invariant. Purely sign-based; **no pixel threshold.**

**2. `classify_events` — CLUSTER-LEVEL apex decision (the robustness fix):**
```python
rep_apex = bool(feat.get("airborne_apex"))
cluster_floor_evidence = any(
    mfeat[fr]["vy_flip"]
    or (mfeat[fr]["vy_pre"] is not None and mfeat[fr]["vy_pre"] > 0)   # descends-in
    for fr, _, _, _ in members)
airborne_apex = rep_apex and not cluster_floor_evidence
turn_b = turn and not airborne_apex     # apex gives ZERO bounce evidence
```
When `airborne_apex` is confirmed:
- `vy_bounce = ... and not airborne_apex` → forbids it auto-anchoring as a **MANDATORY** pure
  BOUNCE. This runs **BEFORE** the vy anchor (line ~640), so the un-skippable anchor is never
  posted and the DP parity is never corrupted (the prompt's hard requirement).
- `turn_b` (apex-gated) replaces raw `turn` in `bscore` **and** in `e["turn"]` (read by the
  firewall `_reward`'s "need positive B evidence" gate). With no bnc/vy/turn_b, `_reward(BOUNCE)`
  returns `None` → an apex can **NEVER** be a bounce, by the same LOGICAL firewall gate that holds
  `confusion_H→B == 0`. A real racket HIT at an apex keeps its hit evidence and is placed as a HIT.

### Why CLUSTER-level, not rep-frame-level (the adversarial finding that shaped the design)

A naive rep-frame-only gate is **unsafe**. The raw per-frame `airborne_apex` flag fires within ±8f
of **4/9 GT bounces on BOTH fixtures** — because the **rising-OUT leg just after a real bounce**
also reads `vy_pre<0, vy_post~0` (the ball is ascending then cresting). If a future track's cluster
rep frame landed on that rising-out frame, a rep-only gate would kill a real bounce.

The clean separator is the **whole cluster**: a real floor-bounce cluster ALWAYS contains floor
evidence — a `vy_flip` member (descend-then-ascend) OR a descending-in member (`vy_pre>0`). A pure
airborne apex has **neither**. Audited cluster-level decision:

```
HERO  apex clusters: [12,18], [231], [411,412]   → 0/9 overlap a GT bounce   SAFE
CACHE apex clusters: [288]                        → 0/9 overlap a GT bounce   SAFE
```

e.g. CACHE cluster `[63,64,65,69]` (GT b62) has an apex member (f69, the rising-out leg) BUT also
`vy_flip` + descending-in members → `cluster_floor_evidence=True` → **preserved**. The cluster rule
survives the rep frame landing anywhere on a real bounce arc.

## Proof: 0/9 GT bounces are airborne apexes

Two independent ball tracks (the hero live track + the frozen cache Kalman track), the exact gate
decision per cluster:

```
=== HERO: clusters DECIDED airborne_apex ===   [12,18] [231] [411,412]  → 0 overlap GT (SAFE)
=== CACHE: clusters DECIDED airborne_apex ===  [288]                    → 0 overlap GT (SAFE)
```

Every real bounce cluster carries floor evidence; not one is flagged. The gate is keyed on physics
(a sommet d'arc n'est pas un rebond), not on a clip.

## Generalization beyond the hero clip — the cache too

The same gate removes a **second** airborne-apex phantom on a totally different track: the frozen
demo3 event cache had a spurious BOUNCE @288 (`vy_pre=−3.4, vy_post=+16.0` — a clear y-MAX, off any
GT). S8 removes it:

| cache (test_event_confusion_regression) | BEFORE | AFTER |
|---|---|---|
| bounce F1 | 0.889 | **0.941** |
| hit F1 | 0.737 | 0.737 (held) |
| confusion H→B / B→H | 0 / 0 | **0 / 0** (held) |
| spurious BOUNCE @288 | present | **gone** |

One gate, two independent tracks, two phantom apexes removed, zero real bounces lost → the gate is
physics-anchored, not overfit.

## LIVE validation (≥2 runs) — honest no-op on the current MPS track

`venv/bin/python run_pipeline_8s.py /Users/vuong/Documents/CourtSide-CV/tennis.mp4 -s 73 -d 13
--device mps --match-mode -o /tmp/s8/out.mp4` (run from the worktree → resolves the worktree's
`vision/events.py`). **Run 1 ≡ Run 2 byte-identical** (MPS deterministic on this box).

The live MPS ball track at f231 is **smoother than the canonical fixture** (y 222→216→209→203→200→
201→203 — a mild apex), so **@231 is NOT a candidate on the live track** and there is no apex
phantom to remove there. Controlled WITH-vs-WITHOUT replay on the EXACT serialized live track:

| live run2 track | BASELINE | S8 |
|---|---|---|
| pred BOUNCE | `[126,150,254,300,320,369,455,508,574]` | **identical** |
| pred HIT | `[70,117,146,177,229,272,313,348,396,467]` | **identical** |
| confusion H→B / B→H | 0 / 4 | **0 / 4 (byte-identical)** |
| bounce F1 / hit F1 | 0.556 / 0.444 | **0.556 / 0.444 (byte-identical)** |

**S8 is a perfect no-op on the live track** (zero regression). The live spurious bounces
(126/150/300/320) are NOT apexes — measured `vy_pre = None` (the direction read abstains on the
sparse/gapped track); they are the **DP-placement-wall phantoms** documented in
`courtside-events-plateau-ball-tracking-wall` / `courtside-s3-event-recall-negative`, explicitly
OUT of S8 scope. S8 acts only where a confirmed airborne apex exists (the canonical fixture, which
the PM committed precisely because it reproduces the user's @231 phantom that a fresh live track
may smooth out). This mirrors S7's accepted pattern ("BYTE-IDENTICAL to no-knee on the noisy live
track; the win is on the canonical instrument").

## Constraints — all met

- **0 real bounces lost**: 9/9 GT preserved (hero & cache); cluster rule provably never flags a
  GT cluster (0/9 on both tracks).
- **confusion_H→B = 0 + firewall intact**: hero H→B=0, cache 0/0. The apex routes through the
  SAME logical `_reward` firewall (no new score, no bypass); it can't turn a bounce into a hit
  (it only removes BOUNCE evidence — a real bounce cluster keeps its floor evidence so its
  `vy`/`bnc` anchor stands).
- **Floors**: bounce F1 0.941 ≥ 0.889; hit F1 0.625 ≥ 0.625 (canonical) / 0.737 (cache).
- **felix + all tests green** (run from the worktree, main venv):
  `test_event_confusion_regression` (0/0, **bF1 0.941**), `test_bounce_regression` (felix
  F1 0.800), `test_bounce_wasb_regression` (0.865), `test_vx_veto`, `test_sharp_turns` (17/17 pool,
  5/5), `test_near_court_knee` (7/7), `test_static_fp_reject` (13/13), `test_ball_anchored_prox`.
- **Zero hardcoding / anti-overfit**: gate = a SIGN of slope (`vy_pre<0 ∧ vy_post>=0`) + a cluster
  evidence test (`vy_flip ∨ vy_pre>0`). Scale-invariance proven on synthetic apexes at 1×/2×/4×
  resolution (the flag holds; vy magnitudes scale but the sign pattern is invariant). No pixel
  threshold, no tuned band, no clip-specific frame. A y-MAX is a y-MAX at any resolution.

## Files
- `vision/events.py` — the only production change.
- Replay harnesses (under `/tmp/s8/`, not committed): `replay.py` (canonical fixture, re-derives
  turns with the S7 generators to match the 0.889 baseline), `cache_replay.py`,
  `cluster_decision_audit.py` (0/9 GT proof), `replay_live.py` (controlled live no-op proof).

## Out of scope (per prompt — do NOT do here)
The user also wants an "OUT" badge on real off-court bounces. That is **S9** (in/out calling), which
needs court boundaries — the hero clip has no homography (grazing angle; auto-court abstains). S8
removes only the apex phantom; the OUT badge waits on court calibration. No court/homography code
touched here.
