# CR — Event-methodo improvement (bounce/hit beyond F1 0.842/0.632)

**Branch:** `feat/event-methodo-improve` (from `feat/accuracy-overhaul`)
**Scope:** `vision/events.py` ONLY (the classifier) + the committed demo3 cache + tests.
**Session:** 2/3, autonomous ultracode.

## Headline

| metric | baseline | **this branch** | Δ |
|---|---|---|---|
| confusion H→B | 0 | **0** | held (non-negotiable) |
| confusion B→H | 0 | **0** | held |
| bounce F1 | 0.842 | **0.900** | **+0.058** (recall 8/9 → 9/9, **f174 recovered**) |
| hit F1 | 0.632 | **0.700** | **+0.068** (recall 6/8 → 7/8, **f525 recovered**) |
| mean F1 | 0.737 | **0.800** | +0.063 |

Both class F1s rise, the firewall stays 0/0, and the gain is **plateau-stable**
(0/0 + 0.900/0.700 hold across `near_wrist` 0.4–0.6 — structural, not knife-edge).
All 7 tests green (4 felix/regression + 3 demo3); felix bounce paths byte-untouched.

## Iteration journal (measured on `tools/event_eval/run_demo3.py`)

| # | change | bounce F1 | hit F1 | H→B / B→H | note |
|---|---|---|---|---|---|
| 0 | baseline (old 78%-far cache) | 0.842 | 0.632 | 0 / 0 | miss: b f174, h f333/f525 |
| 1 | far-wrist generator (old cache) | 0.842 | 0.632 | 0 / 0 | no-op: old far-select = a spectator |
| 2 | **rebuild cache** (far-pose 78%→100%) + far-wrist | 0.842 | **0.737** | 0 / 0 | **f525 recovered** |
| 3 | + both-sides wrist scan | 0.737 | 0.727 | 0 / **1** | f333 recovered but **B→H breaks** → reverted to far-only |
| 4 | + y-MAX proximity relaxation | **0.900** | 0.700 | 0 / 0 | **f174 recovered**; f162 promoted (−1 hit) |
| 5 | edge-artifact bounce DEMOTION | 0.900 | 0.600 | **1** / 0 | shifted DP → firewall broke → **reverted** |
| — | **final** (2 + 4) | **0.900** | **0.700** | 0 / 0 | locked |

## Diagnosis — where the three named misses actually live

- **f333 (hit):** IN the turn pool, NOT in `hit_cands`. It is a **NEAR hit** the ball
  reaches at the near player's wrist (dwrist 0.25) with a clear ~20px swing — but
  `detect_hits` drops it (its ±ball-window stitches the f159 dropout's ball position
  to the f162 wrist peak; an internal NMS/bookkeeping bug). Recovering it needs the
  NEAR-side scan, which **breaks the firewall** (see below) → left as the one miss.
- **f525 (hit):** a **FAR hit**. `detect_hits` emits ZERO far hits because its
  amplitude floor is FRAME-scaled (`0.008·fh ≈ 8.6px`), and the far player is ~10×
  smaller so a full far swing only moves the wrist ~3px/frame — below the floor.
  **Recovered** by the box-height-normalized far-wrist generator.
- **f174 (bounce):** IN the turn pool (apex turn at the y-maximum f179). A
  **classification** block, not a generation gap: the floor bounce lands where the
  RECEIVING near player stands (ball at racket height, dwrist 0.18), so the
  `dmin < near` proximity term of `hit_like` forbids it from BOUNCE — even though
  the player is NOT swinging there (norm wrist-speed 0.029 vs 0.10 at a real hit).
  **Recovered** by the y-maximum proximity relaxation.

## Fix 1 — far-court wrist-hit candidate generator (THE reusable idea)

`detect_wrist_hits` (new): the far player's racket-wrist speed **normalized by its
own box height** (scale-free), local-peaked, gated on the ball being within
`near_box_h` box-heights of that wrist. Wired into `classify_events`
(`far_wrist_hits=True`, no call-site change in prod), deduped vs existing hit AND
bounce candidates.

**Firewall safety (by construction):** such a candidate sets `hit=True` in its merge
cluster ⇒ `hit_like=True` ⇒ `_reward(BOUNCE)` returns `None`. It can ONLY ever be
labeled HIT — confusion-into-bounce is impossible. The ball-at-wrist gate is the
precision mechanism (no contact geometry ⇒ no candidate). NEAR scanning is opt-in
(`sides=("near","far")`): it recovers f333 but a near swing also fires next to the
far-court bounce f174 (no vx/vy reversal yet at the swing-peak frame) and the merged
hit candidate flips that turn-bounce to HIT (**B→H=1**, measured). So default = FAR
only.

## Fix 2 — y-maximum proximity relaxation (recovers f174)

The `hit_like` proximity term `dmin < near` is suppressed for an event that carries
POSITIVE bounce-trajectory evidence and NO swing: a `turn` sitting at a local
y-MAXIMUM (`_is_y_maximum` — the ball's lowest screen point, where a floor bounce
happens) OR with vx PRESERVED, and no hit source. `_is_edge_artifact` guards the
y-max read against tracking dropouts (ball off the bottom edge + a teleport, demo3
f159: y=1060/1080 then a 696px jump).

**Firewall safety:** a real racket HIT *requires a swing* ⇒ `detect_hits` or the
far-wrist generator gives it a hit source ⇒ `hit=True` ⇒ it stays `hit_like` and is
never relaxed; OR it reverses vx (vx_flip True, not False) ⇒ not relaxed. The
relaxation only ever reaches a SWING-LESS event, and a swing-less event at a
y-maximum next to a stationary player is a floor bounce, not a contact. The
alternation DP is the second guard. **0/0 verified across the `near_wrist` 0.4–0.6
plateau** (`test_firewall_plateau_over_near_wrist`).

## Cache regeneration (documented, as the prompt allows)

The committed `tests/fixtures/cache/demo3_event.json` was rebuilt with
`scripts/build_demo3_event_cache.py` so `players_per_frame` carries the merged
far-select's dense pose: **far-player coverage 78.6% → 100%** (gap-filled by the
TwoPlayerTracker; `locked: P1(far) 100% P2(near) 100%`). The Kalman and WASB ball
tracks are byte-identical (both deterministic: Kalman 441/650, WASB 68/650), which
**isolates the cache change to the proximity/wrist features** — exactly the
substrate the far-wrist generator needs. The far-wrist generator was a verified
NO-OP on the old 78%-far cache (its far slot was a top-right spectator), so the gain
is attributable to the dense far pose, not a cache artifact. Old cache backed up to
`/tmp/demo3_event_OLD78.json`.

## What did NOT work (and why — anti-overfit honesty)

- **NEAR-side wrist scan** (would recover f333): breaks the firewall (B→H=1) — a
  bounce-near-player is geometrically a near-wrist contact at the swing-peak frame.
- **Edge-artifact bounce DEMOTION** (would kill the spurious f159 bounce): removing
  one spurious bounce shifts the rally-alternation parity and **flips a real hit to
  bounce (H→B=1)** — the DP is delicately balanced around the tracking glitch.
- The residual spurious FPs (bounces f159/f288; hits f162/f351/f581/f622) are
  **ball-tracker / detect_hits glitch artifacts**, not classifier errors. They can't
  be removed from `vision/events.py` without breaking the firewall (proven above).
  Fixing them is upstream work (ball tracking / detect_hits NMS), out of this scope.

## Regression proof

`venv/bin/python tests/<name>.py` — all green:
`test_event_confusion_regression` (now floors **bounce 0.88 / hit 0.68**),
`test_wrist_hits_and_relaxation` (new, 6 cases), `test_sharp_turns`,
`test_bounce_regression` (felix 0.800), `test_bounce_wasb_regression` (felix 0.865),
`test_vx_veto` (felix 0.9), `test_far_coverage` (84.4%). `classify_events` is
no-crash / no-event on poseless or None-slot input (felix prod path safe).

## Most reusable takeaway

**A frame-scaled magnitude threshold is blind to a perspective-small subject.**
`detect_hits`' `0.008·frame_height` wrist-speed floor structurally cannot see the
far player; the fix is to normalize the motion by the SUBJECT's own size (box
height), not the frame's. The same lesson applies anywhere a per-frame magnitude
gate meets a foreground/background size disparity. Corollary for the firewall: a
proximity-only `hit_like` over-fires on a bounce-at-a-player; gate it on a SWING
(positive contact evidence), because a ball near a non-swinging wrist is a bounce,
not a hit.
