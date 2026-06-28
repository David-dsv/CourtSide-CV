# CR — S2: less-coupled decoder + the residual confusions (OPT-IN, post-S1)

**Branch:** `feat/s2-less-coupled-decoder` (from `feat/accuracy-overhaul` @4099645, post-S1+S3).
**Touches:** `vision/events.py` only (+ two measurement tools in `tools/event_eval/`).
**Method:** ULTRACODE, measure-driven. Apples-to-apples decoder comparison on the committed
canonical post-S1 dump; a 6-candidate fix search dual-gated (cache regression + live accuracy)
and adversarially verified; final confirmation on a LIVE full-video `_stats.json`.

## TL;DR

1. **The default decoder flip stays FORBIDDEN, as the prompt measured.** On the committed
   cache the anchor decoder regresses (bounce F1 0.889→0.714, hit F1 0.737→0.500 — fails
   `test_event_confusion_regression`); live it drops 5 bounces. Confirmed.
2. **A "less-coupled decoder" does NOT fix either residual on the canonical input.** Swapping
   to the anchor decoder fixes the cold-start B→H@65 but leaves the dominant H→B@271 untouched
   (it is a *feature/anchor* artifact, not a decoder-coupling one) and costs bounce recall.
3. **The dominant residual H→B@271 is fixed by a targeted, cache-safe FEATURE gate**, not a
   decoder change: an at-wrist vy-flip with no ball-only bounce corroboration is demoted from a
   mandatory pure-bounce anchor. **LIVE: H→B 1→0, hit F1 0.400→0.625, bounce F1 0.706→0.750**,
   the committed cache **byte-identical** (0/0, 0.889/0.737), felix intact.
4. **The residual B→H@65 is IRREDUCIBLE in the global DP** (a cross-track trap; all three
   cold-start levers are NO-GO; the DP reroutes around any cold-start veto). Out of scope, left
   honest. (It is not @308 — that frame number came from the degraded extract; on the canonical
   input the two residuals are H→B@271 and B→H@65, measured below.)

## The canonical input + measured residuals (this is what I optimized against)

Benchmarks on **`tennis.mp4 -s 73 -d 13`** (the FULLSTACK invocation), via the committed dump
`tests/fixtures/methodo/demo3_methodo_inputs_fullvideo_postS1.json`. Post-S1 baseline (global
decoder), reproduced exactly:

| metric | post-S1 baseline (global) |
|---|---|
| pool recall | 14/17 (NO CANDIDATE: B174, B308, H394 — candidate-gen, out of scope) |
| confusion H→B / B→H | **1 / 1** |
| bounce F1 | 0.706 (R 0.67) |
| hit F1 | 0.400 (R 0.38) |
| conf H→B | pred BOUNCE@271 stole **GT_H@268** (a real far HIT) |
| conf B→H | pred HIT@65 stole **GT_B@62** (cold-start phantom) |
| spurious | (108,HIT),(231,BOUNCE),(292,HIT),(487,HIT) |

## Why a decoder swap is the wrong layer (apples-to-apples on the SAME dump)

`classify_events` already exposes `decoder=` (global default; `_anchor_parity_decode` exists).
On the **identical** canonical dump:

| decoder | CACHE bF1/hF1 | CACHE gate | LIVE H→B | LIVE B→H | LIVE bF1 | LIVE hF1 |
|---|---|---|---|---|---|---|
| **global** (default) | 0.889 / 0.737 | **PASS** | 1 | 1 | 0.706 | 0.400 |
| anchor (raw) | 0.714 / 0.500 | **FAIL** | 1 | **0** | 0.571 | 0.615 |

- The anchor decoder **fixes B→H@65** (drops the phantom) but **does NOT fix H→B@271** (both
  decoders honor f271's MANDATORY vy-bounce anchor — the bug is upstream of the decoder).
- It **fails the cache gate** and **regresses live bounce recall** (drops B62, B120, B569 —
  they have no committed anchor; the segment-local decoder cannot place unanchored edge events).

So a decoder swap trades the wrong things. The cross-track disagreement (live anchor +hit-F1 /
cache anchor −F1) is itself a non-generalization signal. **Default-flip forbidden — confirmed.**

## Root cause of each residual (per-event, from the internal `evs` state)

### H→B@271 — a real far HIT that fakes a floor bounce (FIXED)
`f271: src=[turn] vy=True vx_flip=None(abstains) dmin=0.29(AT a wrist) hit=0 bnc=0` → it is
GT_H@268. The committed `vx_flip_veto` physics (a struck ball reverses horizontal; a floor
bounce preserves it) is INERT here because `vx_flip` ABSTAINS (the far-court ball's horizontal
motion is foreshortened / sub-deadband). So `vy_bounce` stays True → `anchor=BOUNCE`,
**MANDATORY** → the firewall is bypassed and both decoders emit BOUNCE.

The separator is the **independent ball-only bounce detector** `bnc`: a genuine floor bounce is
either OFF any wrist (`dmin >= near`) or, if it lands near a player's feet, is ALSO seen by
`detect_bounces_robust` (`bnc=True`). A far racket HIT that fakes a vy-flip sits AT the racket
(`dmin < near`) with `bnc=False`. Measured on the at-wrist vy-anchors:

| frame | dmin | bnc | GT | meaning |
|---|---|---|---|---|
| LIVE f271 | 0.29 | **0** | GT_H@268 | far HIT faking a bounce → **demote** |
| LIVE f456 | 0.16 | **1** | GT_B@456 | real bounce near feet → **keep** |
| CACHE f456 | 0.21 | **1** | GT_B@456 | real bounce near feet → **keep** |

### B→H@65 — cold-start phantom hit (IRREDUCIBLE, out of scope)
`f65: src=[hit] hit_like=True turn=0 dmin=1.45(ball OFF the wrist)` near GT_B@62. It is the
ONLY candidate near GT_B@62 (a candidate-gen gap for that bounce). The firewall correctly
forbids it→BOUNCE; the global DP labels it HIT to seed the chain → it lands on GT_B@62.

**The cross-track trap (why no feature/cold-start rule fixes it):** on the LIVE track real hits
have `dmin<near` and phantoms `dmin>=far`, a clean separator — BUT on the CACHE track two REAL
hits (GT_H@268=f270 dmin=2.24, GT_H@394=f396 dmin=2.39) ALSO sit at `dmin>=far` and are
correctly labeled HIT. So any rule of the form "hit + far dmin (+ no turn) ⇒ not a HIT / drop"
KILLS cache f270/f396 and fails the gate. And any **cold-start veto** is rerouted by the
globally-coupled DP (it just seeds the chain one event earlier and reaches f65 as a same-label
missed-beat continuation). Measured: all three cold-start levers are NO-GO (below).

## The fix (targeted, in the global path, generalizes)

`vision/events.py`, right after the existing `vx_flip_veto` line in `classify_events`:

```python
_vxf = feat.get("vx_flip")
vy_bounce = bool(vy) and not (_vxf is not None and bool(_vxf))
# at-wrist corroboration gate (the vx-gate's sibling for the abstain case):
if vy_bounce and dmin < near and not bnc:
    vy_bounce = False
```

A vy-flip AT a wrist (`dmin < near`) is only a GOLD floor bounce if the ball-only detector also
fired (`bnc`). Demoting `vy_bounce` routes f271 back through `hit_like` (since `dmin<near`), so
the firewall forbids BOUNCE and the alternation places it as the HIT it is.

- **Scale-free:** keyed on `near_wrist` (a box-height-normalized fraction) and the independent
  `bnc` signal. No new constant, no clip-specific frame/threshold.
- **Strengthens the firewall** (it can only ever remove a BOUNCE anchor, never add one) → it
  cannot create H→B by construction.
- **vx-gate sibling:** vx-gate catches racket redirects via horizontal reversal; this catches
  the ones whose vx is unreadable (abstains) but whose ball sits at the racket.

### Measured effect (all three gates)

| gate | metric | before | after |
|---|---|---|---|
| **CACHE** (`test_event_confusion_regression`) | H→B / B→H | 0 / 0 | **0 / 0** |
| | bounce F1 / hit F1 | 0.889 / 0.737 | **0.889 / 0.737** (byte-identical) |
| | f270 & f396 still HIT | yes | **yes** (untouched — opposite end of dmin) |
| **LIVE** (`_stats.json`, `score_stats.py`) | confusion **H→B** | **1** | **0** |
| | confusion B→H | 1 | 1 (@65, out of scope) |
| | bounce F1 | 0.706 | **0.750** |
| | hit F1 | 0.400 | **0.625** |
| | spurious | 4 | 3 |
| **FELIX** (`test_bounce_regression`, `…_wasb_…`) | F1 | 0.800 / 0.865 | **0.800 / 0.865** (events.py is not on the felix path) |

The LIVE full-video `_stats.json` (the final arbiter) reproduced the offline dump prediction
exactly: H→B=0, B→H=1, bounce F1 0.750, hit F1 0.625, spurious [(108,HIT),(231,BOUNCE),
(487,HIT)]. f271 is now correctly emitted as HIT.

## The 6-candidate fix search (dual-gated, adversarially verified)

| candidate | cache gate | live H→B | live B→H | live bF1 | live hF1 | fixes65 | fixes271 | verdict |
|---|---|---|---|---|---|---|---|---|
| cold-start-corrob | PASS | 1 | 1 | 0.706 | 0.375 | ✗ | ✗ | NO-GO |
| cold-start-far-veto | PASS | 1 | 1 | 0.706 | 0.375 | ✗ | ✗ | NO-GO |
| cold-start-best-start | PASS | 1 | 1 | 0.706 | 0.375 | ✗ | ✗ | NO-GO |
| anchor-edge-recover (anchor decoder) | PASS @floor 0.80 | 0 | 0 | 0.800 | 0.714 | ✓ | ✓ | **inert in prod** |
| **at271-vy-anchor-refine** | **PASS 0.889** | **0** | 1 | **0.750** | **0.625** | ✗ | ✓ | **SHIP** |
| at271-vyflip-vxgate-extend | PASS 0.889 | 0 | 1 | 0.750 | 0.625 | ✗ | ✓ | SHIP (= at271) |

Two independent agents converged on **byte-identical** code for the shipped fix
(`vy_bounce and dmin<near and not bnc → demote`). Adversarial verifier on it: `claim_holds=true`,
`scale_free=true`, `generalizes=likely`, f270/f396 preserved, firewall strengthened.

**Why anchor-edge-recover is NOT shipped (adversarial verifier caught it):** the variant rewrites
only `_anchor_parity_decode` but leaves `decoder="global"` the default — so it is **inert in
production** and the regression test (which calls `classify_events` with no decoder arg) exercises
none of its code; its numbers only appear under `--decoder anchor`. It also sits **exactly on the
cache floor (bF1=0.800, zero margin)** and drops cache f270/f396. A real prod fix via that path
would require flipping the default decoder (forbidden) AND re-pointing the test — net worse than
the targeted feature gate.

## Recommendation

1. **SHIP the targeted at-wrist bnc-corroboration gate** (this branch). It fixes the dominant
   live H→B@271 on the *default global decoder*, with the cache byte-identical, felix intact, a
   strengthened firewall, and zero new constants. LIVE-confirmed on `_stats.json`.
2. **Do NOT flip the decoder default.** The anchor decoder fails the cache gate raw and the only
   variant that passes is inert in prod / floor-margin-zero. The `decoder="anchor"` param remains
   available to callers as an experimental opt-in but is **not recommended** — it does not
   generalize (cross-track F1 disagreement) and does not fix H→B@271.
3. **Leave B→H@65 (and the candidate-gen misses B174/B308/H394) for upstream.** B→H@65 is
   irreducible in the global DP (cross-track trap + DP reroutes vetoes); the bounce misses are
   candidate generation (S1/ball-track territory), not the decoder.

## Tools added (`tools/event_eval/`)
- `dual_score.py` — scores `events.py` on BOTH the committed cache AND the canonical live dump in
  one call (force-loads the variant by path → worktree-safe), with a CACHE GATE PASS/FAIL verdict.
- `decoder_probe.py` — apples-to-apples global vs anchor on one dump + per-event internal state
  dump (anchor / vy / hit_like / dmin / label), the instrument behind the root-cause table above.

## Reusable lessons
1. **Match the residual to its LAYER before picking the tool.** The prompt anticipated a *decoder*
   artifact; the dominant residual was a *feature/anchor* artifact, fixable without touching the
   decoder at all — and the decoder swap would have made it worse.
2. **The cache lies in BOTH directions and DIFFERENTLY per feature.** The dmin separator that is
   clean on the live track is ambiguous on the cache (real far hits at dmin≥far). Validate every
   feature gate on BOTH tracks; ship only when both agree.
3. **A globally-coupled DP reroutes single-position vetoes.** Cold-start/edge vetoes don't stick;
   the fix has to change the event's *evidence*, not patch the decoder's bookkeeping.
4. **Validate on a LIVE `_stats.json`, not a frozen cache.** Done — the live arbiter reproduced
   the dump prediction exactly.
