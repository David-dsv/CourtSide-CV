# CR — Far-select A: low far-conf + static-spectator rejection → unified `hcv` selector

**Branch:** `feat/farselect-conf-motion` (worktree `CourtSide-CV-farA`, from `feat/accuracy-overhaul`)
**Scope touched:** `vision/pose.py` only (+ throwaway measurement scripts under `tools/pose_gt/`).
**Session:** 1/5 (autonomous, no questions asked).

## Problem (already diagnosed before this session)

In `vision/pose.py`, the far/P1 player slot locked onto a STATIC spectator/structure
above the net. Measured on the human pose GT (`tests/fixtures/pose_gt/tennis_demo3.pose_gt.json`,
225 verified frames): a far box was present 100% of the time but on the **TRUE** far
player only **0.9%** (2/225), mean **572 px** off. The far player IS detectable
(generic YOLO finds it at conf 0.08 / imgsz 1280). It is a **SELECTION** bug, not a
detection / pose / re-train problem.

## What broke the naive fixes (the felix trap)

The two test clips have **structurally OPPOSITE** far-player geometry:

| clip | camera | far player xr | far player size | poses (valid)? | the villain |
|---|---|---|---|---|---|
| **demo3** | steep broadcast | **central** (~0.42) | tiny (h≈0.11·H) | **no** (too small) | lateral lower-stands spectator (xr 0.80, valid sometimes) |
| **felix** | corner / grazing | **off-centre** (~0.71–0.80) | bigger (h≈0.06–0.07·H) | **yes** (conf 0.80–0.87) | central static structure (xr 0.40, h 0.03·H) |

So:
- **Pure centrality** (the previously-committed `c50b00b`) → demo3 **84.9%** but felix **4.5%** (it picks felix's central structure).
- **Pure keypoint-validity tier** → felix 98.5% but demo3 **26.7%** (demo3's player can't pose; a valid spectator wins).

Neither generalises. A score that wins one clip is overfit to that clip's geometry.

## Iteration journal (REAL measured numbers, far-CORRECT within 0.10·W of GT)

Method change that unblocked iteration: I built a **posed-candidate cache**
(`tools/pose_gt/_strat_sweep.py` → `/tmp/strat_cache.json`, 225 demo3 + 66 felix
frames, every above-net candidate posed once). After that, every strategy is
evaluated **instantly** (no re-posing). felix has no human GT, so a **proxy-GT**
(`_build_felix_gt.py`, 66 frames) is built from the OLD base-branch selector's
RELIABLE felix picks (only frames where OLD returned a real skeleton, ≥10 kpts).

| # | strategy | demo3 | felix | min | note |
|---|---|---|---|---|---|
| baseline | (pre-session) | 0.9% | — | — | static spectator lock |
| 0 | `central` (committed c50b00b) | 84.9% | **4.5%** | 4.5% | felix regression — the blocker |
| 1 | `valid → central` (two-tier, broken) | 26.7% | 98.5% | 26.7% | tier-1 admits valid spectator on demo3 |
| 2 | `valid+not-lateral → central` | 37.3% | 84.8% | 37.3% | lateral gate helps, still split |
| 3 | `blend` (geom + wv·valid), best wv=0.5 | 56.4% | 65.2% | 56.4% | compromise hurts both |
| 4 | **`height` (tallest on-court)** | 63.6% | 89.4% | 63.6% | the cross-geometry signal appears |
| 5 | `hcv` wc=0.6 (h+conf+central+valid) | 71.6% | 95.5% | 71.6% | |
| 6 | `hcv` wc=1.0 | 79.6% | 87.9% | 79.6% | |
| 7 | **`hcv` wc=1.2 (CHOSEN)** | **84.4%** | **87.9%** | **84.4%** | mid-plateau |
| 8 | `hcv` wc=1.3 | 84.9% | 86.4% | 84.9% | demo3 ceiling, felix margin thinner |
| 9 | `hcv` wc=1.5 | 84.9% | 78.8% | 78.8% | centrality starts breaking felix → cliff |

**Authoritative LIVE numbers (real `estimate_players_pose`, not cache replay):**
- demo3: far DETECTED 225/225, far **CORRECT 191/225 = 84.9%**, mean 121 px
  (`tools/pose_gt/measure_far_coverage.py`).
- felix: far DETECTED 66/66, far **CORRECT 58/66 = 87.9%**, mean 91 px
  (`tools/pose_gt/_measure_felix.py`).

The live demo3 number (84.9%) **matches/exceeds** the cache prediction (84.4%) — the
gain reproduces on the production path, not just on a frozen cache. (This is the
lesson from `courtside-event-methodo-prod-gap`: validate on a LIVE run.)

## The selector that ships

A SINGLE unified score over on-court above-net candidates (replaces the broken
two-tier). For each far candidate, with `hmax` = the tallest on-court far box this frame:

```
on-court := cy >= 0.25·net_y  AND  0.12 < cx/W < 0.88
S = 1.0·(h / hmax) + 0.3·conf + 1.2·central + 0.3·valid          # central = 1 − |xr−0.5|·2
```

Pick `argmax S` among on-court candidates (fall back to all far boxes if none qualify).
`valid` (a real ≥10-kpt skeleton) is a **soft weighted term, never a hard reject** —
that is the fix for the keypoint-gate backfire.

Why each term, grounded in the measured candidate structure:
- **height/hmax** — the load-bearing cross-geometry signal: the real far PLAYER is the
  tallest on-court above-net box on BOTH clips (a distant *player* > a distant *seated
  spectator*). Normalised → scale-free.
- **central (wc=1.2)** — demo3's player is central; felix's tolerates it. The plateau
  shows demo3 needs it and felix survives it up to ~1.3.
- **conf** — the real player detects more confidently than structures.
- **valid (0.3)** — **load-bearing for felix**: removing it drops felix 87.9% → **62.1%**
  (its real player poses; its structure villain doesn't).

## Proof felix is NOT regressed

- felix far-CORRECT **87.9%** live on the proxy-GT (built from OLD's reliable picks),
  vs the committed `central` code's **4.5%**. 87.9% agreement with OLD = NEW essentially
  reproduces OLD's good felix behaviour **while** fixing demo3 from 0.9% → 84.9%.
- The 5 mandated regression tests all PASS, byte-identical (the far-selection change
  touches no bounce/event path):
  `test_bounce_regression` F1 0.800 · `test_bounce_wasb_regression` F1 0.865 ·
  `test_vx_veto` floor 0.9 · `test_event_confusion_regression` H→B 0 / B→H 0, bounce
  F1 0.842 · `test_sharp_turns` all pass.

## Honest limits & overfit risk

- **Plateau, not a knife-edge** (anti-overfit evidence): a ±grid around the pick —
  wc∈{1.1,1.2,1.25}, wh∈{0.8,1.0}, wconf∈{0.2,0.4}, wval∈{0.2,0.3} — all stay **≥80%**
  on both clips. The choice is not balanced on a single tuned constant. (wval=0 and
  wc≥1.5 do fall off — those terms are genuinely required, not decoration.)
- **demo3 ceiling is 89.3%, not 100%**: 24/225 GT frames have **no candidate** near the
  true player (a pure DETECTION gap, unrecoverable by ANY selector). 84.9% = **95% of
  the achievable ceiling**. Closing the rest needs better far DETECTION, not selection.
- **felix GT is a PROXY** (from OLD's picks), 66 frames sampled every 50. It validates
  "NEW ≈ OLD on felix + much better on demo3", but is not a human GT. Don't
  over-optimise to it (I deliberately stopped at the plateau, not at felix's max).
- **Two clips only.** Zero-hardcoding is respected (every threshold is a ratio of
  frame/net_y, no demo3 pixels), but generalisation to a 3rd geometry is unproven.
  The on-court lateral band (0.12–0.88) is loose by design so a moderately off-centre
  player on a new angle still qualifies.
- **The original angle-A ideas were measured and dropped on evidence**: lowering far
  det-conf to 0.08 is KEPT (it makes the far player detectable). **Motion / static-
  spectator rejection HURTS on demo3** (the tiny far player barely moves → reads as
  "static"), so it stays OFF by default behind `track_state` — available, not wired.

## Single most reusable idea (for the integrator)

> **The real far player is the TALLEST on-court above-net box** — a scale-free,
> camera-geometry-agnostic signal that beats centrality (which is geometry-specific and
> breaks corner angles). Combine height (normalised by the frame's tallest on-court far
> box) + a SOFT validity bonus (never a hard keypoint reject) + mild centrality/conf.
> The unified score `h/hmax + 0.3·conf + 1.2·central + 0.3·valid` over an on-court gate
> wins BOTH the steep-broadcast (central, tiny, can't-pose) and corner-grazing
> (off-centre, bigger, poses) far player from one formula. Validity must be weighted,
> not gated.
