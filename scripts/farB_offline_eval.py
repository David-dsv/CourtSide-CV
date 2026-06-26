"""THROWAWAY diagnostic harness for far-player SELECTION (session farselect B).

Fast offline eval: loads the committed demo3 person-box + ball-candidate cache and
the human pose GT, then scores how often a given FAR-SELECTION rule picks a box
within 10%W of the true far player. No YOLO, no video decode → sub-second per rule.

Why this is faithful: the diagnosis (memory courtside-far-player-selection-not-detection)
proved the far player is ALREADY in the candidate set (measured here: a person box
lands within 192px of the true far player on 91.6% of frames even in the conf0.2/640
cache). So far-CORRECT is governed by WHICH above-net box the rule selects — exactly
what this harness measures. The end-to-end production scorer (imgsz 1280) is the final
gate; this is the fast inner loop.

Usage:  python scripts/farB_offline_eval.py            # baseline + all rules
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_raw_dets.json"
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"


def load():
    cache = json.load(open(CACHE))
    gt = json.load(open(GT))
    fw, fh = cache["width"], cache["height"]
    pb = cache["person_boxes"]            # [[x1,y1,x2,y2,conf], ...] per frame
    rb = cache["raw_ball_candidates"]     # [[cx,cy,score], ...] per frame
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    return pb, rb, fw, fh, gtbf


def above_net(b, net_y):
    return (b[1] + b[3]) / 2.0 < net_y


def score_rule(rule, pb, rb, fw, fh, gtbf, net_y_frac=0.5):
    """rule(cands, ball_cands, fw, fh, net_y, ctx) -> chosen box (x1,y1,x2,y2,conf) or None.
    Returns (correct_pct, detected_pct, mean_dist_when_chosen)."""
    net_y = net_y_frac * fh
    correct = chosen = checked = 0
    dists = []
    for fidx, (gx, gy) in gtbf.items():
        if fidx >= len(pb):
            continue
        checked += 1
        boxes = [np.array(b, dtype=float) for b in pb[fidx]]
        ball = [np.array(c, dtype=float) for c in (rb[fidx] if fidx < len(rb) else [])]
        ctx = {"fidx": fidx, "pb": pb, "rb": rb}
        pick = rule(boxes, ball, fw, fh, net_y, ctx)
        if pick is None:
            continue
        chosen += 1
        cx = (pick[0] + pick[2]) / 2.0
        cy = (pick[1] + pick[3]) / 2.0
        d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
        dists.append(d)
        if d < 0.10 * fw:
            correct += 1
    cpct = 100.0 * correct / max(1, checked)
    dpct = 100.0 * chosen / max(1, checked)
    md = float(np.mean(dists)) if dists else float("nan")
    return cpct, dpct, md, checked


# ---------------------------------------------------------------------------
# RULES. Each takes the per-frame above/below-net candidates and returns the
# chosen FAR box. Geometry-only, derived from the scene (zero hardcoding).
# ---------------------------------------------------------------------------

def rule_baseline_biggest_conf(boxes, ball, fw, fh, net_y, ctx):
    """Mimics the pipeline's bias: among above-net boxes, pick max h*conf
    (the spectator wins because it's big/sharp). This reproduces ~baseline."""
    far = [b for b in boxes if above_net(b, net_y)]
    if not far:
        return None
    return max(far, key=lambda b: (b[3] - b[1]) * b[4])


def rule_central_conf(boxes, ball, fw, fh, net_y, ctx):
    """Prefer above-net boxes that are horizontally central AND confident.
    Singles play happens in the central horizontal zone; spectators sprawl to
    the edges. Central derived as |cx/W - 0.5| (no pixel constant)."""
    far = [b for b in boxes if above_net(b, net_y)]
    if not far:
        return None
    def s(b):
        cx = (b[0] + b[2]) / 2 / fw
        central = 1.0 - 2.0 * abs(cx - 0.5)        # 1 at center, 0 at edges
        return central * b[4]
    return max(far, key=s)


# --- court-band derivation (approach B core) ----------------------------------

def derive_far_band(all_boxes_below_above, fw, fh, net_y):
    """Derive the FAR court band (where the far player's feet land) from scene
    geometry, zero-hardcoded. The far player stands just below the far baseline;
    the near player just below the near baseline. We use the BELOW-net boxes
    (near player + near-side parasites) to anchor the near baseline, and the
    inter-player span sets the perspective scale. Returns (top_y, bot_y) of the
    plausible far-feet band. Falls back to a broadcast-geometry default."""
    # not used directly in scoring helper; band logic lives inside the rules below
    raise NotImplementedError


def _far_static_penalty(ctx, b, fw, fh, net_y, win=12, cell=0.04):
    """Static-occupancy penalty: how often a box at THIS grid cell appears in a
    temporal window around the current frame. A fixture/spectator recurs at the
    same cell; a moving player does not. Geometry+occupancy, derived from the
    scene, not a pixel constant. cell = grid size as a frame fraction."""
    pb = ctx["pb"]
    fidx = ctx["fidx"]
    gx = round((b[0] + b[2]) / 2 / (cell * fw))
    gy = round((b[1] + b[3]) / 2 / (cell * fh))
    cnt = 0
    lo, hi = max(0, fidx - win), min(len(pb), fidx + win + 1)
    for f in range(lo, hi):
        if f == fidx:
            continue
        for bb in pb[f]:
            if (bb[1] + bb[3]) / 2 >= net_y:
                continue
            if round((bb[0] + bb[2]) / 2 / (cell * fw)) == gx and \
               round((bb[1] + bb[3]) / 2 / (cell * fh)) == gy:
                cnt += 1
                break
    return cnt / max(1, hi - lo - 1)        # fraction of window frames present here


def rule_courtband_central(boxes, ball, fw, fh, net_y, ctx):
    """Approach B: gate above-net candidates to the derived FAR COURT BAND, then
    prefer central. The band is derived from the scene: the near player's feet
    (the tallest below-net box) anchor the near baseline; the far band is the
    upper portion of the above-net region bounded so spectators high in the
    stands (cy/H very small) and mid-stand clusters are excluded by REQUIRING the
    candidate's feet to sit in the lower third of the above-net region (closest to
    the net = on the court)."""
    far = [b for b in boxes if above_net(b, net_y)]
    if not far:
        return None
    feet = np.array([b[3] for b in far])
    # the far court band hugs the net from above: feet between the highest
    # above-net feet and the net. Players are LOWER (feet nearer net) than the
    # bulk of stand spectators. Use the feet distribution to set a band that is a
    # FRACTION of the above-net feet span — derived, not pixels.
    top = feet.min()
    span = max(1.0, net_y - top)
    # candidates whose feet are in the lower (court-side) part of the above-net
    # region: feet_y >= top + 0.45*span. This keeps on-court far players, drops
    # high-stand spectators. 0.45 is a geometric fraction of the scene span.
    band_lo = top + 0.45 * span
    incourt = [b for b in far if b[3] >= band_lo]
    pool = incourt if incourt else far
    def s(b):
        cx = (b[0] + b[2]) / 2 / fw
        central = 1.0 - 2.0 * abs(cx - 0.5)
        return central * b[4]
    return max(pool, key=s)


def rule_central_ball_static(boxes, ball, fw, fh, net_y, ctx):
    """Full approach-B blend: central preference + far-side ball proximity bonus
    + static-occupancy penalty. All terms scene-derived. Conf is a soft weight."""
    far = [b for b in boxes if above_net(b, net_y)]
    if not far:
        return None
    far_balls = [c for c in ball if c[1] < net_y]
    diag = (fw ** 2 + fh ** 2) ** 0.5
    def s(b):
        cx = (b[0] + b[2]) / 2 / fw
        cy = (b[1] + b[3]) / 2
        central = 1.0 - 2.0 * abs(cx - 0.5)
        sc = central + 0.5 * float(b[4])
        if far_balls:
            d = min(((cx * fw - c[0]) ** 2 + (cy - c[1]) ** 2) ** 0.5 for c in far_balls)
            sc += max(0.0, 0.5 - d / (0.25 * diag))      # bonus if near far-side ball
        sc -= 0.6 * _far_static_penalty(ctx, b, fw, fh, net_y)
        return sc
    return max(far, key=s)


def rule_production(boxes, ball, fw, fh, net_y, ctx):
    """Mirror the EXACT production far-selection in vision/pose.py so the offline
    cache predicts the live far-CORRECT without a CPU pose run:
      1. cand pre-filter (h>=40, aspect>=1.1, 0.05<xr<0.95)
      2. adaptive band = (min(cand feet)-0.04H, max+0.04H)
      3. far pool = above-net cand
      4. geometric far gate: centrality>=0.5 AND feet in band
      5. pick max centrality (fallback to far pool if none gated)."""
    cand = []
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        w, h = x2 - x1, y2 - y1
        xr = (x1 + x2) / 2 / fw
        if xr < 0.05 or xr > 0.95 or h < 40:
            continue
        if h / max(w, 1.0) < 1.1:
            continue
        cand.append(b)
    if not cand:
        return None
    feet = [b[3] for b in cand]
    band = (min(feet) - 0.04 * fh, max(feet) + 0.04 * fh)
    far = [b for b in cand if (b[1] + b[3]) / 2 < net_y]
    if not far:
        return None
    def central(b):
        return max(0.0, 1.0 - 2.0 * abs((b[0] + b[2]) / 2 / fw - 0.5))
    gated = [b for b in far if central(b) >= 0.5 and band[0] <= b[3] <= band[1]]
    pool = gated if gated else far
    return max(pool, key=central)


if __name__ == "__main__":
    pb, rb, fw, fh, gtbf = load()
    print(f"frame {fw}x{fh}  GT frames {len(gtbf)}  net_y={0.5*fh:.0f}")
    print(f"{'rule':32s} {'far-CORRECT':>12s} {'far-chosen':>11s} {'mean-dist':>10s}")
    for name, rule in [
        ("baseline_biggest_conf", rule_baseline_biggest_conf),
        ("central_conf", rule_central_conf),
        ("courtband_central", rule_courtband_central),
        ("central_ball_static", rule_central_ball_static),
        ("PRODUCTION (mirror of pose.py)", rule_production),
    ]:
        c, d, md, n = score_rule(rule, pb, rb, fw, fh, gtbf)
        print(f"{name:32s} {c:11.1f}% {d:10.1f}% {md:9.0f}px")
