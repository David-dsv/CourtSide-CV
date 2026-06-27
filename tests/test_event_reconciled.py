"""Regression for the S1xS2 reconciliation (feat/event-methodo-reconciled).

Two sessions improved vision/events.py and COLLIDED on the `hit_like` line:
  S1 (event-methodo-prod):   vx-gate of the vy anchor (vy_bounce = vy and not vx_flip)
                             -> live confusion 0/0, bounce F1 0.667.
  S2 (event-methodo-improve): far-wrist hit generator + y-MAXIMUM proximity
                             relaxation -> cache bounce F1 0.900, hit F1 0.700.
A NAIVE merge of the two leaked confusion_H->B=1 on the demo3 cache at f146 — the
OUTGOING-ARC apex (a y-MINIMUM) of the far hit committed at f134 fragmented into
its own cluster and the alternation DP labeled it a spurious BOUNCE that the
matcher pinned on the real GT HIT@141. Worse, f174's S2 recovery was PARASITIC on
that very spurious f146(B)/f162(H) parity-bridge: killing the leak killed f174.

The reconciliation (in classify_events / _global_alternation_decode):
  - CONTACT SHADOW: a swing-less, non-floor-bounce-shape cluster in the outgoing
    window AFTER a committed far-wrist hit is forced hit_like (kills the f146 leak).
  - GENTLE-APEX BOUNCE ANCHOR: a y-MAXIMUM turn with a GENTLE velocity-turn-angle
    (the ball glides through the apex, ~0 deg) and NO racket evidence is promoted to
    a mandatory BOUNCE anchor, recovering f174 on its OWN floor-bounce physics (the
    same vx_flip_veto / detect_sharp_turns family) — NOT the parasitic bridge. A
    racket HIT sharply reverses the velocity vector (160-179 deg) and can never
    qualify.

This test asserts: confusion 0/0 AND bounce F1 >= 0.90 AND hit F1 >= 0.70 AND the
specific frames f174 (recovered) and f146 (no leak) behave correctly — and that the
0/0 firewall is PLATEAU-stable across the cadence seed (step0_frac), the knob the
adversarial review flagged as the fragile one for the alternative fix.

Fast: committed demo3 cache, no GPU / no video decode.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa: E402
from vision.shots import detect_hits  # noqa: E402
from vision.events import (  # noqa: E402
    classify_events, detect_turning_points, detect_sharp_turns, BOUNCE)
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def _load():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wasb_real = c["wasb_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    return fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h


def _events(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, **kw):
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    b_cands, _r, _n = detect_bounces_robust(sm, [0.0] * len(sm), fps, fh, fw)
    turns = sorted(set(detect_turning_points(sm, fps, fh))
                   | set(detect_sharp_turns(kal, kal_real, fps, fw, fh)))
    hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[])
    events, _rep = classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=sm,
        wasb_centers=wasb, wasb_is_real=wasb_real, **kw)
    return events


def test_reconciled_confusion_and_f1():
    """0/0 firewall + both class F1s clear the reconciled floors + f174 recovered."""
    d = _load()
    events = _events(*d[:8])
    fps = d[0]
    res = match_events([{"frame": e["frame"], "label": e["label"]} for e in events],
                       d[8], d[9], round(0.15 * fps))
    s = scores(res)
    assert res["conf_HtoB"] == 0, f"confusion_H->B={res['conf_HtoB']} (the user's bug)"
    assert res["conf_BtoH"] == 0, f"confusion_B->H={res['conf_BtoH']}"
    assert s["bounce"]["F1"] >= 0.90, f"bounce F1 {s['bounce']['F1']:.3f} < 0.90"
    assert s["hit"]["F1"] >= 0.70, f"hit F1 {s['hit']['F1']:.3f} < 0.70"
    assert 174 not in res["FN_b"], "GT bounce f174 not recovered (the S2 gain)"
    return s


def test_f146_no_leak_f174_is_a_bounce():
    """The exact frames the reconciliation turns on: NO predicted BOUNCE within
    tolerance of GT HIT@141 (the killed f146 leak), and a predicted BOUNCE within
    tolerance of GT bounce@174 (the gentle-apex recovery, around f179)."""
    d = _load()
    fps = d[0]
    events = _events(*d[:8])
    tol = round(0.15 * fps)
    preds = [(e["frame"], e["label"]) for e in events]
    # no BOUNCE leaks onto the GT hit at 141
    leak = [f for (f, lab) in preds if lab == BOUNCE and abs(f - 141) <= tol]
    assert not leak, f"H->B leak: predicted BOUNCE {leak} on GT HIT@141"
    # a real BOUNCE near GT bounce 174
    rec = [f for (f, lab) in preds if lab == BOUNCE and abs(f - 174) <= tol]
    assert rec, "GT bounce@174 has no predicted BOUNCE nearby (gentle-apex lost)"


def test_firewall_plateau_over_step0():
    """The 0/0 firewall holds across a BAND around the production cadence seed
    (default step0_frac=0.55), so the gentle-apex recovery is not a knife-edge.
    Measured 0/0 plateau (far-wrist OFF default) = step0_frac 0.50-0.75; we assert
    0.50-0.70, which brackets the 0.55 default with margin on both sides. (Below
    ~0.50 a too-tight seed cadence merges a real bounce into a hit beat — a B->H
    that predates this work and is unrelated to the gentle-apex anchor.)"""
    d = _load()
    fps = d[0]
    for s0 in (0.50, 0.55, 0.60, 0.65, 0.70):
        events = _events(*d[:8], step0_frac=s0)
        res = match_events([{"frame": e["frame"], "label": e["label"]} for e in events],
                           d[8], d[9], round(0.15 * fps))
        assert res["conf_HtoB"] == 0, f"H->B != 0 at step0_frac={s0}"
        assert res["conf_BtoH"] == 0, f"B->H != 0 at step0_frac={s0}"


def test_gentle_apex_toggle_is_firewall_safe():
    """Turning the far-wrist generator off must never break the firewall (the
    contact shadow + gentle-apex are no-ops without far-wrist candidates, but the
    firewall must hold either way)."""
    d = _load()
    fps = d[0]
    for fwh in (True, False):
        events = _events(*d[:8], far_wrist_hits=fwh)
        res = match_events([{"frame": e["frame"], "label": e["label"]} for e in events],
                           d[8], d[9], round(0.15 * fps))
        assert res["conf_HtoB"] == 0 and res["conf_BtoH"] == 0, (
            f"firewall broke at far_wrist_hits={fwh}: "
            f"H->B={res['conf_HtoB']} B->H={res['conf_BtoH']}")


def test_no_bare_turn_phantom_bounce():
    """Every labeled BOUNCE must carry FLOOR-BOUNCE evidence (vy-flip, a real
    bounce candidate, or a y-MAXIMUM/gentle apex) — never a bare trajectory turn.
    The PM caught the live over-fire: detect_turning_points admits y-MINIMUM turns
    (mid-air arc apexes) for recall, and the parity DP was filling them as phantom
    BOUNCEs (demo3 LIVE f229; on a noisier track they also land on GT hits => H->B).
    The bounce_shape gate in _reward forbids that. Here we assert it on the cache:
    no labeled BOUNCE lands >tol from EVERY GT bounce (no spurious bounce FP)."""
    d = _load()
    fps = d[0]
    gt_b = d[8]
    events = _events(*d[:8])
    tol = round(0.15 * fps)
    phantom = [e["frame"] for e in events if e["label"] == BOUNCE
               and not any(abs(e["frame"] - g) <= tol for g in gt_b)]
    assert not phantom, f"phantom (bare-turn) BOUNCE FPs survived: {phantom}"


if __name__ == "__main__":
    s = test_reconciled_confusion_and_f1()
    test_f146_no_leak_f174_is_a_bounce()
    test_firewall_plateau_over_step0()
    test_gentle_apex_toggle_is_firewall_safe()
    test_no_bare_turn_phantom_bounce()
    print(f"  PASS test_reconciled_confusion_and_f1 "
          f"(bounce F1={s['bounce']['F1']:.3f} hit F1={s['hit']['F1']:.3f}, 0/0)")
    print("  PASS test_no_bare_turn_phantom_bounce")
    print("  PASS test_f146_no_leak_f174_is_a_bounce")
    print("  PASS test_firewall_plateau_over_step0")
    print("  PASS test_gentle_apex_toggle_is_firewall_safe")
    print("PASS")
