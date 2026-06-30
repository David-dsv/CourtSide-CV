"""Unit + regression test for detect_near_court_knees (S7 — near-court grazing
bounce candidate generator). Fast, no GPU / no video — replays the committed
canonical post-S1 live methodo dump (the faithful instrument: it byte-matches the
live _stats.json, see docs/research/event-rootcause-2026-06-28-CR.md).

The wall this fixes (measured, feat/s7-near-court-bounce): the canonical pool was
14/17 — GT bounces 174 (38deg) and 308 (55deg) had NO candidate. They are
near-court GRAZING bounces: the ball descends fast, the floor barely redirects it
so its image-y keeps INCREASING (no y-extremum -> detect_turning_points blind) and
vy never inverts (no flip -> detect_bounces_robust / direction blind); the
velocity turn angle is too soft (< the 70deg sharp-turn floor) and the near-court
angle series is too flat for find_peaks to surface. Their separating signature is
the vy KNEE: vy strongly positive (descending) then settling ~0 WITHOUT flipping.

This test asserts both targets are recovered AND is DISCRIMINATING: it proves the
pre-fix pool (turning_points + sharp_turns only) does NOT contain 174/308, so a
revert would fail it. It also bounds the false-candidate budget (no flood).

Run:  venv/bin/python tests/test_near_court_knee.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.events import (  # noqa: E402
    detect_near_court_knees, detect_turning_points, detect_sharp_turns,
    classify_events)
from vision.bounce import detect_bounces_robust  # noqa: E402
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

DUMP = ROOT / "tests" / "fixtures" / "methodo" / "demo3_methodo_inputs_fullvideo_postS1.json"
GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]
TARGETS = [174, 308]   # the near-court NO-CANDIDATE bounces S7 recovers
WIN = 8


def _load():
    d = json.loads(DUMP.read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    is_real = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    hits = d["hits"]
    ppf = d["methodo_ppf"]
    return d, fps, fw, fh, raw, is_real, ac, hits, ppf


def test_recovers_near_court_grazing_bounces_174_308():
    _d, fps, fw, fh, raw, is_real, _ac, _h, _p = _load()
    cands = detect_near_court_knees(raw, is_real, fps, fw, fh)
    for gt in TARGETS:
        assert any(abs(c - gt) <= WIN for c in cands), (
            f"detect_near_court_knees did not recover near-court grazing bounce {gt} "
            f"within +-{WIN}f (got {cands})")


def test_discriminating_pre_fix_pool_misses_targets():
    """MEANINGFULNESS GUARD: the pre-fix generators (turning_points + sharp_turns) do
    NOT cover 174/308, so the knee is the ONLY generator that supplies them — i.e.
    the recovery this suite asserts is load-bearing, not a coincidence already
    covered upstream. (The actual revert-CATCHING tests are test_recovers_* and
    test_end_to_end_*: with the knee removed they go red because 174/308 fall out of
    the pool. This guard makes those two MEANINGFUL by proving nothing else fills
    the gap.)"""
    _d, fps, fw, fh, raw, is_real, ac, _h, _p = _load()
    pre_pool = set(detect_turning_points(ac, fps, fh))
    pre_pool |= set(detect_sharp_turns(raw, is_real, fps, fw, fh))
    for gt in TARGETS:
        assert not any(abs(c - gt) <= WIN for c in pre_pool), (
            f"pre-fix pool unexpectedly already covers {gt}; the test is no longer "
            f"discriminating (got near {[c for c in pre_pool if abs(c-gt)<=WIN]})")


def test_no_flood():
    """Every knee candidate must land near a GT event (zero spurious frames was the
    measured result) and the count is bounded — a grazing-knee generator that fired
    mid-flight would corrupt the alternation DP."""
    _d, fps, fw, fh, raw, is_real, _ac, _h, _p = _load()
    cands = detect_near_court_knees(raw, is_real, fps, fw, fh)
    near = [c for c in cands if any(abs(c - e) <= WIN for e in GT_B + GT_H)]
    assert len(cands) <= 12, f"too many knee candidates ({len(cands)}) — flooding"
    assert near == cands, (
        f"knee candidates not near any GT event (spurious): "
        f"{[c for c in cands if c not in near]}")


def test_near_court_gated():
    """The detector must only fire below the net line (near court). Feed a clean
    descending-then-flat arc entirely ABOVE net_y -> zero candidates."""
    fps, fw, fh = 50.0, 1920, 1080
    net_y = fh * 0.47
    n = 60
    raw = []
    for f in range(n):
        # far court (y well above net), descending then flat — a knee shape, but FAR
        y = int(net_y * 0.5 + min(f, 25) * 5)   # stays < net_y
        raw.append((600, y))
    is_real = [True] * n
    cands = detect_near_court_knees(raw, is_real, fps, fw, fh)
    assert cands == [], f"knee fired in the FAR court (above net): {cands}"


def test_teleport_guard_rejects_jitter_knee():
    """A track GLITCH (a single physically-impossible vertical jump) can fake a
    descend-then-settle slope. The teleport guard (the S3 anti-jitter lesson) must
    reject it: real grazing knees have a max single-frame |dy| <~4.5% of the frame
    height; measured glitches are >=17%. Build a clean near-court knee, then inject
    a >8%-fh jump into its window and assert the candidate disappears."""
    fps, fw, fh = 50.0, 1920, 1080
    net_y = fh * 0.47
    n = 40
    # clean near-court knee at f=20: descend ~25px/f then settle ~3px/f, all near-court
    raw = []
    for f in range(n):
        if f <= 20:
            y = int(net_y + 40 + (20 - (20 - f)) * 0)  # placeholder, set below
        raw.append(None)
    # build explicit y: descending fast pre, flat post
    ys = []
    y = int(net_y + 60)
    for f in range(n):
        if f < 20:
            y += 25
        else:
            y += 3
        ys.append(y)
    raw = [(600, ys[f]) for f in range(n)]
    is_real = [True] * n
    base = detect_near_court_knees(raw, is_real, fps, fw, fh)
    assert any(abs(c - 20) <= WIN for c in base), "control knee not detected"
    # inject a teleport spike inside the window of f=20 (one frame jumps >8% fh)
    raw_glitch = list(raw)
    raw_glitch[19] = (600, ys[19] + int(0.15 * fh))   # +162px impossible jump
    cands = detect_near_court_knees(raw_glitch, is_real, fps, fw, fh)
    assert not any(abs(c - 20) <= WIN for c in cands), (
        f"teleport guard failed to reject the jitter-faked knee (got {cands})")
    # ADVERSARIAL (gap-skip hole, found in review): a track reinit is usually
    # FLANKED BY GAPS. With a gap punched into the same window, the guard must STILL
    # reject — it measures the impossible step over the surviving consecutive real
    # pair, NOT gated on an all-real window. (The buggy all-real-gated version let
    # this through.) Keep the teleport step on a consecutive real pair (18->19).
    raw_gap = list(raw_glitch)
    is_real_gap = list(is_real)
    raw_gap[22] = None          # a gap elsewhere in the window
    is_real_gap[22] = False
    cands2 = detect_near_court_knees(raw_gap, is_real_gap, fps, fw, fh)
    assert not any(abs(c - 20) <= WIN for c in cands2), (
        f"teleport guard disengaged when a gap coexists in the window (got {cands2})")


def test_empty_on_degenerate_input():
    assert detect_near_court_knees([None] * 3, [False] * 3, 50.0, 1920, 1080) == []
    assert detect_near_court_knees([(1, 1)] * 4, [True] * 4, 50.0, 1920, 1080) == []


def _replay(with_knee):
    _d, fps, fw, fh, raw, is_real, ac, hits, ppf = _load()
    b_cands, _r, _n = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = set(detect_turning_points(ac, fps, fh))
    turns |= set(detect_sharp_turns(raw, is_real, fps, fw, fh))
    if with_knee:
        turns |= set(detect_near_court_knees(raw, is_real, fps, fw, fh))
    events, _rep = classify_events(
        b_cands, hits, raw, is_real, ppf, fps, fw, fh,
        turning_frames=sorted(turns), smoothed_centers=ac,
        wasb_centers=None, wasb_is_real=None)
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    tol = round(0.15 * fps)
    res = match_events(ev, GT_B, GT_H, tol)
    return res, scores(res)


def test_anti_overfit_plateau():
    """ANTI-OVERFIT: the recovery must NOT be a knife-edge magic number tuned to 3
    frames. Across a WIDE grid of (window k, descent floor, settle ceiling) the
    generator must keep recovering BOTH targets with ZERO spurious frames — proving
    the grazing-knee is a large-margin physical signature, not a coincidence. (The
    full classify_events 0/0-firewall sweep is verified offline; here we assert the
    generator's own stability, which is fast and dependency-light.)"""
    _d, fps, fw, fh, raw, is_real, _ac, _h, _p = _load()
    import itertools
    for wf, dm, sm in itertools.product(
            (0.04, 0.06, 0.08), (0.008, 0.010, 0.012, 0.015), (0.008, 0.010, 0.012)):
        cands = detect_near_court_knees(
            raw, is_real, fps, fw, fh,
            win_frac=wf, desc_min_frac=dm, settle_max_frac=sm)
        for gt in TARGETS:
            assert any(abs(c - gt) <= WIN for c in cands), (
                f"target {gt} lost at win_frac={wf} desc={dm} settle={sm}")
        fp = [c for c in cands if not any(abs(c - e) <= WIN for e in GT_B + GT_H)]
        assert fp == [], f"spurious at win_frac={wf} desc={dm} settle={sm}: {fp}"


def test_end_to_end_bounce_recall_up_confusion_held():
    """With the knee pool, classify_events recovers 174+308 as BOUNCE, bounce F1
    rises vs the pre-fix pool, hit F1 holds, and confusion_H->B stays 0 (firewall
    intact). These are the floors from the S7 prompt."""
    res0, s0 = _replay(with_knee=False)
    res1, s1 = _replay(with_knee=True)
    # targets recovered (no longer bounce misses)
    for gt in TARGETS:
        assert gt not in res1["FN_b"], f"target bounce {gt} still missed end-to-end"
    # bounce F1 improves, hit F1 does not regress
    assert s1["bounce"]["F1"] >= s0["bounce"]["F1"], (
        f"bounce F1 regressed: {s0['bounce']['F1']:.3f} -> {s1['bounce']['F1']:.3f}")
    assert s1["bounce"]["F1"] >= 0.75, f"bounce F1 below floor: {s1['bounce']['F1']:.3f}"
    assert s1["hit"]["F1"] >= 0.625, f"hit F1 below floor: {s1['hit']['F1']:.3f}"
    # firewall: a near-court knee must NEVER turn a hit into a bounce
    assert res1["conf_HtoB"] == 0, f"confusion H->B leaked: {res1['conf_HtoB']}"


if __name__ == "__main__":
    _d, fps, fw, fh, raw, is_real, ac, hits, ppf = _load()
    cands = detect_near_court_knees(raw, is_real, fps, fw, fh)
    print(f"detect_near_court_knees: {len(cands)} candidates {cands}")
    print(f"  recovers 174: {any(abs(c-174)<=WIN for c in cands)}  "
          f"recovers 308: {any(abs(c-308)<=WIN for c in cands)}")
    res0, s0 = _replay(False)
    res1, s1 = _replay(True)
    print(f"  pre-fix : bounce F1 {s0['bounce']['F1']:.3f}, hit F1 {s0['hit']['F1']:.3f}, "
          f"H->B {res0['conf_HtoB']} B->H {res0['conf_BtoH']}, misses {res0['FN_b']}")
    print(f"  with-knee: bounce F1 {s1['bounce']['F1']:.3f}, hit F1 {s1['hit']['F1']:.3f}, "
          f"H->B {res1['conf_HtoB']} B->H {res1['conf_BtoH']}, misses {res1['FN_b']}")
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
            print(f"  PASS {name}")
    print("PASS")
