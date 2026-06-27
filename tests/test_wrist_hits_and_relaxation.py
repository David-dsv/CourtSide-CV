"""Unit + regression tests for the two feat/event-methodo-improve levers:

  1. detect_wrist_hits — the box-height-NORMALIZED far-wrist hit generator that
     recovers far hits detect_hits' FRAME-scaled amplitude floor structurally drops
     (the far player is ~10x smaller, so its full swing never crosses the floor).
  2. the y-MAXIMUM proximity relaxation in classify_events that frees a floor bounce
     landing at the receiving player's racket height (demo3 f174).

Both must lift recall WITHOUT ever breaking the score-free firewall (H->B == 0 AND
B->H == 0). Fast: replays the committed demo3 event cache (no GPU / no video).

Run:  venv/bin/python tests/test_wrist_hits_and_relaxation.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa: E402
from vision.shots import detect_hits  # noqa: E402
from vision.events import (  # noqa: E402
    classify_events, detect_turning_points, detect_sharp_turns, detect_wrist_hits)
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"
WIN = 8


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


def _pools(fps, fw, fh, kal, kal_real, ppf):
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    b, _r, _n = detect_bounces_robust(sm, [0.0] * len(sm), fps, fh, fw)
    turns = sorted(set(detect_turning_points(sm, fps, fh))
                   | set(detect_sharp_turns(kal, kal_real, fps, fw, fh)))
    hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[])
    return sm, b, turns, hits


def _run(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, sm, b, turns, hits,
         gt_b, gt_h, **kw):
    ev, _rep = classify_events(
        b, hits, kal, kal_real, ppf, fps, fw, fh, turning_frames=turns,
        smoothed_centers=sm, wasb_centers=wasb, wasb_is_real=wasb_real, **kw)
    res = match_events([{"frame": e["frame"], "label": e["label"]} for e in ev],
                       gt_b, gt_h, round(0.15 * fps))
    return res, scores(res)


def test_far_wrist_generator_recovers_f525():
    """The far-wrist generator must emit a contact candidate that brackets the far
    GT hit f525 (which detect_hits' frame-scaled floor drops)."""
    fps, fw, fh, kal, kal_real, _w, _wr, ppf, _b, _h = _load()
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    fwh = detect_wrist_hits(ppf, sm, fps, fw, fh)         # default = far only
    assert any(abs(h["frame"] - 525) <= WIN for h in fwh), (
        f"far-wrist generator did not bracket f525 (got {[h['frame'] for h in fwh]})")


def test_far_wrist_never_flips_a_gt_bounce():
    """Safety: even if a far-wrist hit fires near a GT bounce, it must never FLIP
    that bounce to HIT — the firewall (a hit candidate sets hit_like, but a real
    floor bounce keeps its vy/y-max bounce evidence and B->H stays 0). We assert the
    end-to-end invariant: no GT bounce is taken by a HIT prediction (conf_BtoH==0),
    rather than the brittle 'no far-wrist cand within 8f of a GT bounce' (a far hit
    legitimately occurs near a far-court bounce; what matters is the LABEL)."""
    data = _load()
    fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h = data
    sm, b, turns, hits = _pools(fps, fw, fh, kal, kal_real, ppf)
    res, _s = _run(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, sm, b, turns,
                   hits, gt_b, gt_h, far_wrist_hits=True)
    assert res["conf_BtoH"] == 0, (
        f"a far-wrist hit flipped a GT bounce to HIT (B->H={res['conf_BtoH']})")


def test_firewall_holds_and_recall_lifted():
    """End-to-end on the committed cache: confusion 0/0 AND both recall targets
    (bounce f174, hit f525) recovered vs the pre-improve baseline."""
    data = _load()
    fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h = data
    sm, b, turns, hits = _pools(fps, fw, fh, kal, kal_real, ppf)
    res, s = _run(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, sm, b, turns,
                  hits, gt_b, gt_h)
    assert res["conf_HtoB"] == 0, f"H->B={res['conf_HtoB']} (the user's bug)"
    assert res["conf_BtoH"] == 0, f"B->H={res['conf_BtoH']}"
    assert 174 not in res["FN_b"], "GT bounce f174 not recovered"
    assert 525 not in res["FN_h"], "GT hit f525 not recovered"
    assert s["bounce"]["F1"] >= 0.88, f"bounce F1 {s['bounce']['F1']:.3f} < 0.88"
    assert s["hit"]["F1"] >= 0.68, f"hit F1 {s['hit']['F1']:.3f} < 0.68"


def test_firewall_plateau_over_near_wrist():
    """The y-max relaxation must not be a knife-edge: confusion stays 0/0 across a
    WIDE near_wrist band (proves the relaxation is structural, not threshold-tuned)."""
    data = _load()
    fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h = data
    sm, b, turns, hits = _pools(fps, fw, fh, kal, kal_real, ppf)
    for nw in (0.40, 0.45, 0.50, 0.55, 0.60):
        res, _s = _run(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, sm, b,
                       turns, hits, gt_b, gt_h, near_wrist=nw)
        assert res["conf_HtoB"] == 0, f"H->B != 0 at near_wrist={nw}"
        assert res["conf_BtoH"] == 0, f"B->H != 0 at near_wrist={nw}"


def test_far_wrist_toggle_never_breaks_firewall():
    """far_wrist_hits ON or OFF, the firewall must hold (the generator only ever
    adds firewall-safe hit candidates)."""
    data = _load()
    fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h = data
    sm, b, turns, hits = _pools(fps, fw, fh, kal, kal_real, ppf)
    for flag in (True, False):
        res, _s = _run(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, sm, b,
                       turns, hits, gt_b, gt_h, far_wrist_hits=flag)
        assert res["conf_HtoB"] == 0 and res["conf_BtoH"] == 0, (
            f"firewall broke with far_wrist_hits={flag}")


def test_degenerate_inputs_safe():
    assert detect_wrist_hits([], [], 50.0, 1920, 1080) == []
    assert detect_wrist_hits([[]] * 3, [None] * 3, 50.0, 1920, 1080) == []


if __name__ == "__main__":
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
            print(f"  PASS {name}")
    print("PASS")
