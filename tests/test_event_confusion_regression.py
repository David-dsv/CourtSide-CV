"""
Bounce<->shot CONFUSION regression test (fast, no GPU / no video decode).

Runs the unified event methodology (vision.events.classify_events) on the
committed demo3 event cache (Kalman dense pre-spline centers + is_real + locked
poses + sparse WASB 2nd source) and asserts the cross confusion matrix vs the GT
bounce+shot fixtures does not regress:

  - confusion_H->B == 0   (NON-NEGOTIABLE: no real HIT shown as a bounce)
  - bounce F1 >= floor
  - hit F1    >= floor

This is the guardrail for the methodology. It replays a cache, so it's fast and
deterministic (no model inference). The floors are set just below the measured
methodology numbers to lock the gain without flaking.

Run:  venv/bin/python tests/test_event_confusion_regression.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa: E402
from vision.shots import detect_hits  # noqa: E402
from vision.events import (  # noqa: E402
    classify_events, detect_turning_points, detect_sharp_turns,
    detect_near_court_knees)
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"

# Floors locked at the measured methodology numbers with a small noise margin.
# confusion_H->B is a HARD 0 (the user's bug, non-negotiable) and confusion_B->H
# is also held at 0.
#
# 2026-07-01 — cache REBUILT from a live canonical dump (scripts/
# rebuild_demo3_event_cache_from_dump.py): the old freeze predated farselect +
# ball-density and put real contacts 2+ box-heights from the ball, so it
# punished any rule that trusts the proximity measurement while live improved
# (the "cache lies" trap, docs/PM-HANDOFF.md §4). On the rebuilt cache the
# replay equals the measured LIVE run (score_stats.py on runs E/F): bounce F1
# 0.941 (P=1.0), hit F1 0.857 (P=1.0), confusion 0/0, 0 spurious — the
# contact-corroborated hit evidence + redirect exception + k-flexible parity
# (vision/events.py) and the argmin contact-frame fix + locked-pose hits
# (vision/shots.py, run_pipeline_8s.py). Floors raised to lock the gain.
CONF_HtoB_MAX = 0
CONF_BtoH_MAX = 0
BOUNCE_F1_FLOOR = 0.90
HIT_F1_FLOOR = 0.80


def evaluate():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wasb_real = c["wasb_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]

    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    b_cands, _re, _nd = detect_bounces_robust(all_centers, speeds, fps, fh, fw)
    turns = detect_turning_points(all_centers, fps, fh)
    # far-court apex bounces (no y-extremum but a sharp velocity-vector turn),
    # read on the RAW pre-spline track + is_real mask:
    sharp = detect_sharp_turns(kal, kal_real, fps, fw, fh)
    # near-court grazing-bounce knees (S7) — prod derives these too; the test
    # mirrors run_pipeline_8s.py's candidate pool exactly.
    knees = detect_near_court_knees(kal, kal_real, fps, fw, fh)
    turns = sorted(set(turns) | set(sharp) | set(knees))
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=[],
                       far_aware=True)
    events, report = classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=all_centers,
        wasb_centers=wasb, wasb_is_real=wasb_real)
    evs = [{"frame": e["frame"], "label": e["label"]} for e in events]
    tol = round(0.15 * fps)
    res = match_events(evs, gt_b, gt_h, tol)
    return res, scores(res), report


def test_no_hit_as_bounce():
    res, _s, _r = evaluate()
    assert res["conf_HtoB"] <= CONF_HtoB_MAX, (
        f"confusion_H->B = {res['conf_HtoB']} > {CONF_HtoB_MAX}: a real HIT is "
        f"shown as a BOUNCE (the user's bug). spurious={res['spurious']}")
    assert res["conf_BtoH"] <= CONF_BtoH_MAX, (
        f"confusion_B->H = {res['conf_BtoH']} > {CONF_BtoH_MAX}: a real BOUNCE is "
        f"shown as a HIT.")


def test_bounce_hit_f1_floors():
    _res, s, _r = evaluate()
    assert s["bounce"]["F1"] >= BOUNCE_F1_FLOOR, (
        f"bounce F1 {s['bounce']['F1']:.3f} < floor {BOUNCE_F1_FLOOR}")
    assert s["hit"]["F1"] >= HIT_F1_FLOOR, (
        f"hit F1 {s['hit']['F1']:.3f} < floor {HIT_F1_FLOOR}")


if __name__ == "__main__":
    res, s, report = evaluate()
    print(f"demo3 confusion: H->B={res['conf_HtoB']} B->H={res['conf_BtoH']} | "
          f"bounce F1={s['bounce']['F1']:.3f} hit F1={s['hit']['F1']:.3f} | "
          f"WASB consistency {report['consistency']:.0%}")
    ok = (res["conf_HtoB"] <= CONF_HtoB_MAX
          and s["bounce"]["F1"] >= BOUNCE_F1_FLOOR
          and s["hit"]["F1"] >= HIT_F1_FLOOR)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
