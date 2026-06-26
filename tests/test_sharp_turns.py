"""Unit + regression test for detect_sharp_turns (far-court apex-bounce candidate
generator). Fast, no GPU / no video — replays the committed demo3 event cache.

The wall this fixes (measured, feat/ball-tracking-density): the candidate POOL on
demo3 was 15/17 — GT bounces 120 and 369 were in NO candidate because they are
far-court APEX bounces whose image-y barely dips (~1-6px), so detect_turning_points
(a y-extremum detector) and detect_bounces_robust produce nothing there. Their
VELOCITY VECTOR still turns sharply (120: 154deg, 369: 80deg), which
detect_sharp_turns catches. This test asserts both are recovered AND the
false-candidate budget stays small (it must not flood the pool).

Run:  venv/bin/python tests/test_sharp_turns.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from vision.events import detect_sharp_turns  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"

WIN = 8  # ±frames = the eval tolerance


def _load():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    return fps, fw, fh, kal, kal_real, gt_b, gt_h


def test_recovers_apex_bounces_120_369():
    fps, fw, fh, kal, kal_real, _b, _h = _load()
    cands = detect_sharp_turns(kal, kal_real, fps, fw, fh)
    for gt in (120, 369):
        assert any(abs(c - gt) <= WIN for c in cands), (
            f"detect_sharp_turns did not recover the far-court apex bounce {gt} "
            f"within ±{WIN}f (got {cands})")


def test_false_candidate_budget_bounded():
    """It must add signal, not noise: most candidates land near a GT event and the
    count is bounded (no flooding the pool / firewall budget)."""
    fps, fw, fh, kal, kal_real, gt_b, gt_h = _load()
    cands = detect_sharp_turns(kal, kal_real, fps, fw, fh)
    near = [c for c in cands if any(abs(c - e) <= WIN for e in gt_b + gt_h)]
    # majority of candidates near a true event, and total bounded (was 18 measured).
    assert len(cands) <= 24, f"too many sharp-turn candidates ({len(cands)}) — flooding"
    assert len(near) >= len(cands) * 0.7, (
        f"only {len(near)}/{len(cands)} candidates near a GT event — too noisy")


def test_empty_on_degenerate_input():
    """Pure-function safety: no crash on short / all-None input."""
    assert detect_sharp_turns([None] * 3, [False] * 3, 50.0, 1920, 1080) == []
    assert detect_sharp_turns([(1, 1)] * 4, [True] * 4, 50.0, 1920, 1080) == []


def _pool_recall():
    """End-to-end: with sharp-turns the candidate pool covers >=16/17 GT events."""
    from vision.bounce import smooth_ball_trajectory, detect_bounces_robust
    from vision.shots import detect_hits
    from vision.events import detect_turning_points
    fps, fw, fh, kal, kal_real, gt_b, gt_h = _load()
    c = json.loads(CACHE.read_text())
    ppf = c["players_per_frame"]
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    b, _r, _n = detect_bounces_robust(sm, [0.0] * len(sm), fps, fh, fw)
    pool = set(detect_turning_points(sm, fps, fh))
    pool |= set(detect_sharp_turns(kal, kal_real, fps, fw, fh))
    pool |= set(int(x[0]) for x in b)
    pool |= set(int(h["frame"]) for h in detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[]))
    hit = sum(1 for e in gt_b + gt_h if any(abs(e - p) <= WIN for p in pool))
    return hit, len(gt_b) + len(gt_h)


def test_pool_recall_improved():
    hit, total = _pool_recall()
    assert hit >= 16, f"candidate-pool recall {hit}/{total} < 16 (apex bounces lost)"


if __name__ == "__main__":
    fps, fw, fh, kal, kal_real, gt_b, gt_h = _load()
    cands = detect_sharp_turns(kal, kal_real, fps, fw, fh)
    near = [c for c in cands if any(abs(c - e) <= WIN for e in gt_b + gt_h)]
    hit, total = _pool_recall()
    print(f"detect_sharp_turns: {len(cands)} candidates, {len(near)} near a GT event")
    print(f"  recovers 120: {any(abs(c-120)<=WIN for c in cands)}  "
          f"recovers 369: {any(abs(c-369)<=WIN for c in cands)}")
    print(f"  candidate-pool recall (turns+sharp+robust+hits): {hit}/{total}")
    ok = (any(abs(c - 120) <= WIN for c in cands)
          and any(abs(c - 369) <= WIN for c in cands)
          and hit >= 16 and len(cands) <= 24)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
