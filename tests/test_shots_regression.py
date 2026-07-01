"""
Shot-detection regression test (fast, no GPU / no video decode).

Runs the PRODUCTION shot detector (vision.shots) on a committed cache: the locked
P1/P2 pose + smoothed ball trajectory for the demo3 clip (tennis.mp4 -s 73 -d 13),
plus the hand-annotated strike ground truth. Asserts the detector recalls the GT
strikes (wrist-speed peak signal) without the --demo-override sidecar, and that
the FH/BH classifier doesn't regress to "always wrong" (unknown is acceptable,
flat-out wrong is not).

Rebuild the cache after a TwoPlayerTracker / pose change:
    venv/bin/python scripts/build_demo3_pose_cache.py

Run:  venv/bin/python -m pytest tests/test_shots_regression.py -q
or:   venv/bin/python tests/test_shots_regression.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.shots import detect_hits, classify_forehand_backhand

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_pose.json"
GT = ROOT / "data" / "output" / "tennis_demo3_annotations.json"

# Floor below which the test fails.
# 2026-07-01 — cache rebuilt from a live canonical dump (locked poses, spline
# ball, bounce_frames=[] = prod-exact for the methodo path) and detect_hits'
# contact frame fixed to the ball↔wrist argmin (it was the last frame of the
# search window, a systematic +0.2s bias). Measured on the rebuilt cache:
# 5/8 strikes matched by the DETECTOR alone (the far player's two remaining
# swings sit under the frame-scaled amplitude floor — see far-swing recall in
# docs/research/accuracy-overhaul-CR.md); the events arbitration recovers one
# more via turn+at-wrist (live hit F1 0.857, P=1.0). 20 candidates on a 13s
# rally is the measured post-fix count: the detector is now deliberately
# high-recall (no bounce_guard in the methodo path) and vision/events.py owns
# the precision (contact-corroborated evidence + firewall) — phantom candidates
# no longer reach the overlay, so the anti-noise cap guards the CANDIDATE
# generator, not the final markers.
RECALL_FLOOR = 5      # of 8 GT strikes matched within ±0.30s (detector alone)
FHB_FLOOR = 3         # of the matched non-unknown strikes, correctly classified
MAX_HITS = 24         # anti-noise: candidate count on a 13s rally (measured 20)


def _match(pred_frames, truth_frames, tol):
    cand = sorted((abs(p - t), p, t) for p in pred_frames for t in truth_frames
                  if abs(p - t) <= tol)
    up, ut, m = set(), set(), 0
    for _, p, t in cand:
        if p in up or t in ut:
            continue
        up.add(p)
        ut.add(t)
        m += 1
    return m


def evaluate():
    cache = json.loads(CACHE.read_text())
    gt = json.loads(GT.read_text())
    fps = cache["fps"]
    fh, fw = cache["height"], cache["width"]
    bc = [tuple(c) if c else None for c in cache["ball_centers_smoothed"]]
    ppf = cache["players_per_frame"]
    bounce_frames = set(cache.get("bounce_frames", []))

    hits = detect_hits(bc, ppf, fps, fh, fw, bounce_frames=bounce_frames)
    gt_strikes = gt["strikes"]
    tol = round(0.30 * fps)

    matched, fhb_correct, fhb_wrong, fhb_unknown = 0, 0, 0, 0
    used = set()
    for s in gt_strikes:
        near = [h for h in hits
                if abs(h["frame"] - s["demo_frame"]) <= tol and h["frame"] not in used]
        if not near:
            continue
        h = min(near, key=lambda h: abs(h["frame"] - s["demo_frame"]))
        used.add(h["frame"])
        matched += 1
        fhb = classify_forehand_backhand(h, ppf)
        if fhb == "unknown":
            fhb_unknown += 1
        elif fhb == s["type"]:
            fhb_correct += 1
        else:
            fhb_wrong += 1
    n_non_unknown = matched - fhb_unknown
    return {
        "n_hits": len(hits), "matched": matched, "n_gt": len(gt_strikes),
        "fhb_correct": fhb_correct, "fhb_wrong": fhb_wrong, "fhb_unknown": fhb_unknown,
        "fhb_accuracy": (fhb_correct / n_non_unknown) if n_non_unknown else 0.0,
        "tol": tol,
    }


def test_shots_recall():
    r = evaluate()
    assert r["matched"] >= RECALL_FLOOR, (
        f"Shot recall regressed: {r['matched']}/{r['n_gt']} matched < floor {RECALL_FLOOR} "
        f"(hits={r['n_hits']}, tol=±{r['tol']}f)")
    assert r["n_hits"] <= MAX_HITS, (
        f"Shot detector too noisy: {r['n_hits']} hits > {MAX_HITS} on a 13s rally "
        f"(noise floor regressed)")


def test_fh_bh_not_broken():
    """FH/BH geometric classification is hard in 2D single-frame (~65% ceiling).
    We assert it's not regressed to 'mostly wrong': among non-unknown verdicts,
    at least FHB_FLOOR are correct, and 'unknown' is acceptable (honest)."""
    r = evaluate()
    n_non_unknown = r["matched"] - r["fhb_unknown"]
    if n_non_unknown < FHB_FLOOR:
        return  # too few committed verdicts to assert accuracy
    assert r["fhb_correct"] >= FHB_FLOOR, (
        f"FH/BH accuracy regressed: {r['fhb_correct']}/{n_non_unknown} correct "
        f"({r['fhb_wrong']} wrong, {r['fhb_unknown']} unknown)")


if __name__ == "__main__":
    r = evaluate()
    print(f"Shot regression (demo3 cache): matched {r['matched']}/{r['n_gt']} GT "
          f"strikes (tol ±{r['tol']}f); {r['n_hits']} hits detected")
    print(f"  FH/BH: {r['fhb_correct']} correct, {r['fhb_wrong']} wrong, "
          f"{r['fhb_unknown']} unknown "
          f"(accuracy {r['fhb_accuracy']*100:.0f}% on {r['matched']-r['fhb_unknown']} verdicts)")
    ok = r["matched"] >= RECALL_FLOOR and r["n_hits"] <= MAX_HITS
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
