"""
felix END-TO-END bounce regression (fast, no GPU / no video decode).

Replays the FULL production event chain — candidate generators (robust bounce +
turning points + sharp turns + near-court knees) + hit candidates from the
locked poses + classify_events arbitration — on the frozen felix live inputs
(WASB track + locked P1/P2 poses, `scripts/build_felix_e2e_fixture.py`), and
scores the final BOUNCE events against the human 16-bounce ground truth.

Why this exists: the older felix guards pin the DETECTOR layer only
(vision/bounce.py on frozen detections). The 2026-07-02 accuracy session showed
the user-visible quality lives in the ARBITRATION output, and that the felix
camera angle (elevated three-quarter side view) is the pipeline's second
geometry — none of the demo3 (broadcast) event guards exercise it. This test
locks the measured end-to-end level: live F1 0.865, recall 16/16 (was 0.83),
on the exact replay verified to reproduce the live run's markers.

Floors sit just under the measured numbers (noise margin: the 16-bounce GT is
small, so ±1 TP swings F1 by ~0.03). Hits have no felix GT yet — the test only
caps the hit-marker count as a flood guard (annotating ~10 felix strikes is
the known next step to score them properly).

Run:  venv/bin/python tests/test_felix_e2e_regression.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.bounce import detect_bounces_robust  # noqa: E402
from vision.shots import detect_hits  # noqa: E402
from vision.events import (  # noqa: E402
    classify_events, detect_turning_points, detect_sharp_turns,
    detect_near_court_knees)
from tools.bounce_eval.eval_bounces import match_bounces  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "methodo" / "felix_e2e_inputs.json"
GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"

# Measured on the frozen live inputs (== the live run's markers, verified):
# TP 16/16, FP 5, F1 0.865. Floors leave one-TP margin on the small GT.
BOUNCE_F1_FLOOR = 0.82
BOUNCE_RECALL_FLOOR = 15 / 16          # at most one GT bounce may be lost
MAX_PRED_BOUNCES = 26                  # flood guard (measured 21)
MAX_PRED_HITS = 30                     # flood guard only — no felix hit GT yet


def evaluate():
    d = json.loads(FIXTURE.read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    is_real = d["ball_is_real"]
    ppf = d["methodo_ppf"]

    # candidate pool — derived with the CURRENT code, exactly as prod
    # (run_pipeline_8s.py methodo path: bounce_frames deferred = [], locked
    # poses feed both detect_hits and the proximity feature, far_aware on).
    b_cands, _re, _nd = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = set(detect_turning_points(ac, fps, fh))
    turns |= set(detect_sharp_turns(raw, is_real, fps, fw, fh))
    turns |= set(detect_near_court_knees(raw, is_real, fps, fw, fh))
    hits = detect_hits(ac, ppf, fps, fh, fw, bounce_frames=[], far_aware=True)

    events, _report = classify_events(
        b_cands, hits, raw, is_real, ppf, fps, fw, fh,
        turning_frames=sorted(turns), smoothed_centers=ac,
        wasb_centers=None, wasb_is_real=None)

    pred_b = sorted(e["frame"] for e in events if e["label"] == "BOUNCE")
    pred_h = sorted(e["frame"] for e in events if e["label"] == "HIT")

    gt = [int(b["frame"]) for b in json.loads(GT.read_text())["bounces"]]
    tol = round(0.15 * fps)
    tp, fp, fn = match_bounces(pred_b, gt, tol)
    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    prec = n_tp / max(1, n_tp + n_fp)
    rec = n_tp / max(1, n_tp + n_fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "pred_b": pred_b, "pred_h": pred_h,
        "tp": n_tp, "fp": n_fp, "fn": n_fn, "fn_frames": fn,
        "precision": prec, "recall": rec, "f1": f1,
    }


def test_bounce_end_to_end_floors():
    r = evaluate()
    assert r["recall"] >= BOUNCE_RECALL_FLOOR, (
        f"felix e2e bounce recall {r['tp']}/{r['tp'] + r['fn']} < floor "
        f"{BOUNCE_RECALL_FLOOR:.3f} — lost GT bounces at {r['fn_frames']}")
    assert r["f1"] >= BOUNCE_F1_FLOOR, (
        f"felix e2e bounce F1 {r['f1']:.3f} < floor {BOUNCE_F1_FLOOR}")


def test_marker_flood_guards():
    r = evaluate()
    assert len(r["pred_b"]) <= MAX_PRED_BOUNCES, (
        f"{len(r['pred_b'])} bounce markers > {MAX_PRED_BOUNCES} — flooding")
    assert len(r["pred_h"]) <= MAX_PRED_HITS, (
        f"{len(r['pred_h'])} hit markers > {MAX_PRED_HITS} — flooding "
        f"(no felix hit GT yet; this is a sanity cap, not an accuracy floor)")


if __name__ == "__main__":
    r = evaluate()
    print(f"felix e2e: {len(r['pred_b'])} bounces / {len(r['pred_h'])} hits | "
          f"TP {r['tp']} FP {r['fp']} FN {r['fn']} | "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    ok = (r["recall"] >= BOUNCE_RECALL_FLOOR and r["f1"] >= BOUNCE_F1_FLOOR
          and len(r["pred_b"]) <= MAX_PRED_BOUNCES
          and len(r["pred_h"]) <= MAX_PRED_HITS)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
