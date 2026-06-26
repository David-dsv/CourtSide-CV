"""
Iteration driver — measure BASELINE vs METHODOLOGY on the demo3 benchmark.

Loads the committed demo3 event cache (Kalman dense pre-spline centers + is_real,
the SPARSE WASB track as the independent 2nd source, and locked poses), runs:

  BASELINE  = the production pipeline's CURRENT event story on this clip. WASB is
              blind on demo3 (10%), so the production Kalman path is used:
                bounces = detect_bounces_from_trajectory (NO vx-flip veto — the
                          Kalman path doesn't thread pre-spline+mask, CHANTIER 3)
                hits    = detect_hits
  METHODO   = vision.events.classify_events (unified classify + alternation),
              direction read on the dense KALMAN track, WASB used only for the
              independent consistency metric.

and scores BOTH against the GT bounce+shot fixtures with the cross confusion
matrix. This is the BEFORE/AFTER thermometer the CR requires.

Run:  venv/bin/python tools/event_eval/run_demo3.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import (  # noqa: E402
    smooth_ball_trajectory, detect_bounces_robust, detect_bounces_from_trajectory)
from vision.shots import detect_hits  # noqa: E402
from vision.events import classify_events, detect_turning_points  # noqa: E402
from tools.event_eval.event_eval import match_events, print_report, scores  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def load():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wasb_real = c["wasb_is_real"]
    ppf = c["players_per_frame"]
    tb = json.loads(GT_B.read_text())
    ts = json.loads(GT_S.read_text())
    gt_b = [int(b["frame"]) for b in tb["bounces"]]
    gt_h = [int(s["frame"]) for s in ts["shots"]]
    return fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h


def baseline_events(fps, fw, fh, kal, ppf):
    """Reproduce the production Kalman-path event story (the demo3 reality)."""
    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    b_cands = detect_bounces_from_trajectory(all_centers, speeds, fps, fh, fw)
    bounce_frames = [b[0] for b in b_cands]
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=bounce_frames)
    evs = [{"frame": int(f), "label": "BOUNCE"} for (f, x, y) in b_cands]
    evs += [{"frame": int(h["frame"]), "label": "HIT"} for h in hits]
    return evs, b_cands, hits


def methodo_events(fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, **kw):
    """Unified classifier. Direction read on the dense KALMAN track; WASB only
    feeds the independent consistency metric."""
    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    # candidate generators (RECALL): robust bounce detector + ALL trajectory
    # turning points + the wrist-speed hit detector. The classifier (DECISION)
    # arbitrates the label of each.
    b_cands, _re, _nd = detect_bounces_robust(all_centers, speeds, fps, fh, fw)
    turns = detect_turning_points(all_centers, fps, fh)
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=[])
    events, report = classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=all_centers,
        wasb_centers=wasb, wasb_is_real=wasb_real, **kw)
    evs = [{"frame": e["frame"], "label": e["label"]} for e in events]
    return evs, events, report


def main(verbose=True, **kw):
    fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, gt_b, gt_h = load()
    tol = round(0.15 * fps)
    kdet = sum(1 for c in kal if c is not None)
    wdet = sum(1 for c in wasb if c is not None)
    if verbose:
        print(f"demo3 cache: {len(kal)} frames @ {fps}fps {fw}x{fh}; "
              f"Kalman {kdet}/{len(kal)} ({100*kdet/len(kal):.0f}%, primary) | "
              f"WASB {wdet}/{len(wasb)} ({100*wdet/len(wasb):.0f}%, 2nd source)\n")

    b_evs, b_cands, b_hits = baseline_events(fps, fw, fh, kal, ppf)
    rb = match_events(b_evs, gt_b, gt_h, tol)
    if verbose:
        print_report(rb, fps=fps, tol=tol, title="BASELINE (Kalman curve-fit bounces / detect_hits)")
        print()

    m_evs, m_events, m_report = methodo_events(
        fps, fw, fh, kal, kal_real, wasb, wasb_real, ppf, **kw)
    rm = match_events(m_evs, gt_b, gt_h, tol)
    if verbose:
        print_report(rm, fps=fps, tol=tol, title="METHODOLOGY (classify+alternation+WASB)")
        print(f"  WASB consistency: {m_report['consistency']:.0%} "
              f"({m_report['agree']} agree / {m_report['disagree']} disagree / "
              f"{m_report['unreadable']} unreadable)")
        print()

    sb, sm = scores(rb), scores(rm)
    if verbose:
        print("── SUMMARY (baseline -> methodology) ──")
        print(f"  confusion_H->B : {rb['conf_HtoB']} -> {rm['conf_HtoB']}   (target 0)")
        print(f"  confusion_B->H : {rb['conf_BtoH']} -> {rm['conf_BtoH']}   (target 0)")
        print(f"  bounce F1      : {sb['bounce']['F1']:.3f} -> {sm['bounce']['F1']:.3f}")
        print(f"  hit F1         : {sb['hit']['F1']:.3f} -> {sm['hit']['F1']:.3f}")
        print(f"  WASB consistency: {m_report['consistency']:.0%}")
    return {"baseline": rb, "methodo": rm, "report": m_report,
            "events": m_events, "sb": sb, "sm": sm}


if __name__ == "__main__":
    main()
