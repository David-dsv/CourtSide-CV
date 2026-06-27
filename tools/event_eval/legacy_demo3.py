"""
LEGACY-path iteration harness (Session feat/legacy-bounce-shot-sep).

The DEFAULT prod path (no --event-methodo, default --ball-tracker kalman) is what
the user sees in the annotated video. On the demo3 cache it reproduces EXACTLY the
production legacy event story:

    bounces = detect_bounces_from_trajectory(smoothed_kalman)   # NO vx-flip veto
    hits    = detect_hits(smoothed_kalman, poses, bounce_frames=bounces)
              (bounce_guard inside detect_hits is the only bounce<->hit arbitration)

This script measures that story against the demo3 GT bounce+shot fixtures with the
cross confusion matrix — the BEFORE/AFTER thermometer for the legacy separation
work. It is the fast (no GPU/video) proxy for the LIVE `_stats.json` run
(`python run_pipeline_8s.py tennis.mp4 -s 73 -d 13 --device cpu`): the cache holds
the same Kalman dense pre-spline centers + the locked P1/P2 poses the live pipeline
feeds, so a gain here predicts the live gain (always confirm on the live run).

Run:  venv/bin/python tools/event_eval/legacy_demo3.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import (  # noqa: E402
    smooth_ball_trajectory, detect_bounces_from_trajectory)
from vision.shots import detect_hits  # noqa: E402
from tools.event_eval.event_eval import match_events, print_report, scores  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def load():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    return fps, fw, fh, kal, kal_real, ppf, gt_b, gt_h


def legacy_events(fps, fw, fh, kal, kal_real, ppf):
    """EXACT prod legacy story on the Kalman path (default tracker, no methodo)."""
    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    b_cands = detect_bounces_from_trajectory(
        all_centers, speeds, fps, fh, fw, raw_centers=kal, is_real=kal_real)
    bounce_frames = [b[0] for b in b_cands]
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=bounce_frames,
                       wrist_prox_max=1.2)
    evs = [{"frame": int(f), "label": "BOUNCE"} for (f, x, y) in b_cands]
    evs += [{"frame": int(h["frame"]), "label": "HIT"} for h in hits]
    return evs, b_cands, hits


def main(verbose=True):
    fps, fw, fh, kal, kal_real, ppf, gt_b, gt_h = load()
    tol = round(0.15 * fps)
    evs, b_cands, hits = legacy_events(fps, fw, fh, kal, kal_real, ppf)
    res = match_events(evs, gt_b, gt_h, tol)
    if verbose:
        kdet = sum(1 for c in kal if c is not None)
        print(f"demo3 cache: {len(kal)} frames @ {fps}fps {fw}x{fh}; "
              f"Kalman {kdet}/{len(kal)} ({100*kdet/len(kal):.0f}%)\n")
        print_report(res, fps=fps, tol=tol,
                     title="LEGACY (detect_bounces_from_trajectory / detect_hits)")
        s = scores(res)
        print(f"\n  >>> bounce F1 {s['bounce']['F1']:.3f}  hit F1 {s['hit']['F1']:.3f}")
        print(f"  >>> n preds: {len([e for e in evs if e['label']=='BOUNCE'])} bounce "
              f"/ {len([e for e in evs if e['label']=='HIT'])} hit")
    return res, scores(res)


if __name__ == "__main__":
    main()
