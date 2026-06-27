"""
Score a PRODUCTION `_stats.json` (live pipeline output) against the demo3
bounce+shot ground truth, using the SAME cross confusion matrix as run_demo3.py.

This is the LIVE thermometer the event-methodo-prod CR requires: it does NOT
read a frozen cache — it scores whatever the real pipeline actually emitted.

The pipeline writes bounce/shot frames as ABSOLUTE frames (start_frame + f).
The GT is clip-relative 0-based. We convert stats frames back to clip-relative
by subtracting frame_range[0] (== start_frame), so the two align.

Run:
  venv/bin/python tools/event_eval/score_stats.py path/to/<name>_stats.json
  # optional: --truth-bounces / --truth-shots to point at other GT
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.event_eval.event_eval import match_events, print_report, scores  # noqa: E402

GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def stats_to_events(stats):
    """Map a _stats.json into the (frame-relative) event list the matcher wants.

    Each bounce -> {"frame": rel, "label": "BOUNCE"}; each shot -> "HIT".
    Frames are made clip-relative by subtracting frame_range[0] (start_frame).
    """
    off = int(stats.get("frame_range", [0, 0])[0])
    evs = []
    for b in stats.get("bounces", []):
        evs.append({"frame": int(b["frame"]) - off, "label": "BOUNCE"})
    for s in stats.get("shots", []):
        evs.append({"frame": int(s["frame"]) - off, "label": "HIT"})
    return evs, off


def main():
    ap = argparse.ArgumentParser(description="Score a live _stats.json vs demo3 GT.")
    ap.add_argument("stats", help="Path to the <name>_stats.json from a live run.")
    ap.add_argument("--truth-bounces", default=str(GT_B))
    ap.add_argument("--truth-shots", default=str(GT_S))
    ap.add_argument("--tolerance-frames", type=int, default=None)
    args = ap.parse_args()

    stats = json.loads(Path(args.stats).read_text())
    tb = json.loads(Path(args.truth_bounces).read_text())
    ts = json.loads(Path(args.truth_shots).read_text())
    fps = stats.get("fps") or tb.get("fps") or ts.get("fps") or 50.0
    tol = (args.tolerance_frames if args.tolerance_frames is not None
           else round(0.15 * fps))

    evs, off = stats_to_events(stats)
    gt_b = [int(b["frame"]) for b in tb["bounces"]]
    gt_h = [int(s["frame"]) for s in ts["shots"]]

    n_b = sum(1 for e in evs if e["label"] == "BOUNCE")
    n_h = sum(1 for e in evs if e["label"] == "HIT")
    print(f"stats: {Path(args.stats).name}  fps={fps}  start_frame(off)={off}")
    print(f"predicted: {n_b} bounces / {n_h} shots   vs GT {len(gt_b)} bounces / {len(gt_h)} shots\n")

    res = match_events(evs, gt_b, gt_h, tol)
    print_report(res, fps=fps, tol=tol, title=f"PROD LIVE ({Path(args.stats).name})")
    s = scores(res)
    print("\n── KEY METRICS (live prod) ──")
    print(f"  confusion_H->B : {res['conf_HtoB']}   (target 0)")
    print(f"  confusion_B->H : {res['conf_BtoH']}   (target 0)")
    print(f"  bounce F1      : {s['bounce']['F1']:.3f}  "
          f"(R={s['bounce']['R']:.3f}, P={s['bounce']['P']:.3f})")
    print(f"  hit F1         : {s['hit']['F1']:.3f}  "
          f"(R={s['hit']['R']:.3f}, P={s['hit']['P']:.3f})")
    return res


if __name__ == "__main__":
    main()
