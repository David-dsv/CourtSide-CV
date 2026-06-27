"""Score a LIVE prod _stats.json (bounces + shots) against the demo3 GT with the
cross confusion matrix. This is the LIVE thermometer the prompt requires (not a
cache replay).

_stats.json emits ABSOLUTE frames (start_frame + segment_frame). The demo3 GT is
in SEGMENT-frame coordinates (frame_offset 0), so we subtract frame_range[0].

Usage:  python tools/event_eval/score_stats.py /path/to/<name>_stats.json
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.event_eval.event_eval import match_events, print_report, scores

GT_B = ROOT / "tests/fixtures/bounces/tennis_demo3.bounces.json"
GT_S = ROOT / "tests/fixtures/shots/tennis_demo3.shots.json"


def main():
    stats_path = Path(sys.argv[1])
    st = json.loads(stats_path.read_text())
    fps = st.get("fps", 50.0)
    fr0 = st.get("frame_range", [0, 0])[0]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    # _stats.json bounces/shots carry absolute frames; map back to segment frames.
    evs = []
    for b in st.get("bounces", []):
        evs.append({"frame": int(b["frame"]) - fr0, "label": "BOUNCE"})
    for s in st.get("shots", []):
        evs.append({"frame": int(s["frame"]) - fr0, "label": "HIT"})
    tol = round(0.15 * fps)
    res = match_events(evs, gt_b, gt_h, tol)
    print(f"LIVE _stats.json: {stats_path.name}  (frame_range start={fr0}, fps={fps})")
    print(f"  emitted: {len(st.get('bounces',[]))} bounces / {len(st.get('shots',[]))} shots")
    print_report(res, fps=fps, tol=tol, title="LIVE PROD (legacy path)")
    s = scores(res)
    print(f"\n  >>> bounce F1 {s['bounce']['F1']:.3f}  hit F1 {s['hit']['F1']:.3f}  "
          f"confusion H->B {res['conf_HtoB']} / B->H {res['conf_BtoH']}")


if __name__ == "__main__":
    main()
