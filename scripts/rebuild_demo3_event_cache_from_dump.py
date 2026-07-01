"""Rebuild the demo3 EVENT cache from a LIVE methodo-inputs dump.

Why rebuild: the previous demo3_event.json froze a ball track + locked poses
from a much older build (pre farselect / pre ball-density / decoded from the
re-encoded clip). Its poses put real racket contacts 2+ box-heights away from
the ball (physically impossible at a contact), so any rule that trusts the
proximity measurement is punished by the fixture while improving live — the
cache lied (again; see docs/PM-HANDOFF.md §4). This script re-freezes the cache
from the EXACT inputs a live canonical run serialized
(COURTSIDE_DUMP_METHODO_INPUTS on `tennis.mp4 -s 73 -d 13`), so the guardrail
tests replay today's production reality:

  kalman_centers / kalman_is_real  <- raw_ball_centers / ball_is_real (pre-spline)
  players_per_frame                <- methodo_ppf (LOCKED, gap-filled P1/P2 —
                                      what prod feeds detect_hits AND
                                      classify_events since the locked-hits fix)
  wasb_centers / wasb_is_real      <- carried over from the previous cache
                                      (same 650-frame clip; consistency-report
                                      only, never used for classification)

Run:  venv/bin/python scripts/rebuild_demo3_event_cache_from_dump.py <methodo_dump.json>
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
OUT_POSE = ROOT / "tests" / "fixtures" / "cache" / "demo3_pose.json"


def main(dump_path):
    d = json.loads(Path(dump_path).read_text())
    old = json.loads(OUT.read_text())
    n = len(d["raw_ball_centers"])
    assert n == len(old["kalman_centers"]), (
        f"frame count mismatch: dump {n} vs old cache {len(old['kalman_centers'])} "
        "— not the same clip?")
    cache = {
        "video": "tennis.mp4 -s 73 -d 13 (live canonical run)",
        "fps": d["fps"], "width": d["width"], "height": d["height"],
        "kalman_centers": d["raw_ball_centers"],
        "kalman_is_real": d["ball_is_real"],
        "wasb_centers": old["wasb_centers"],
        "wasb_is_real": old["wasb_is_real"],
        "players_per_frame": d["methodo_ppf"],
    }
    OUT.write_text(json.dumps(cache))
    print(f"rebuilt {OUT} from {dump_path}: {n} frames, "
          f"ball {sum(1 for c in d['raw_ball_centers'] if c)}/{n}, "
          f"pose>=2 {sum(1 for s in d['methodo_ppf'] if len(s) >= 2)}/{n}")

    # The shot-detector cache (test_shots_regression) re-frozen from the same
    # dump: the spline ball track + the SAME locked poses prod now feeds
    # detect_hits. bounce_frames=[] is prod-exact for the methodo path (bset is
    # deferred there — see run_pipeline_8s.py).
    pose_cache = {
        "video": "tennis.mp4 -s 73 -d 13 (live canonical run)",
        "fps": d["fps"], "width": d["width"], "height": d["height"],
        "ball_centers_smoothed": d["all_ball_centers"],
        "players_per_frame": d["methodo_ppf"],
        "bounce_frames": [],
    }
    OUT_POSE.write_text(json.dumps(pose_cache))
    print(f"rebuilt {OUT_POSE} (spline ball + locked poses, bounce_frames=[])")


if __name__ == "__main__":
    main(sys.argv[1])
