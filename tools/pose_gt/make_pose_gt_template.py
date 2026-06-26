"""
Build a POSE GROUND-TRUTH template for the demo3 benchmark clip, so the human
only has to CORRECT/FILL where the pipeline is wrong or missing — the fastest
path to a real far-player-pose GT (the current SOTA blocker: the pipeline locks
the far player only ~23% of frames, but our accuracy cache assumed ~78%).

What it does
------------
Runs the pipeline's real pose path (estimate_players_pose + TwoPlayerTracker)
on data/output/tennis_demo3.mp4 and dumps, per ANNOTATED frame, the pipeline's
current P1(far)/P2(near) boxes + 17 COCO keypoints. The human then opens the
JSON and, for each frame the pipeline got wrong/missing, fixes the box (and
optionally the wrists, the only kpts the firewall needs) and sets
"verified": true. Frames left "verified": false are treated as pipeline-guesses
(not GT) by the scorer.

Output schema (tests/fixtures/pose_gt/tennis_demo3.pose_gt.json)
---------------------------------------------------------------
{
  "video": "tennis.mp4", "clip": "tennis_demo3.mp4",
  "fps": 50.0, "frame_wh": [1920,1080], "frame_offset_source": 3650,
  "coco_keypoints": ["nose","left_eye",...,"right_ankle"],  # 17, standard order
  "frames": [
    { "frame": 0,                         # demo_frame (0-based in the clip)
      "p1_far":  {"box":[x1,y1,x2,y2]|null, "wrists":[[lx,ly],[rx,ry]]|null,
                  "present": true|false, "verified": false},
      "p2_near": { ... } },
    ...
  ]
}

`present`=false means the player is genuinely not in frame (rare); the scorer
counts a far-slot HIT only when present && a box is given. Annotate a SAMPLE —
you do NOT need all 650 frames: the event frames (bounces/hits ±a few) plus a
uniform sample are enough to measure far-slot coverage honestly.

Usage
-----
  python tools/pose_gt/make_pose_gt_template.py            # all frames
  python tools/pose_gt/make_pose_gt_template.py --step 5   # every 5th frame
  python tools/pose_gt/make_pose_gt_template.py --events-only   # only near GT events
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
GT_EVENTS = ROOT / "data" / "output" / "tennis_demo3_annotations.json"
OUT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"

COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
L_WRIST, R_WRIST = 9, 10


def _wrists(kps_xy, kps_conf):
    if kps_xy is None:
        return None
    out = []
    for idx in (L_WRIST, R_WRIST):
        x, y = float(kps_xy[idx][0]), float(kps_xy[idx][1])
        out.append([round(x, 1), round(y, 1)])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=1, help="annotate every Nth frame")
    ap.add_argument("--events-only", action="store_true",
                    help="only frames at/near GT bounces+hits (±3f)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pose-model", default="yolov8m-pose.pt")
    args = ap.parse_args()

    from utils.video_utils import VideoReader
    from vision.pose import estimate_players_pose
    from vision.player_track import TwoPlayerTracker
    from models.yolo_detector import TennisYOLODetector
    from ultralytics import YOLO

    if not CLIP.exists():
        logger.error(f"{CLIP} missing — regenerate: run_pipeline_8s.py tennis.mp4 -s 73 -d 13 "
                     f"-o {CLIP}")
        sys.exit(1)

    ev = json.load(open(GT_EVENTS))
    event_frames = set()
    for b in ev.get("bounces_true", []):
        event_frames.update(range(b["demo_frame"] - 3, b["demo_frame"] + 4))
    for s in ev.get("strikes", []):
        event_frames.update(range(s["demo_frame"] - 3, s["demo_frame"] + 4))

    rdr = VideoReader(str(CLIP))
    fw, fh, fps = rdr.width, rdr.height, rdr.fps
    total = rdr.frame_count
    detector = TennisYOLODetector(device=args.device)
    pose_model = YOLO(args.pose_model)
    tracker = TwoPlayerTracker(fw, fh, None, fps)

    tracks = []  # per-frame {1: pose|None, 2: pose|None}
    for i, frame in enumerate(rdr.iter_frames()):
        if i >= total:
            break
        players = estimate_players_pose(detector.person_model, pose_model, frame,
                                        args.device, court_zones=None)
        tracker.add_frame(players)
    locked = tracker.finalize()  # {1:[pose|None]*N, 2:[pose|None]*N}
    rdr.release()

    frames_out = []
    for i in range(total):
        if args.events_only and i not in event_frames:
            continue
        if (not args.events_only) and (i % args.step != 0) and (i not in event_frames):
            continue
        row = {"frame": i}
        for slot, key in ((1, "p1_far"), (2, "p2_near")):
            p = locked[slot][i] if i < len(locked[slot]) else None
            if p is None:
                row[key] = {"box": None, "wrists": None, "present": True, "verified": False}
            else:
                box = p.get("box")
                row[key] = {
                    "box": [round(float(v), 1) for v in box] if box is not None else None,
                    "wrists": _wrists(p.get("kps_xy"), p.get("kps_conf")),
                    "present": True, "verified": False,
                }
        frames_out.append(row)

    doc = {
        "video": "tennis.mp4", "clip": "tennis_demo3.mp4",
        "fps": float(fps), "frame_wh": [fw, fh], "frame_offset_source": 3650,
        "coco_keypoints": COCO,
        "_note": ("Pipeline-prefilled POSE template. For each frame the pipeline got "
                  "the FAR player (p1_far) wrong or missing, fix box+wrists and set "
                  "verified:true. present:false = player truly not in frame. Annotate a "
                  "SAMPLE (events + uniform) — not all frames. Far-slot coverage = "
                  "verified&present&box / total verified&present."),
        "frames": frames_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)
    n_far = sum(1 for f in frames_out if f["p1_far"]["box"] is not None)
    logger.info(f"Wrote {OUT} — {len(frames_out)} frames "
                f"(pipeline far-box present on {n_far}/{len(frames_out)} = "
                f"{100*n_far/max(1,len(frames_out)):.1f}%). Now CORRECT + set verified:true.")


if __name__ == "__main__":
    main()
