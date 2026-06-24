"""
Measure real side-swaps in a match video — ground truth for the match-mode
side-swap detector (feat/match-mode / vision/match_tracker.py).

A "side-swap" (change of ends) is when BOTH players cross the net divider and
SETTLE on the opposite side for a sustained period — NOT an ephemeral net
approach/poach. Tennis rules: players change ends on odd games.

Approach (geometry, no model retraining):
  1. Run the standard pose pass (estimate_players_pose + TwoPlayerTracker) to
     get per-frame player centroids and the dynamic net divider.
  2. For each frame, classify the two locked players as above/below the divider
     (far/near). Track the (side of P1) over time.
  3. A side-swap candidate = the side assignment of P1 flips AND stays flipped
     for >= SETTLE_FRAMES (filters poaches). Emit each swap with its frame +
     direction. Persist a few sample frames around it.

Output: tests/fixtures/side_swaps/<stem>.swaps.json
  {
    "video": "...", "fps": ..., "frame_range": [s, e],
    "net_divider_seed": <y or null>,
    "swaps": [{"frame": int, "p1_side_before": "far"|"near", "p1_side_after": ...}],
    "n_swaps": int,
    "notes": "..."
  }

Run AFTER the main pipeline (reuses the same pose path; do NOT run concurrently
with a pipeline run on the same video — both are GPU/CPU heavy).

Usage:
  python tools/measure_side_swaps.py tennis.mp4 -s 0 -d 235 --device mps
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from tqdm import tqdm

from models.yolo_detector import TennisYOLODetector
from utils.video_utils import VideoReader
from vision.pose import estimate_players_pose, _net_y
from vision.player_track import TwoPlayerTracker

# A swap must PERSIST this long (in seconds) to count as a real change of ends,
# not a net poach. Tennis change-of-ends lasts the whole next game (minutes);
# a poach lasts < ~1 s.
SETTLE_SEC = 3.0


def _player_y(pose):
    """Vertical centroid of a pose dict (feet/hips), or None."""
    if pose is None or not pose.get("kpts"):
        return None
    kpts = pose["kpts"]
    # kpts: array Nx3 (x,y,conf). Use the lower-body joints if present, else mean.
    ys = [k[1] for k in kpts if k is not None and len(k) >= 2 and k[2] > 0.3]
    return float(np.mean(ys)) if ys else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-s", "--start", type=float, default=0.0, help="start seconds")
    ap.add_argument("-d", "--duration", type=float, default=None, help="duration seconds (default: all)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pose-model", default="yolov8m-pose.pt")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--settle-sec", type=float, default=SETTLE_SEC)
    args = ap.parse_args()

    rdr = VideoReader(args.video)
    fps = rdr.fps
    total = rdr.frame_count
    start_frame = int(args.start * fps)
    end_frame = total if args.duration is None else min(total, int((args.start + args.duration) * fps))
    max_frames = end_frame - start_frame
    fw, fh = rdr.width, rdr.height
    logger.info(f"{args.video}: {fw}x{fh} {fps:.2f}fps, frames {start_frame}-{end_frame} ({max_frames})")

    detector = TennisYOLODetector(device=args.device)
    from ultralytics import YOLO
    pose_model = YOLO(args.pose_model)

    if start_frame > 0:
        rdr.seek(start_frame)

    # crude net seed: no court zones here, let the tracker derive the divider.
    net_y_seed = None

    tracker = TwoPlayerTracker(fw, fh, net_y_seed, fps)
    # per-frame raw players (for centroid) + locked tracks
    raw_per_frame = [None] * max_frames
    for i, frame in enumerate(tqdm(rdr.iter_frames(), total=max_frames, desc="pose")):
        if i >= max_frames:
            break
        players = estimate_players_pose(
            detector.person_model, pose_model, frame, args.device,
            court_zones=None)
        raw_per_frame[i] = players
        tracker.add_frame(players)
    rdr.release()

    player_tracks = tracker.finalize()  # {1: [...], 2: [...]}

    # Build a per-frame side-of-P1 signal using the locked P1 track y vs divider.
    # Use the tracker's divider if exposed; else derive the running net y from
    # the two locked centroids each frame.
    p1_side = []  # "far" | "near" | None
    for i in range(max_frames):
        p1 = player_tracks[1][i] if i < len(player_tracks[1]) else None
        p2 = player_tracks[2][i] if i < len(player_tracks[2]) else None
        y1 = _player_y(p1)
        y2 = _player_y(p2)
        if y1 is None or y2 is None:
            p1_side.append(None)
            continue
        divider = (y1 + y2) / 2.0
        p1_side.append("far" if y1 < divider else "near")

    # Detect persistent flips of P1's side.
    settle_frames = int(args.settle_sec * fps)
    swaps = []
    i = 0
    # find the first non-None side to anchor
    while i < len(p1_side) and p1_side[i] is None:
        i += 1
    if i < len(p1_side):
        cur = p1_side[i]
        while i < len(p1_side):
            if p1_side[i] is not None and p1_side[i] != cur:
                # candidate flip at i — does it persist?
                j = i
                while j < len(p1_side) and (p1_side[j] is None or p1_side[j] == p1_side[i]):
                    j += 1
                if (j - i) >= settle_frames:
                    swaps.append({
                        "frame": int(start_frame + i),
                        "p1_side_before": cur,
                        "p1_side_after": p1_side[i],
                    })
                    cur = p1_side[i]
                    i = j
                    continue
                i = j
                continue
            i += 1

    result = {
        "video": args.video,
        "fps": float(fps),
        "frame_range": [start_frame, end_frame],
        "settle_sec": args.settle_sec,
        "swaps": swaps,
        "n_swaps": len(swaps),
        "notes": (
            "Ground-truth side-swaps measured by tracking P1's side of the net "
            "divider (running midpoint of the two locked player centroids) and "
            "counting persistent flips (>= settle_sec). Ephemeral net poaches "
            "are filtered. Use to validate vision/match_tracker.py detection."
        ),
    }

    out = args.output or str(Path("tests/fixtures/side_swaps") / (Path(args.video).stem + ".swaps.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"n_swaps={len(swaps)} -> {out}")
    for sw in swaps:
        t = sw["frame"] / fps
        logger.info(f"  swap @ f{sw['frame']} ({t:.1f}s): {sw['p1_side_before']} -> {sw['p1_side_after']}")


if __name__ == "__main__":
    main()
