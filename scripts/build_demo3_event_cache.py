"""Build the demo3 EVENT cache: dense WASB pre-spline ball centers + is_real
mask + locked P1/P2 poses, for the bounce-vs-shot methodology benchmark.

Why a NEW cache (vs the existing demo3_pose.json):
  - demo3_pose.json is the Kalman-sparse trajectory (64% non-None post-spline)
    with NO is_real mask. The vx-flip veto / classify methodology must read
    direction on the RAW pre-spline centers WITH an is_real mask (CHANTIER 0 in
    docs/research/bounce-vs-shot-direction.md), so it needs the WASB dense path.
  - This cache mirrors the pipeline's WASB path exactly: HeatmapBallTracker
    .track_ball(frames) gives raw_ball_centers (pre-spline), is_real = [c is not
    None]. We also keep the Kalman trajectory for completeness and the locked
    pose tracks (for the player-proximity feature + the shot detector).

Frame convention: this clip is data/output/tennis_demo3.mp4 (tennis.mp4 -s 73
-d 13, 650 frames @ 50fps). Frame index here is 0-based within the clip = the
GT's `demo_frame` = the segment frame index run_pipeline_8s.py produces. So the
cache, the GT fixtures, and the pipeline all align on demo_frame.

Run once:  venv/bin/python scripts/build_demo3_event_cache.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from utils.video_utils import VideoReader  # noqa: E402
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory  # noqa: E402

# The clip IS the segment: 0-based frame = demo_frame. Decode the pre-cut clip so
# there's zero offset ambiguity (its frame 0 == GT demo_frame 0).
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
WASB_WEIGHTS = ROOT / "training" / "wasb" / "wasb_tennis.pth.tar"
OUT = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"


def build_wasb_centers(clip_path, device, frame_step):
    """Run the production WASB path and return (raw_centers, fps, fw, fh, n)."""
    from models.wasb_ball_tracker import HeatmapBallTracker
    reader = VideoReader(str(clip_path))
    fps, fw, fh = reader.fps, reader.width, reader.height
    frames = list(reader.iter_frames())
    reader.release()
    tracker = HeatmapBallTracker(str(WASB_WEIGHTS), device=device)
    raw = tracker.track_ball(frames, frame_step=frame_step)
    return raw, fps, fw, fh, len(frames), frames


def build_kalman_centers(frames, fw, fh, fps, device):
    """YOLO+Kalman trajectory (the sparse fallback path), for completeness."""
    from models.yolo_detector import TennisYOLODetector
    det = TennisYOLODetector(device=device)
    raw = []
    for fr in frames:
        d = det.detect(fr, classes=[32])
        ball_mask = d["class_ids"] == 32
        cands = []
        if ball_mask.any():
            boxes = d["boxes"][ball_mask]
            scores = d["scores"][ball_mask]
            for j in range(len(boxes)):
                if scores[j] >= 0.2:
                    bx = boxes[j]
                    cx, cy = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
                    cands.append((cx, cy, float(scores[j])))
        raw.append(cands)
    # reuse the regression-test tracker (production Kalman config)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tbr", ROOT / "tests" / "test_bounce_regression.py")
    tbr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbr)
    return det, tbr._track(raw, fw, fh, fps)


def build_poses(det, frames, fw, fh, fps, device, ball_centers):
    """Locked P1/P2 pose tracks (the production Pass 1.5 path)."""
    from ultralytics import YOLO
    from vision.pose import estimate_players_pose
    from vision.player_track import TwoPlayerTracker
    pose_model = YOLO("yolov8m-pose.pt")
    net_y = fh * 0.47
    ptracker = TwoPlayerTracker(fw, fh, net_y, fps)
    n = len(frames)
    for i, fr in enumerate(frames):
        ball_xy = ball_centers[i] if i < len(ball_centers) else None
        players = estimate_players_pose(
            det.person_model, pose_model, fr, device, ball_xy=ball_xy, court_zones=None)
        ptracker.add_frame(players)
        if i % 50 == 0:
            print(f"  pose {i}/{n}", flush=True)
    tracks = ptracker.finalize()

    def _ser(p):
        if p is None:
            return None
        kx = p.get("kps_xy")
        kc = p.get("kps_conf")
        return {
            "box": [float(v) for v in p["box"]],
            "kps_xy": kx.tolist() if kx is not None else None,
            "kps_conf": kc.tolist() if kc is not None else None,
        }
    ppf = []
    for i in range(n):
        far = tracks[1][i] if i < len(tracks[1]) else None
        near = tracks[2][i] if i < len(tracks[2]) else None
        slot = []
        if near is not None:
            slot.append(_ser(near))
        if far is not None:
            slot.append(_ser(far))
        ppf.append(slot)
    return ppf, ptracker.coverage()


def main():
    device = "mps"  # WASB needs a GPU-ish backend; falls to cpu if mps unavailable
    try:
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    # WASB is trained ~30fps; demo3 is 50fps. frame_step=1 keeps the 3-frame window
    # tightest (closest to per-frame motion); we measured recall and kept it.
    frame_step = 1
    print(f"WASB on {CLIP.name} (device={device}, frame_step={frame_step})...")
    wasb_raw, fps, fw, fh, n, frames = build_wasb_centers(CLIP, device, frame_step)
    wasb_det = sum(1 for c in wasb_raw if c is not None)
    print(f"  WASB: {wasb_det}/{n} detected ({100*wasb_det/n:.0f}%)")

    # YOLO + pose run far faster on MPS (Apple Silicon) than CPU; use the same
    # backend WASB used. This only changes BUILD speed, not the cached numbers
    # the regression then reads.
    yolo_device = device

    print("YOLO+Kalman trajectory...", flush=True)
    det, kalman_centers = build_kalman_centers(frames, fw, fh, fps, yolo_device)
    kal_det = sum(1 for c in kalman_centers if c is not None)
    print(f"  Kalman: {kal_det}/{n} detected ({100*kal_det/n:.0f}%)", flush=True)

    print("Locked P1/P2 poses...", flush=True)
    ppf, cov = build_poses(det, frames, fw, fh, fps, yolo_device, wasb_raw)
    print(f"  locked: P1(far) {cov[1]*100:.0f}%  P2(near) {cov[2]*100:.0f}%")

    # serialize centers None-safe
    def _ser_centers(cs):
        return [[float(c[0]), float(c[1])] if c is not None else None for c in cs]

    out = {
        "video": str(CLIP),
        "source_video": "tennis.mp4",
        "source_start_sec": 73.0,
        "frame_convention": "demo_frame (0-based within the clip == GT demo_frame)",
        "fps": fps, "width": fw, "height": fh, "n_frames": n,
        "frame_step_wasb": frame_step,
        # raw pre-spline tracker centers + is_real mask (CHANTIER 0 contract):
        "wasb_centers": _ser_centers(wasb_raw),
        "wasb_is_real": [c is not None for c in wasb_raw],
        "kalman_centers": _ser_centers(kalman_centers),
        "kalman_is_real": [c is not None for c in kalman_centers],
        "players_per_frame": ppf,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nSaved -> {OUT}  ({n} frames, {OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
