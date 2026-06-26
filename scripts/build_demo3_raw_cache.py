"""Build the demo3 RAW ball-candidate cache: every per-frame YOLO26 ball detection
(cx, cy, score) BEFORE any static-FP filter / Kalman tracking, plus the player
boxes per frame. This is what the demo3_event.json cache lacks — it only stores
the already-tracked centers, so a player-aware FILTER (which must run on the raw
candidates, before tracking) and a re-tracking experiment can't be tested from it.

Mirrors run_pipeline_8s.py Pass 1a exactly: detector.detect(frame, classes=[32])
at ball_conf_thresh=0.15. The player boxes come from the generic person model
(the same boxes the pose-pass / parasite filter would see). Frame index = the
clip's 0-based index = GT demo_frame (the clip IS the segment).

Run once:  venv/bin/python scripts/build_demo3_raw_cache.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
OUT = ROOT / "tests" / "fixtures" / "cache" / "demo3_raw_dets.json"
# Match build_demo3_event_cache.py EXACTLY (the frozen benchmark fixture the
# tests + run_demo3 consume): TennisYOLODetector(device) -> default conf 0.2,
# default imgsz 640, candidate threshold >= 0.2. This makes _track(raw) reproduce
# the committed kalman_centers, so any filter/re-track is measured against the
# SAME baseline the tests use. (Production run_pipeline_8s.py uses conf 0.15 /
# imgsz 1280 — a separate, denser config; we mirror the fixture here on purpose.)
BALL_CONF = 0.2
IMGSZ = 640


def main():
    import numpy as np  # noqa
    from utils.video_utils import VideoReader
    from models.yolo_detector import TennisYOLODetector

    device = "mps"
    try:
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    reader = VideoReader(str(CLIP))
    fps, fw, fh = reader.fps, reader.width, reader.height
    frames = list(reader.iter_frames())
    reader.release()
    print(f"loaded {len(frames)} frames @ {fps}fps {fw}x{fh} (device={device})")

    det = TennisYOLODetector(device=device, conf_threshold=BALL_CONF, imgsz=IMGSZ)

    raw_ball = []   # per-frame list of [cx, cy, score]
    person_boxes = []  # per-frame list of [x1,y1,x2,y2,score]
    for i, fr in enumerate(frames):
        d = det.detect(fr, classes=[0, 32])
        cids = d["class_ids"]
        boxes = d["boxes"]
        scores = d["scores"]
        bcands = []
        pboxes = []
        for j in range(len(boxes)):
            bx = boxes[j]
            if cids[j] == 32 and scores[j] >= BALL_CONF:
                cx, cy = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
                bcands.append([cx, cy, float(scores[j])])
            elif cids[j] == 0:
                pboxes.append([float(bx[0]), float(bx[1]), float(bx[2]),
                               float(bx[3]), float(scores[j])])
        raw_ball.append(bcands)
        person_boxes.append(pboxes)
        if i % 50 == 0:
            print(f"  det {i}/{len(frames)}", flush=True)

    out = {
        "video": str(CLIP),
        "frame_convention": "demo_frame (0-based within clip == GT demo_frame)",
        "fps": fps, "width": fw, "height": fh, "n_frames": len(frames),
        "ball_conf_thresh": BALL_CONF,
        "raw_ball_candidates": raw_ball,     # [[cx,cy,score], ...] per frame
        "person_boxes": person_boxes,        # [[x1,y1,x2,y2,score], ...] per frame
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    nball = sum(len(c) for c in raw_ball)
    print(f"\nSaved -> {OUT}  ({len(frames)} frames, {nball} ball candidates, "
          f"{OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
