"""
Cache raw ball detections for fast bounce/tracking experimentation.

Runs the EXPENSIVE step (YOLO ball detection per frame) ONCE over a frame
window and dumps every candidate (cx, cy, score) per frame to a JSON file.
Downstream experiments (tracking, bounce detection) then iterate in seconds
against this cache instead of re-decoding the video + re-running YOLO each time.

This mirrors EXACTLY what run_pipeline_8s.py Pass 1a does (same detector,
same conf threshold, same class filter), so results transfer back to the
pipeline 1:1.

Usage:
    venv/bin/python experiments/cache_detections.py felix.mp4 --frames 1850 \
        --out experiments/cache/felix_dets.json
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402
from loguru import logger  # noqa: E402

from models.yolo_detector import TennisYOLODetector  # noqa: E402
from utils.video_utils import VideoReader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=int, default=0, help="start frame")
    ap.add_argument("--frames", type=int, default=1850, help="num frames to cache")
    ap.add_argument("--conf", type=float, default=0.15, help="ball conf threshold (pipeline uses 0.15)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="experiments/cache/felix_dets.json")
    args = ap.parse_args()

    detector = TennisYOLODetector(
        conf_threshold=args.conf, iou_threshold=0.45, device=args.device, imgsz=1280
    )

    reader = VideoReader(args.video)
    fps = reader.fps
    w, h = reader.width, reader.height
    if args.start > 0:
        reader.seek(args.start)

    raw = []  # per-frame list of [cx, cy, score]
    n = 0
    for frame in tqdm(reader.iter_frames(), total=args.frames, desc="caching dets"):
        if n >= args.frames:
            break
        det = detector.detect(frame, classes=[32])
        cands = []
        mask = det["class_ids"] == 32
        if mask.any():
            boxes = det["boxes"][mask]
            scores = det["scores"][mask]
            for i in range(len(boxes)):
                if scores[i] >= args.conf:
                    bx = boxes[i]
                    cx, cy = int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2)
                    cands.append([cx, cy, float(scores[i])])
        raw.append(cands)
        n += 1
    reader.release()

    out = {
        "video": str(args.video),
        "start": args.start,
        "frames": n,
        "fps": fps,
        "width": w,
        "height": h,
        "conf": args.conf,
        "detections": raw,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    n_with = sum(1 for c in raw if c)
    logger.info(f"Cached {n} frames ({n_with} with >=1 detection) -> {args.out}")


if __name__ == "__main__":
    main()
