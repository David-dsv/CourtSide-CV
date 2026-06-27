"""
Build a BALL ground-truth template for the demo3 clip, prefilled with the
pipeline's CURRENT ball track so the human only CORRECTS where it's wrong/missing.

Why: bounce/hit recall is plateaued by ball-tracking quality (esp. cold-start at
the clip START — 0 bounces detected before 3.5s while GT has 3). To measure and
fix the ball track honestly we need a per-frame BALL ground truth, the same way
the pose GT unblocked the far-player work.

Coverage strategy (cold-start priority): DENSE over the first ~4s (every frame),
then a regular sample (every Nth) for the rest — enough to measure tracking
coverage + accuracy without annotating all 650 frames.

Output: tests/fixtures/ball_gt/tennis_demo3.ball_gt.json
{
  "video": "tennis.mp4", "clip": "tennis_demo3.mp4",
  "fps": 50.0, "frame_wh": [1920,1080], "frame_offset_source": 3650,
  "frames": [
    { "frame": 0,                       # demo_frame (0-based in the clip)
      "ball": [x, y] | null,            # pipeline's current detection (to verify/fix)
      "visible": true,                  # set false if the ball is occluded/off-frame
      "verified": false } , ...
  ]
}

Usage:
  python tools/ball_gt/make_ball_gt_template.py                 # dense<4s + every 3rd
  python tools/ball_gt/make_ball_gt_template.py --dense-sec 4 --step 3
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
OUT = ROOT / "tests" / "fixtures" / "ball_gt" / "tennis_demo3.ball_gt.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense-sec", type=float, default=4.0,
                    help="annotate EVERY frame in the first N seconds (cold-start priority)")
    ap.add_argument("--step", type=int, default=3, help="after dense window, annotate every Nth")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from utils.video_utils import VideoReader
    from models.yolo_detector import TennisYOLODetector

    if not CLIP.exists():
        logger.error(f"{CLIP} missing — regenerate: run_pipeline_8s.py tennis.mp4 -s 73 -d 13 -o {CLIP}")
        sys.exit(1)

    rdr = VideoReader(str(CLIP))
    fw, fh, fps = rdr.width, rdr.height, rdr.fps
    total = rdr.frame_count
    detector = TennisYOLODetector(device=args.device)
    dense_n = int(args.dense_sec * fps)

    # which frames to annotate: dense window + sample
    want = set(range(min(dense_n, total)))
    want |= set(range(0, total, args.step))
    want = sorted(want)

    # prefill the pipeline's current ball detection on each wanted frame
    ball_by_frame = {}
    for i, frame in enumerate(rdr.iter_frames()):
        if i >= total:
            break
        if i not in want:
            continue
        det = detector.detect(frame, classes=[32])  # 32 = sports_ball
        mask = det["class_ids"] == 32
        if mask.any():
            boxes = det["boxes"][mask]
            scores = det["scores"][mask]
            bi = int(scores.argmax())
            x = float((boxes[bi][0] + boxes[bi][2]) / 2.0)
            y = float((boxes[bi][1] + boxes[bi][3]) / 2.0)
            ball_by_frame[i] = [round(x, 1), round(y, 1)]
        else:
            ball_by_frame[i] = None
    rdr.release()

    frames_out = [{"frame": i, "ball": ball_by_frame.get(i), "visible": True,
                   "verified": False} for i in want]
    doc = {
        "video": "tennis.mp4", "clip": "tennis_demo3.mp4",
        "fps": float(fps), "frame_wh": [fw, fh], "frame_offset_source": 3650,
        "dense_sec": args.dense_sec, "step": args.step,
        "_note": ("Per-frame BALL ground truth, prefilled with the pipeline's current "
                  "ball DETECTION (YOLO26 class 32). Click the real ball to correct; set "
                  "visible:false if occluded/off-frame. Dense over the first dense_sec "
                  "(cold-start priority), sampled after. Used to measure ball-track "
                  "coverage + accuracy honestly (tools/ball_gt/measure_ball_coverage.py)."),
        "frames": frames_out,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)
    n_hit = sum(1 for f in frames_out if f["ball"] is not None)
    logger.info(f"Wrote {OUT} — {len(frames_out)} frames to annotate "
                f"(pipeline detected a ball on {n_hit}/{len(frames_out)} = "
                f"{100*n_hit/max(1,len(frames_out)):.1f}%). Now CORRECT + set verified.")


if __name__ == "__main__":
    main()
