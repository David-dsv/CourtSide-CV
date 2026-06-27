"""Build a RAW ball-detection cache for fast strategy iteration.

Runs the YOLO26 ball model ONCE over a clip at a LOW conf + chosen imgsz and dumps
EVERY candidate box+score per frame. Strategy scripts (conf floor, gating,
selection) then replay this cache instantly instead of re-running YOLO (~7 min/run
on CPU). Mirrors the far-select `_strat_sweep.py` posed-cache pattern.

  python tools/ball_gt/build_det_cache.py --imgsz 1280 --conf 0.03 \
      --out tests/fixtures/ball_gt/demo3_det_cache_1280.json

The cache stores the absolute frame index (0-based within the clip) so it aligns
with the GT frame numbers and the pipeline's per-frame arrays.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.video_utils import VideoReader

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default=str(CLIP))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.03,
                    help="LOW floor so every weak candidate is captured for replay")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames read from the clip start (e.g. felix=1850 to "
                         "match its bounce GT window; demo3 is short so omit).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rdr = VideoReader(args.clip)
    fw, fh, fps = rdr.width, rdr.height, rdr.fps
    frames = []
    for f in rdr.iter_frames():
        if args.max_frames is not None and len(frames) >= args.max_frames:
            break
        frames.append(f)
    rdr.release()
    print(f"Clip {args.clip}: {fw}x{fh} {fps}fps {len(frames)} frames", flush=True)

    from ultralytics import YOLO
    yolo26 = ROOT / "training" / "yolo26_ball_best.pt"
    model = YOLO(str(yolo26))
    single = (len(model.names) == 1)
    cls = [0] if single else [1]
    print(f"Model {yolo26.name}: single-class={single}, ball-cls={cls}", flush=True)

    per_frame = []
    for i, frame in enumerate(frames):
        res = model(frame, conf=args.conf, iou=0.45, device=args.device,
                    imgsz=args.imgsz, classes=cls, verbose=False)
        cands = []
        b = res[0].boxes
        if b is not None and len(b) > 0:
            xy = b.xyxy.cpu().numpy()
            sc = b.conf.cpu().numpy()
            for k in range(len(xy)):
                x1, y1, x2, y2 = (float(v) for v in xy[k])
                cands.append({
                    "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                    "w": x2 - x1, "h": y2 - y1, "score": float(sc[k]),
                })
        per_frame.append(cands)
        if i % 100 == 0:
            print(f"  frame {i}/{len(frames)}  cands={len(cands)}", flush=True)

    out = {
        "clip": Path(args.clip).name, "width": fw, "height": fh, "fps": fps,
        "n_frames": len(frames), "imgsz": args.imgsz, "conf_floor": args.conf,
        "frames": per_frame,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    total = sum(len(f) for f in per_frame)
    nonempty = sum(1 for f in per_frame if f)
    print(f"\nWrote {args.out}: {total} candidates over {len(frames)} frames "
          f"({nonempty} non-empty)", flush=True)


if __name__ == "__main__":
    main()
