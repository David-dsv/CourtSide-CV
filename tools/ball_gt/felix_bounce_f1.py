"""Cross-clip anti-overfit guard: measure felix bounce F1 LIVE at a chosen
(ball_conf, ball_imgsz), using the SAME prod tracking + bounce algorithm as
test_bounce_regression. The committed felix cache is conf=0.15 — this rebuilds
it live so a ball-conf change is actually reflected (the cache test cannot see it).

  python tools/ball_gt/felix_bounce_f1.py --conf 0.05 --imgsz 1280
  python tools/ball_gt/felix_bounce_f1.py --conf 0.15 --imgsz 1280   # ~ committed baseline

Detects nothing about demo3 — it answers only: does the lower ball conf flood
felix with parasites and wreck its bounce F1? (the generalization question).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from vision.bounce import detect_bounces_from_trajectory, smooth_ball_trajectory  # noqa: E402
# reuse the prod-faithful Kalman tracker + matcher from the regression test
# (tests/ is not a package → load the module by file path)
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "_tbr", str(ROOT / "tests" / "test_bounce_regression.py"))
_tbr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tbr)
_track, _match = _tbr._track, _tbr._match

GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"
CACHE = ROOT / "tests" / "fixtures" / "cache" / "felix_dets.json"


def build_raw(conf, imgsz, device, n_frames):
    """Live YOLO26 ball detections on felix (start=0), all candidates >= conf."""
    from utils.video_utils import VideoReader
    from ultralytics import YOLO
    rdr = VideoReader(str(ROOT / "felix.mp4"))
    model = YOLO(str(ROOT / "training" / "yolo26_ball_best.pt"))
    cls = [0] if len(model.names) == 1 else [1]
    raw = []
    for i, frame in enumerate(rdr.iter_frames()):
        if i >= n_frames:
            break
        res = model(frame, conf=conf, iou=0.45, device=device, imgsz=imgsz,
                    classes=cls, verbose=False)
        cands = []
        b = res[0].boxes
        if b is not None and len(b) > 0:
            xy = b.xyxy.cpu().numpy()
            sc = b.conf.cpu().numpy()
            for k in range(len(xy)):
                x1, y1, x2, y2 = (float(v) for v in xy[k])
                cands.append((int((x1 + x2) / 2), int((y1 + y2) / 2), float(sc[k])))
        raw.append(cands)
    rdr.release()
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    fw, fh, fps = cache["width"], cache["height"], cache["fps"]
    n = len(cache["detections"])
    gt = json.loads(GT.read_text())
    gt_frames = [b["frame"] for b in gt["bounces"]]

    print(f"felix: {fw}x{fh} {fps}fps, {n} frames; GT {len(gt_frames)} bounces", flush=True)
    raw = build_raw(args.conf, args.imgsz, args.device, n)
    ncands = sum(len(f) for f in raw)
    print(f"live dets @conf={args.conf} imgsz={args.imgsz}: {ncands} candidates "
          f"({sum(1 for f in raw if f)}/{n} non-empty frames)", flush=True)

    centers = _track(raw, fw, fh, fps)
    smoothed = smooth_ball_trajectory(centers, max_gap=int(fps * 0.4))
    is_real = [c is not None for c in centers]
    bounces = detect_bounces_from_trajectory(
        smoothed, [0.0] * len(smoothed), fps, fh, fw,
        raw_centers=centers, is_real=is_real)
    pred = [b[0] for b in bounces]

    tol = round(0.15 * fps)
    tp, fp, fn = _match(pred, gt_frames, tol)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    detected = sum(1 for c in centers if c is not None)
    print(f"\nfelix bounce F1={f1:.3f} P={p:.3f} R={r:.3f} (TP={tp} FP={fp} FN={fn})",
          flush=True)
    print(f"track coverage: {detected}/{n} = {100*detected/n:.1f}% | "
          f"PROD-path floor (test_bounce_regression) = 0.72", flush=True)


if __name__ == "__main__":
    main()
