"""Build a SMALL committed cache that lets tests/test_far_coverage.py exercise the
FAR-player SELECTION logic with NO GPU / NO video decode (fast, deterministic).

For each demo3 pose-GT frame we run the detector + pose ONCE here, and record every
FAR-side (centre above net) candidate box with the only features the selector needs:
center, height, detector conf, and whether it poses to a valid skeleton (nkp>=10).
The test then replays the live ``estimate_players_pose`` far-selection scoring
(hcv: h/hmax + conf + centrality + valid) over these cached candidates and asserts
far-CORRECT stays above a locked floor — isolating SELECTION (what the far-select
change governs) from detection/pose (slow, model-dependent).

Run (slow, once, to regenerate the fixture):
  python tools/pose_gt/build_far_select_cache.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from utils.video_utils import VideoReader  # noqa: E402
from vision.pose import pose_on_crop  # noqa: E402
from models.yolo_detector import TennisYOLODetector  # noqa: E402
from ultralytics import YOLO  # noqa: E402

GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
OUT = ROOT / "tests" / "fixtures" / "pose_gt" / "demo3_far_select_cache.json"

DET_CONF = 0.08       # same far-detection floor the prod selector uses
DET_IMGSZ = 1280
POSE_IMGSZ = 960


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    frames = list(rdr.iter_frames())
    rdr.release()

    out = {"width": fw, "height": fh, "net_y": net_y, "frames": []}
    for n, (fidx, center) in enumerate(sorted(gtbf.items())):
        if fidx >= len(frames):
            continue
        frame = frames[fidx]
        dres = det.person_model(frame, conf=DET_CONF, imgsz=DET_IMGSZ, classes=[0],
                                device="cpu", verbose=False)
        cands = []
        if dres[0].boxes is not None and len(dres[0].boxes) > 0:
            pboxes = dres[0].boxes.xyxy.cpu().numpy()
            pconfs = (dres[0].boxes.conf.cpu().numpy()
                      if dres[0].boxes.conf is not None else np.ones(len(pboxes)))
            for b, c in zip(pboxes, pconfs):
                x1, y1, x2, y2 = [float(v) for v in b]
                cy = (y1 + y2) / 2
                if cy >= net_y:                    # FAR = centre above net
                    continue
                h = y2 - y1
                w = x2 - x1
                aspect = h / max(w, 1.0)
                if h < 0.012 * fh:                 # same noise floor as prod
                    continue
                # pose once to learn the 'valid' flag the selector weights
                res = pose_on_crop(pm, frame, b, "cpu", imgsz=POSE_IMGSZ, conf=0.15)
                nkp = int((res[1] > 0.30).sum()) if (res and res[1] is not None) else 0
                cands.append({"box": [x1, y1, x2, y2], "conf": float(c),
                              "aspect": float(aspect), "nkp": int(nkp)})
        out["frames"].append({"frame": int(fidx), "gt": [float(center[0]),
                                                         float(center[1])],
                              "candidates": cands})
        if (n + 1) % 50 == 0:
            print(f"  ...{n + 1}/{len(gtbf)}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"))
    nframes = len(out["frames"])
    ncands = sum(len(f["candidates"]) for f in out["frames"])
    print(f"wrote {OUT}: {nframes} frames, {ncands} far candidates "
          f"({ncands / max(1, nframes):.1f}/frame)")


if __name__ == "__main__":
    main()
