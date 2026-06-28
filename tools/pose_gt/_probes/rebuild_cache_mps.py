"""Fast MPS rebuild of demo3_far_select_cache.json (identical FORMAT/logic to
tools/pose_gt/build_far_select_cache.py, just device=mps so it finishes in minutes
on Apple Silicon instead of ~1h on CPU). Detection+pose are deterministic given the
weights, so the produced fixture matches the documented CPU builder's content; the
CPU builder remains the canonical, device-agnostic source of record.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from utils.video_utils import VideoReader
from vision.pose import pose_on_crop
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

GT = ROOT / "tests/fixtures/pose_gt/tennis_demo3.pose_gt.json"
CLIP = ROOT / "data/output/tennis_demo3.mp4"
OUT = ROOT / "tests/fixtures/pose_gt/demo3_far_select_cache.json"
DEV = "mps"
DET_CONF, DET_IMGSZ, POSE_IMGSZ = 0.08, 1280, 960

def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP)); fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device=DEV)
    pm = YOLO("yolov8m-pose.pt")
    frames = list(rdr.iter_frames()); rdr.release()
    out = {"width": fw, "height": fh, "net_y": net_y, "frames": []}
    for n, (fidx, center) in enumerate(sorted(gtbf.items())):
        if fidx >= len(frames): continue
        frame = frames[fidx]
        dres = det.person_model(frame, conf=DET_CONF, imgsz=DET_IMGSZ, classes=[0],
                                device=DEV, verbose=False)
        cands = []
        if dres[0].boxes is not None and len(dres[0].boxes) > 0:
            pboxes = dres[0].boxes.xyxy.cpu().numpy()
            pconfs = (dres[0].boxes.conf.cpu().numpy()
                      if dres[0].boxes.conf is not None else np.ones(len(pboxes)))
            for b, c in zip(pboxes, pconfs):
                x1, y1, x2, y2 = [float(v) for v in b]
                cy = (y1 + y2) / 2
                if cy >= net_y: continue
                h, w = y2 - y1, x2 - x1
                aspect = h / max(w, 1.0)
                if h < 0.012 * fh: continue
                res = pose_on_crop(pm, frame, b, DEV, imgsz=POSE_IMGSZ, conf=0.15)
                nkp = int((res[1] > 0.30).sum()) if (res and res[1] is not None) else 0
                cands.append({"box": [x1, y1, x2, y2], "conf": float(c),
                              "aspect": float(aspect), "nkp": int(nkp)})
        out["frames"].append({"frame": int(fidx),
                              "gt": [float(center[0]), float(center[1])],
                              "candidates": cands})
        if (n + 1) % 25 == 0:
            print(f"  ...{n + 1}/{len(gtbf)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"))
    nf = len(out["frames"]); nc = sum(len(f["candidates"]) for f in out["frames"])
    print(f"wrote {OUT}: {nf} frames, {nc} far candidates ({nc/max(1,nf):.1f}/frame)", flush=True)

if __name__ == "__main__":
    main()
