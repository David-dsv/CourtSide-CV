"""THROWAWAY: JOINT sweep over demo3-GT AND felix-proxy-GT of a single unified far
score, to find a config that wins on BOTH (no overfit to one clip). Per candidate:
  S = wc*central + 0.5*human - top_penalty(ramp,knee=0.25) + 0.2*conf
      + wv*(valid_and_on_court)
where valid_and_on_court = posed-as-skeleton AND cy>=0.25*net_y. Poses candidates
(slow). Caches per (clip,frame) detections+poses once, then sweeps wc,wv cheaply.

  python tools/pose_gt/_joint_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import vision.pose as P
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEMO_GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
DEMO = ROOT / "data" / "output" / "tennis_demo3.mp4"
FELIX = ROOT / "felix.mp4"
FELIX_GT = "/tmp/felix_far_gt.json"


def collect(clip, gtbf, det, pm, sample_all=False):
    """For each GT frame: pose all far candidates, cache (cx,cy,asp,conf,nkp,valid)."""
    r = VideoReader(str(clip))
    fw, fh = r.width, r.height
    ny = 0.5 * fh
    frames = list(r.iter_frames())
    r.release()
    cache = {}
    for fidx in gtbf:
        if fidx >= len(frames):
            continue
        f = frames[fidx]
        dres = det.person_model(f, conf=0.08, imgsz=1280, classes=[0],
                                device="cpu", verbose=False)
        pb = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        pc = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        rows = []
        for b, c in zip(pb, pc):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if cy >= ny:
                continue
            w, h = x2 - x1, y2 - y1
            if h < 0.012 * fh:
                continue
            res = P.pose_on_crop(pm, f, b, "cpu", imgsz=960, conf=0.15)
            kcf = res[1] if res else None
            nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
            valid = P.is_valid_player(b, kcf, fw, fh, None, band=None)
            rows.append((cx, cy, h / max(w, 1.0), float(c), nkp, int(valid), h))
        cache[fidx] = rows
    return cache, fw, fh, ny


def score(row, wc, wv, fw, ny):
    cx, cy, asp, conf, nkp, valid, h = row
    central = 1.0 - abs(cx / fw - 0.5) * 2.0
    human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
    s = wc * central + 0.5 * human + 0.2 * conf
    knee = 0.25 * ny
    if cy < knee:
        s -= 1.0 * (1.0 - cy / knee)
    if valid and cy >= knee and h >= 0.04 * 1080:    # valid + on court
        s += wv
    return s


def evalc(cache, gtbf, fw, ny, wc, wv):
    ok = n = 0
    for fidx, (gx, gy) in gtbf.items():
        rows = cache.get(fidx, [])
        if not rows:
            continue
        n += 1
        best = max(rows, key=lambda r: score(r, wc, wv, fw, ny))
        if ((best[0] - gx) ** 2 + (best[1] - gy) ** 2) ** 0.5 < 0.10 * fw:
            ok += 1
    return ok, n


def main():
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")
    dgt = json.load(open(DEMO_GT))
    dgtbf = {f["frame"]: f["p1_far"]["center"] for f in dgt["frames"]
             if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    fgt = {int(k): v for k, v in json.load(open(FELIX_GT)).items()}
    print("collecting demo3...", flush=True)
    dcache, dfw, dfh, dny = collect(DEMO, dgtbf, det, pm)
    print("collecting felix...", flush=True)
    fcache, ffw, ffh, fny = collect(FELIX, fgt, det, pm)

    print("\n  wc   wv  : demo3%   felix%   min(both)")
    best = None
    for wc in (0.5, 0.7, 1.0):
        for wv in (0.0, 0.2, 0.3, 0.35, 0.4, 0.5):
            do, dn = evalc(dcache, dgtbf, dfw, dny, wc, wv)
            fo, fn = evalc(fcache, fgt, ffw, fny, wc, wv)
            dp, fp = 100 * do / max(1, dn), 100 * fo / max(1, fn)
            m = min(dp, fp)
            flag = ""
            if best is None or m > best[0]:
                best = (m, wc, wv, dp, fp); flag = "  <-- best min"
            print(f"  {wc:.1f}  {wv:.2f} : {dp:5.1f}%  {fp:5.1f}%   {m:5.1f}%{flag}")
    print(f"\nBEST joint: wc={best[1]} wv={best[2]} -> demo3 {best[3]:.1f}% "
          f"felix {best[4]:.1f}% (min {best[0]:.1f}%)")


if __name__ == "__main__":
    main()
