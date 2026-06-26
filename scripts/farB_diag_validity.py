"""Diagnose WHY the far player doesn't reach the output despite correct selection.
Hypothesis: the tiny far crop yields <10 confident keypoints → is_valid_player
rejects it → it lands in `fallback`, not `valid`, so the centrality far-pick never
sees it. Run the REAL pose path on a few GT frames and report, for the box nearest
the GT far point: nkp, is_valid_player verdict, and whether it entered the pool.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO
from vision.pose import (pose_on_crop, is_valid_player, _centrality, _pool_by_side,
                         _net_y)

GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    frames = list(rdr.iter_frames())
    rdr.release()
    det = TennisYOLODetector(device="cpu")
    pm = YOLO("yolov8m-pose.pt")

    sample = sorted(gtbf)[::9]   # ~25 frames spread across the clip
    net_y = 0.5 * fh
    n = 0; box_ok = 0; nkp_ge10 = 0; nkp_ge6 = 0; valid_n = 0
    nkps = []
    for fidx in sample:
        if fidx not in gtbf:
            continue
        gx, gy = gtbf[fidx]
        frame = frames[fidx]
        dres = det.person_model(frame, conf=0.12, imgsz=1280, classes=[0],
                                device="cpu", verbose=False)
        pboxes = dres[0].boxes.xyxy.cpu().numpy()
        pconfs = dres[0].boxes.conf.cpu().numpy()
        # find the detected box nearest the GT far point
        if len(pboxes) == 0:
            print(f"f{fidx}: NO person boxes at all"); continue
        cxs = (pboxes[:, 0] + pboxes[:, 2]) / 2
        cys = (pboxes[:, 1] + pboxes[:, 3]) / 2
        d = np.hypot(cxs - gx, cys - gy)
        bi = int(np.argmin(d))
        b = pboxes[bi]
        in_cand = (b[2]-b[0]) and (b[3]-b[1] >= 40) and \
                  (0.05 < (b[0]+b[2])/2/fw < 0.95) and \
                  ((b[3]-b[1])/max(1, b[2]-b[0]) >= 1.1)
        # pose the true-far box
        res = pose_on_crop(pm, frame, b, "cpu", imgsz=960, conf=0.15)
        if res:
            kxy, kcf = res
            nkp = int((kcf > 0.30).sum())
        else:
            nkp = 0; kcf = None
        band = None
        feet = [pboxes[i][3] for i in range(len(pboxes))
                if (pboxes[i][3]-pboxes[i][1] >= 40)]
        if feet:
            band = (min(feet)-0.04*fh, max(feet)+0.04*fh)
        valid = is_valid_player(b, kcf, fw, fh, None, band=band)
        n += 1
        if d[bi] < 0.10 * fw:
            box_ok += 1
        if nkp >= 10:
            nkp_ge10 += 1
        if nkp >= 6:
            nkp_ge6 += 1
        if valid:
            valid_n += 1
        nkps.append(nkp)
        print(f"f{fidx}: true-far nearest box d={d[bi]:.0f}px conf={pconfs[bi]:.2f} "
              f"h={b[3]-b[1]:.0f} central={_centrality(b, fw):.2f} | "
              f"box_ok={d[bi]<0.10*fw} nkp={nkp} is_valid={valid}")
    print(f"\n=== SUMMARY over {n} sampled GT frames ===")
    print(f"true-far BOX localized (<192px): {box_ok}/{n} = {100*box_ok/max(1,n):.0f}%")
    print(f"  of which nkp>=10 (passes gate): {nkp_ge10}/{n} = {100*nkp_ge10/max(1,n):.0f}%")
    print(f"  of which nkp>=6:                {nkp_ge6}/{n} = {100*nkp_ge6/max(1,n):.0f}%")
    print(f"  is_valid_player True:           {valid_n}/{n} = {100*valid_n/max(1,n):.0f}%")
    print(f"  nkp values: {sorted(nkps)}")


if __name__ == "__main__":
    main()
