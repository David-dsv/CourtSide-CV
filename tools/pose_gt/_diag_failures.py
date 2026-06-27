"""THROWAWAY: on ALL GT frames, run the real far-selection and, for the frames
that FAIL (picked far box > 10%W from GT), report the picked box, the true far
candidate's far-score+skeleton, and the picked box's far-score+skeleton, so we see
exactly why the wrong one won. Reuses the real estimate_players_pose pieces but
inlines the far selection to expose internals.

  python tools/pose_gt/_diag_failures.py [det_conf] [skel_bonus]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.video_utils import VideoReader
from vision.pose import (_far_score, _pool_by_side, pose_on_crop, _net_y)
from models.yolo_detector import TennisYOLODetector
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
DET_CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
SKEL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15


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

    correct = checked = 0
    fail_kinds = {"true_not_in_pool": 0, "outscored": 0, "true_not_detected": 0}
    for fidx, (gx, gy) in gtbf.items():
        if fidx >= len(frames):
            continue
        checked += 1
        frame = frames[fidx]
        dres = det.person_model(frame, conf=DET_CONF, imgsz=1280, classes=[0],
                                device="cpu", verbose=False)
        pboxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        pconfs = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        cand = []
        for i, b in enumerate(pboxes):
            x1, y1, x2, y2 = b
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) / 2
            xr = cx / fw
            if xr < 0.05 or xr > 0.95:
                continue
            asp = h / max(w, 1.0)
            if y2 >= net_y:
                if h < 40 or asp < 1.1:
                    continue
                rank = h * float(pconfs[i])
            else:
                if h < 0.012 * fh:
                    continue
                rank = _far_score(xr, asp, float(pconfs[i]),
                                  cy=(y1 + y2) / 2, net_y=net_y)
            cand.append((rank, i))
        cand.sort(reverse=True)
        pool = _pool_by_side(pboxes, cand, net_y, n_per_side=3, min_pool=3)
        far_pool = [pi for pi in pool if pboxes[pi][3] < net_y]
        # is true far even detected?
        det_true = None
        for i, b in enumerate(pboxes):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if b[3] < net_y and ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5 < 0.10 * fw:
                det_true = i
        # score far_pool with skeleton bonus (pose each)
        scored = []
        for pi in far_pool:
            box = pboxes[pi]
            res = pose_on_crop(pm, frame, box, "cpu", imgsz=960, conf=0.15)
            kcf = res[1] if res else None
            nkp = int((kcf > 0.30).sum()) if kcf is not None else 0
            cx = (box[0] + box[2]) / 2
            asp = (box[3] - box[1]) / max(box[2] - box[0], 1.0)
            cyv = (box[1] + box[3]) / 2
            s = (_far_score(cx / fw, asp, float(pconfs[pi]), cy=cyv, net_y=net_y)
                 + SKEL * min(1.0, nkp / 10.0))
            scored.append((s, pi, nkp, cx, cyv))
        scored.sort(reverse=True)
        if not scored:
            continue
        bs, bpi, bnkp, bcx, bcy = scored[0]
        d = ((bcx - gx) ** 2 + (bcy - gy) ** 2) ** 0.5
        if d < 0.10 * fw:
            correct += 1
        else:
            if det_true is None:
                fail_kinds["true_not_detected"] += 1
            elif det_true not in far_pool:
                fail_kinds["true_not_in_pool"] += 1
            else:
                fail_kinds["outscored"] += 1
                # show the outscored case: picked vs true
                tb = pboxes[det_true]
                tcx = (tb[0] + tb[2]) / 2
                tasp = (tb[3] - tb[1]) / max(tb[2] - tb[0], 1.0)
                ts = _far_score(tcx / fw, tasp, float(pconfs[det_true]))
                print(f"  OUTSCORED f{fidx}: picked c=({bcx:.0f},{bcy:.0f}) "
                      f"s={bs:.2f} | TRUE c=({tcx:.0f},{(tb[1]+tb[3])/2:.0f}) "
                      f"s={ts:.2f} (GT x={gx:.0f}) | picked_xr={bcx/fw:.2f} "
                      f"true_xr={tcx/fw:.2f}")

    print(f"det_conf={DET_CONF} skel_bonus={SKEL}")
    print(f"far CORRECT: {correct}/{checked} = {100*correct/max(1,checked):.1f}%")
    print(f"failure breakdown: {fail_kinds}")


if __name__ == "__main__":
    main()
