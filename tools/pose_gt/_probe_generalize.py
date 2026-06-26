"""THROWAWAY: test GENERALIZABLE (non-frame-center) versions of the far-candidate
score, to avoid overfitting "player is at x=0.5" to demo3. Compares:
  A. central-to-FRAME-center (overfit risk)
  B. central-to-MEDIAN-person-x (scene-derived court center)
  C. central-to-NEAR-player-x (the near player x predicts far player x on a court)
  D. aspect-only (pure human-shape, no horizontal prior)
all with the human-aspect + feet-band terms.

  python tools/pose_gt/_probe_generalize.py [conf] [imgsz]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from utils.video_utils import VideoReader
from models.yolo_detector import TennisYOLODetector

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.06
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 1280


def main():
    gt = json.load(open(GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in gt["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    rdr = VideoReader(str(CLIP))
    fw, fh = rdr.width, rdr.height
    net_y = 0.5 * fh
    det = TennisYOLODetector(device="cpu")
    frames = list(rdr.iter_frames())
    rdr.release()

    # cache ALL persons (above and below net) per GT frame
    cache = {}
    for fidx in gtbf:
        if fidx >= len(frames):
            continue
        dres = det.person_model(frames[fidx], conf=CONF, imgsz=IMGSZ,
                                classes=[0], device="cpu", verbose=False)
        boxes = dres[0].boxes.xyxy.cpu().numpy() if dres[0].boxes is not None else []
        confs = dres[0].boxes.conf.cpu().numpy() if dres[0].boxes is not None else []
        rows = []
        for b, c in zip(boxes, confs):
            x1, y1, x2, y2 = b
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            rows.append((cx, cy, w, h, h / max(w, 1.0), float(c), float(y2)))
        cache[fidx] = rows

    def human_band(asp, feet):
        human = 1.0 if asp <= 2.2 else max(0.0, 1.0 - (asp - 2.2) * 0.5)
        fr = feet / fh
        bandness = 1.0 if 0.18 <= fr <= 0.50 else 0.0
        return human, bandness

    def eval_scheme(name, score_fn):
        correct = present = 0
        dists = []
        for fidx, (gx, gy) in gtbf.items():
            rows = cache.get(fidx, [])
            anet = [r for r in rows if r[6] < net_y]   # above-net candidates
            if not anet:
                continue
            present += 1
            best = max(anet, key=lambda r: score_fn(r, rows))
            d = ((best[0] - gx) ** 2 + (best[1] - gy) ** 2) ** 0.5
            dists.append(d)
            if d < 0.10 * fw:
                correct += 1
        print(f"  {name:34s}: {correct}/{present} = {100*correct/max(1,present):.1f}%"
              f"  mean_d={np.mean(dists):.0f}px")

    # A. frame center
    def A(r, rows):
        cx, asp, conf, feet = r[0], r[4], r[5], r[6]
        central = 1.0 - abs(cx / fw - 0.5) * 2.0
        hu, ba = human_band(asp, feet)
        return central + 0.5 * hu + 0.5 * ba + 0.2 * conf

    # B. median person x as court center (scene-derived)
    def B(r, rows):
        cx, asp, conf, feet = r[0], r[4], r[5], r[6]
        med = np.median([rr[0] for rr in rows]) if rows else fw / 2
        central = 1.0 - abs(cx - med) / (fw / 2)
        hu, ba = human_band(asp, feet)
        return central + 0.5 * hu + 0.5 * ba + 0.2 * conf

    # C. near-player x: the largest below-net person predicts far player x
    def C(r, rows):
        cx, asp, conf, feet = r[0], r[4], r[5], r[6]
        below = [rr for rr in rows if rr[6] >= net_y]
        if below:
            near = max(below, key=lambda rr: rr[2] * rr[3])  # largest area
            anchor = near[0]
        else:
            anchor = fw / 2
        central = 1.0 - abs(cx - anchor) / (fw / 2)
        hu, ba = human_band(asp, feet)
        return central + 0.5 * hu + 0.5 * ba + 0.2 * conf

    # D. aspect+band only, no horizontal prior at all
    def D(r, rows):
        asp, conf, feet = r[4], r[5], r[6]
        hu, ba = human_band(asp, feet)
        return hu + ba + 0.3 * conf

    print(f"conf={CONF} imgsz={IMGSZ}  GT frames={len(gtbf)}\n")
    eval_scheme("A frame-center", A)
    eval_scheme("B median-person-x center", B)
    eval_scheme("C near-player-x anchor", C)
    eval_scheme("D aspect+band only (no x prior)", D)


if __name__ == "__main__":
    main()
