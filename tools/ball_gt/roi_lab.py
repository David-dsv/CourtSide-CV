"""ROI-fallback thermometer: sweep dynamic-ROI (window, zoom) on the demo3 ball GT,
mirroring the REAL prod Kalman path (predict -> gate full-frame -> ROI re-detect on
the gap -> re-gate). Honest LIVE measurement: coverage + CORRECT + cold-start.

Two-stage so a zoom/window sweep is fast:
  1) cache full-frame candidates ONCE per (imgsz,conf) at the prod ball config, AND
     cache the per-frame ROI re-detect candidates at every (half,zoom) on the cache grid.
     The ROI cache is keyed by the EXACT crop center the prod Kalman would predict, so
     we must run the Kalman to know the centers -> we do a first Kalman pass on the
     full-frame cache to harvest the gap-frame predictions, THEN run ROI YOLO only on
     those frames (that's exactly what prod does: ROI is a fallback on the gaps).
  2) replay: feed (full-frame + ROI) candidates through the SAME interleaved Kalman and
     score. Stage 2 is instant; iterate window/zoom freely.

  python tools/ball_gt/roi_lab.py --build       # builds caches (slow, YOLO)
  python tools/ball_gt/roi_lab.py --sweep        # replays the grid (instant)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import cv2
from utils.video_utils import VideoReader

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "ball_gt" / "tennis_demo3.ball_gt.json"
CLIP = ROOT / "data" / "output" / "tennis_demo3.mp4"
CACHE = ROOT / "tools" / "ball_gt" / "_roi_lab_cache.json"

# the prod demo3 (clean) full-frame ball config from --ball-auto
FF_IMGSZ, FF_CONF = 1920, 0.05
# the grid of ROI fallback params to sweep (half-window as a FRACTION of W, zoom)
HALF_FRACS = [0.0833, 0.125, 0.1667]   # ~160/240/320 px half on a 1920W clip
ZOOMS = [1.5, 2.0, 2.5]


def load_gt():
    gt = json.load(open(GT))
    return {f["frame"]: f["ball"] for f in gt["frames"]
            if f.get("verified") and f.get("visible", True) and f.get("ball")}


# ───── the prod interleaved Kalman (mirrors run_pipeline_8s.py Pass 1b) ─────
def run_kalman(per_frame_ff, fw, fh, fps, *, roi_fn=None, conf=FF_CONF,
               roi_gate_frac=1.0, roi_min_conf=0.0):
    """per_frame_ff: list per frame of [(cx,cy,score), ...] full-frame candidates.
    roi_fn(frame_idx, px, py) -> [(cx,cy,score),...] or None : called ONLY on a gap
    (no in-gate full-frame match) when a prediction exists. Returns the track +
    the list of gap frame indices where the ROI was invoked + recovered.

    roi_gate_frac: ROI candidates are gated to roi_gate_frac * the (grown) full gate.
        <1 = TIGHTER than full-frame: a fallback recovery must be CLOSE to the
        prediction, else it poisons the Kalman state (off-target ROI hit pulls the
        filter, corrupting neighbour frames). roi_min_conf: ROI candidate must clear
        this conf (the crop upscaling inflates faint parasites)."""
    n = len(per_frame_ff)
    diag = float(np.hypot(fw, fh))

    # static-FP zones (same 8%/20px-grid filter as prod), on full-frame cands only
    from collections import Counter
    pts = []
    for cands in per_frame_ff:
        for cx, cy, sc in cands:
            if sc >= conf:
                pts.append((int(cx) // 20, int(cy) // 20))
    thr = max(10, int(n * 0.08))
    static = {k for k, v in Counter(pts).items() if v >= thr}

    def is_static(cx, cy):
        gx, gy = int(cx) // 20, int(cy) // 20
        return any((gx + dx, gy + dy) in static
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1))

    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 200.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8.0
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
    inited = [False]
    fsu = [0]

    def predict():
        if not inited[0]:
            return None
        p = kf.predict()
        fsu[0] += 1
        return float(p[0][0]), float(p[1][0])

    def init(x, y):
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        inited[0] = True
        fsu[0] = 0

    def gdist(x, y):
        p = kf.statePost
        return float(np.hypot(x - p[0][0], y - p[1][0]))

    base, growth = 150.0, 80.0
    gmax = diag * 0.35
    coast = int(fps * 0.12)
    reinit = diag * 0.50
    last = None
    track = [None] * n
    roi_recovered = []

    for i in range(n):
        cands = [(cx, cy, sc) for cx, cy, sc in per_frame_ff[i]
                 if sc >= conf and not is_static(cx, cy)]
        c = None
        pred = predict()
        gate = min(gmax, base + growth * fsu[0])
        # try full-frame in-gate match first
        matched = False
        if cands:
            if not inited[0]:
                b = max(cands, key=lambda z: z[2])
                init(b[0], b[1])
                c = (b[0], b[1])
                matched = True
            else:
                bd, bdist = None, float("inf")
                for cx, cy, sc in cands:
                    d = gdist(cx, cy)
                    if d < gate and d < bdist:
                        bdist, bd = d, (cx, cy)
                if bd is not None:
                    kf.correct(np.array([[bd[0]], [bd[1]]], np.float32))
                    fsu[0] = 0
                    st = kf.statePost
                    c = (float(st[0][0]), float(st[1][0]))
                    matched = True
        # ── ROI FALLBACK: only when inited, predicting, and no in-gate match ──
        if (not matched) and inited[0] and pred is not None and roi_fn is not None:
            roi_cands = roi_fn(i, pred[0], pred[1])
            if roi_cands:
                rc = [(cx, cy, sc) for cx, cy, sc in roi_cands
                      if sc >= max(conf, roi_min_conf) and not is_static(cx, cy)]
                roi_gate = gate * roi_gate_frac  # tighter gate for the fallback
                bd, bdist = None, float("inf")
                for cx, cy, sc in rc:
                    d = gdist(cx, cy)
                    if d < roi_gate and d < bdist:
                        bdist, bd = d, (cx, cy)
                if bd is not None:
                    kf.correct(np.array([[bd[0]], [bd[1]]], np.float32))
                    fsu[0] = 0
                    st = kf.statePost
                    c = (float(st[0][0]), float(st[1][0]))
                    matched = True
                    roi_recovered.append(i)
        # coast / reinit fallbacks (same as prod)
        if not matched and inited[0]:
            if fsu[0] <= coast and pred:
                c = (pred[0], pred[1])
            elif last is not None and cands:
                nb, nd = None, float("inf")
                for cx, cy, sc in cands:
                    d = np.hypot(cx - last[0], cy - last[1])
                    if d < reinit and d < nd:
                        nd, nb = d, (cx, cy)
                if nb is not None:
                    init(*nb)
                    c = nb
        if c is None and inited[0] and fsu[0] <= coast and pred:
            c = (pred[0], pred[1])
        if c is not None:
            last = c
        track[i] = c
    return track, roi_recovered


def score(track, gtbf, fw, fps, tol_frac=0.02, start_sec=4.0):
    tol = tol_frac * fw
    sn = int(start_sec * fps)
    cov = corr = n = covs = corrs = ns = 0
    for f, g in gtbf.items():
        if f >= len(track):
            continue
        n += 1
        s = f < sn
        ns += s
        t = track[f]
        if t is not None:
            cov += 1
            covs += s
            if np.hypot(t[0] - g[0], t[1] - g[1]) < tol:
                corr += 1
                corrs += s
    return dict(cov=cov, corr=corr, n=n, covs=covs, corrs=corrs, ns=ns)


def fmt(r):
    return (f"cov {r['cov']}/{r['n']}={100*r['cov']/r['n']:.1f}% "
            f"CORR {r['corr']}/{r['n']}={100*r['corr']/r['n']:.1f}% | "
            f"START cov {100*r['covs']/max(1,r['ns']):.1f}% "
            f"CORR {100*r['corrs']/max(1,r['ns']):.1f}%")


def build():
    """Slow YOLO pass: cache full-frame candidates + per-gap ROI candidates for the grid."""
    from models.yolo_detector import TennisYOLODetector
    rdr = VideoReader(str(CLIP))
    fw, fh, fps = rdr.width, rdr.height, rdr.fps
    frames = list(rdr.iter_frames())
    rdr.release()
    det = TennisYOLODetector(device="cpu", imgsz=1280,
                             ball_conf=FF_CONF, ball_imgsz=FF_IMGSZ)
    bm = det.custom_model

    def ff_detect(fr):
        r = bm(fr, conf=FF_CONF, iou=0.45, imgsz=FF_IMGSZ, classes=[0], verbose=False)
        out = []
        if r[0].boxes is not None and len(r[0].boxes):
            b = r[0].boxes.xyxy.cpu().numpy()
            s = r[0].boxes.conf.cpu().numpy()
            for k in range(len(b)):
                out.append([float((b[k][0]+b[k][2])/2), float((b[k][1]+b[k][3])/2), float(s[k])])
        return out

    print(f"Caching full-frame candidates ({FF_IMGSZ}/{FF_CONF}) on {len(frames)} frames…", flush=True)
    per_frame_ff = [ff_detect(fr) for fr in frames]

    # First Kalman pass (no ROI) → harvest the gap-frame predictions to crop around.
    # This mirrors prod: we only know the crop center because the Kalman predicted it.
    _, _ = run_kalman(per_frame_ff, fw, fh, fps)  # warm (unused result)

    # Re-run, recording every (frame_idx, px, py) where the ROI WOULD fire.
    gap_centers = {}

    def record_roi(i, px, py):
        gap_centers[i] = (px, py)
        return None  # don't actually recover during harvest

    run_kalman(per_frame_ff, fw, fh, fps, roi_fn=record_roi)
    print(f"ROI would fire on {len(gap_centers)} gap frames", flush=True)

    # For each gap frame, run ROI YOLO at every (half_frac, zoom) on the grid.
    roi_cache = {}  # f"{half}|{zoom}" -> {frame_idx: [[cx,cy,sc],...]}
    for hf in HALF_FRACS:
        half = int(hf * fw)
        for zoom in ZOOMS:
            key = f"{hf:.4f}|{zoom:.2f}"
            roi_cache[key] = {}
            for i, (px, py) in gap_centers.items():
                fr = frames[i]
                x0, y0 = max(0, int(px - half)), max(0, int(py - half))
                x1, y1 = min(fw, int(px + half)), min(fh, int(py + half))
                if x1 - x0 < 20 or y1 - y0< 20:
                    roi_cache[key][str(i)] = []
                    continue
                crop = fr[y0:y1, x0:x1]
                up = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
                r = bm(up, conf=FF_CONF, iou=0.45, imgsz=1280, classes=[0], verbose=False)
                out = []
                if r[0].boxes is not None and len(r[0].boxes):
                    b = r[0].boxes.xyxy.cpu().numpy()
                    s = r[0].boxes.conf.cpu().numpy()
                    for k in range(len(b)):
                        cx = float((b[k][0]+b[k][2])/2)/zoom + x0
                        cy = float((b[k][1]+b[k][3])/2)/zoom + y0
                        out.append([cx, cy, float(s[k])])
                roi_cache[key][str(i)] = out
            print(f"  cached ROI grid {key}: {len(gap_centers)} gaps", flush=True)

    CACHE.write_text(json.dumps(dict(
        fw=fw, fh=fh, fps=fps, per_frame_ff=per_frame_ff, roi_cache=roi_cache,
        half_fracs=HALF_FRACS, zooms=ZOOMS)))
    print(f"Wrote {CACHE.name}", flush=True)


def sweep():
    c = json.loads(CACHE.read_text())
    fw, fh, fps = c["fw"], c["fh"], c["fps"]
    per_frame_ff = c["per_frame_ff"]
    roi_cache = c["roi_cache"]
    gtbf = load_gt()

    base_track, _ = run_kalman(per_frame_ff, fw, fh, fps)
    base = score(base_track, gtbf, fw, fps)
    print(f"BASELINE (no ROI, {FF_IMGSZ}/{FF_CONF}):  {fmt(base)}", flush=True)
    print("", flush=True)

    for hf in c["half_fracs"]:
        for zoom in c["zooms"]:
            key = f"{hf:.4f}|{zoom:.2f}"
            rc = roi_cache.get(key, {})

            def roi_fn(i, px, py, rc=rc):
                return rc.get(str(i))
            track, recovered = run_kalman(per_frame_ff, fw, fh, fps, roi_fn=roi_fn)
            r = score(track, gtbf, fw, fps)
            half = int(hf * fw)
            d_cov = r["cov"] - base["cov"]
            d_corr = r["corr"] - base["corr"]
            print(f"ROI half={half}px({hf:.3f}W) zoom={zoom}: {fmt(r)}  "
                  f"[Δcov {d_cov:+d} Δcorr {d_corr:+d}, ROI fired+recovered {len(recovered)}]",
                  flush=True)


def guard_sweep():
    """Grid the ROI gate-tightness + min-conf guards for the best window/zoom — the
    real lever: the unguarded ROI lifts coverage but POISONS CORRECT (off-target
    crop hit corrupts the Kalman state). Target: cov AND CORRECT both >= baseline."""
    c = json.loads(CACHE.read_text())
    fw, fh, fps = c["fw"], c["fh"], c["fps"]
    per_frame_ff = c["per_frame_ff"]
    gtbf = load_gt()
    base = score(run_kalman(per_frame_ff, fw, fh, fps)[0], gtbf, fw, fps)
    print(f"BASELINE (no ROI):  {fmt(base)}", flush=True)
    print(f"  target: cov >= {base['cov']}, CORR >= {base['corr']}\n", flush=True)

    configs = [(0.0833, 2.0), (0.0833, 1.5), (0.125, 2.0), (0.1667, 2.0)]
    gate_fracs = [1.0, 0.6, 0.4, 0.25, 0.15]
    min_confs = [0.0, 0.10, 0.15, 0.25]
    for hf, zoom in configs:
        rc = c["roi_cache"][f"{hf:.4f}|{zoom:.2f}"]

        def roi_fn(i, px, py, rc=rc):
            return rc.get(str(i))
        print(f"--- half={int(hf*fw)}px zoom={zoom} ---", flush=True)
        for gf in gate_fracs:
            for mc in min_confs:
                track, recovered = run_kalman(
                    per_frame_ff, fw, fh, fps, roi_fn=roi_fn,
                    roi_gate_frac=gf, roi_min_conf=mc)
                r = score(track, gtbf, fw, fps)
                dcov = r["cov"] - base["cov"]
                dcorr = r["corr"] - base["corr"]
                flag = " ◀ WIN" if (dcov >= 0 and dcorr >= 0 and dcov + dcorr > 0) else \
                       ("" if dcorr >= 0 else " (corr↓)")
                print(f"  gate×{gf:<4} minconf={mc:<4}: "
                      f"cov {r['cov']}={100*r['cov']/r['n']:.1f}% "
                      f"CORR {r['corr']}={100*r['corr']/r['n']:.1f}% "
                      f"[Δcov {dcov:+d} Δcorr {dcorr:+d}, rec {len(recovered)}]{flag}",
                      flush=True)


def diagnose(hf=0.0833, zoom=2.0, gate_frac=1.0, min_conf=0.0):
    """For one config: classify what the ROI does to each GT frame vs baseline —
    recovered-CORRECT, recovered-WRONG (covers but >tol), and POISONED (baseline
    was CORRECT, ROI flips it to WRONG via Kalman-state corruption)."""
    c = json.loads(CACHE.read_text())
    fw, fh, fps = c["fw"], c["fh"], c["fps"]
    per_frame_ff = c["per_frame_ff"]
    rc = c["roi_cache"][f"{hf:.4f}|{zoom:.2f}"]
    gtbf = load_gt()
    tol = 0.02 * fw

    base_track, _ = run_kalman(per_frame_ff, fw, fh, fps)

    def roi_fn(i, px, py):
        return rc.get(str(i))
    roi_track, recovered = run_kalman(per_frame_ff, fw, fh, fps, roi_fn=roi_fn,
                                      roi_gate_frac=gate_frac, roi_min_conf=min_conf)
    recovered = set(recovered)

    def ok(t, g):
        return t is not None and np.hypot(t[0] - g[0], t[1] - g[1]) < tol

    rec_correct = rec_wrong = poisoned = healed = 0
    poison_frames, wrong_frames = [], []
    for f, g in sorted(gtbf.items()):
        if f >= len(base_track):
            continue
        b_ok = ok(base_track[f], g)
        r_ok = ok(roi_track[f], g)
        if f in recovered:
            if r_ok:
                rec_correct += 1
            else:
                rec_wrong += 1
                wrong_frames.append(f)
        else:
            if b_ok and not r_ok:
                poisoned += 1
                poison_frames.append(f)
            elif (not b_ok) and r_ok:
                healed += 1
    print(f"\n=== DIAGNOSE half={int(hf*fw)}px zoom={zoom} ===", flush=True)
    print(f"ROI fired+recovered (in-gate) on {len(recovered)} frames", flush=True)
    print(f"  of GT frames the ROI recovered: CORRECT {rec_correct}, WRONG {rec_wrong}",
          flush=True)
    print(f"  NON-recovered GT frames: POISONED (was-correct→wrong) {poisoned}, "
          f"HEALED (downstream) {healed}", flush=True)
    print(f"  poisoned frames: {poison_frames[:20]}", flush=True)
    print(f"  recovered-but-WRONG frames: {wrong_frames[:20]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--guard-sweep", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--half", type=float, default=0.0833)
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--gate-frac", type=float, default=1.0)
    ap.add_argument("--min-conf", type=float, default=0.0)
    args = ap.parse_args()
    if args.build:
        build()
    if args.guard_sweep:
        guard_sweep()
    elif args.diagnose:
        diagnose(args.half, args.zoom, args.gate_frac, args.min_conf)
    elif args.sweep or not args.build:
        sweep()
