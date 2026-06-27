"""Replay a RAW detection cache under different (conf, selection, gating) strategies
and score coverage + CORRECT vs the human ball GT — INSTANTLY (no YOLO re-run).

Build the cache once (tools/ball_gt/build_det_cache.py), then sweep strategies:

  python tools/ball_gt/replay_strat.py --cache tests/fixtures/ball_gt/demo3_det_cache_1280.json \
      --strat highscore --conf 0.20            # reproduce the scorer baseline
  python tools/ball_gt/replay_strat.py --cache ... --strat kalman --conf 0.05
  python tools/ball_gt/replay_strat.py --cache ... --sweep                 # conf grid

CORRECT = tracked point within tol_frac*W of the GT point. Reports overall +
cold-start (first 4s). The selection strategies mirror run_pipeline_8s.py Pass 1.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "ball_gt" / "tennis_demo3.ball_gt.json"


# ───────────────────────── selection strategies ─────────────────────────
def select_highscore(frames, conf, fw, fh):
    """Baseline: per frame, the highest-score candidate above conf (no gating)."""
    track = [None] * len(frames)
    for i, cands in enumerate(frames):
        ok = [c for c in cands if c["score"] >= conf]
        if ok:
            b = max(ok, key=lambda c: c["score"])
            track[i] = (b["cx"], b["cy"])
    return track


def _static_zones(frames, conf, n):
    """Mirror the pipeline static-FP filter on the conf-thresholded candidates."""
    pts = []
    for cands in frames:
        for c in cands:
            if c["score"] >= conf:
                pts.append((int(c["cx"]) // 20, int(c["cy"]) // 20))
    cnt = Counter(pts)
    thr = max(10, int(n * 0.08))
    return {k for k, v in cnt.items() if v >= thr}


def _is_static(cx, cy, static):
    gx, gy = int(cx) // 20, int(cy) // 20
    return any((gx + dx, gy + dy) in static
               for dx in (-1, 0, 1) for dy in (-1, 0, 1))


def select_kalman(frames, conf, fw, fh, fps, *, no_static=False,
                  init_min_conf=None):
    """Production Kalman path replayed: static-FP filter → adaptive-gate Kalman,
    picking the candidate CLOSEST to the prediction (not the most confident).

    init_min_conf: candidates below this can only EXTEND a track (gated near the
    prediction), never SEED a cold init — a guard against low-conf parasites
    starting a phantom track far from the real ball.
    """
    import cv2
    n = len(frames)
    diag = float(np.hypot(fw, fh))
    static = set() if no_static else _static_zones(frames, conf, n)

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

    def update(x, y):
        kf.correct(np.array([[x], [y]], np.float32))
        fsu[0] = 0

    def gate_dist(x, y):
        p = kf.statePost
        return float(np.hypot(x - p[0][0], y - p[1][0]))

    base, growth = 150.0, 80.0
    gmax = diag * 0.35
    coast = int(fps * 0.12)
    reinit = diag * 0.50
    seed_conf = conf if init_min_conf is None else max(conf, init_min_conf)
    last = None
    track = [None] * n

    for i, cands in enumerate(frames):
        moving = [(c["cx"], c["cy"], c["score"]) for c in cands
                  if c["score"] >= conf and not _is_static(c["cx"], c["cy"], static)]
        c = None
        pred = predict()
        gate = min(gmax, base + growth * fsu[0])
        if moving:
            if not inited[0]:
                seeds = [m for m in moving if m[2] >= seed_conf] or moving
                b = max(seeds, key=lambda z: z[2])
                init(b[0], b[1])
                c = (b[0], b[1])
            else:
                bd, bdist = None, float("inf")
                for cx, cy, s in moving:
                    d = gate_dist(cx, cy)
                    if d < gate and d < bdist:
                        bdist, bd = d, (cx, cy)
                if bd is not None:
                    update(*bd)
                    st = kf.statePost
                    c = (float(st[0][0]), float(st[1][0]))
                elif fsu[0] <= coast and pred:
                    c = (pred[0], pred[1])
                elif last is not None:
                    nb, nd = None, float("inf")
                    for cx, cy, s in moving:
                        if s < seed_conf:
                            continue
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
    return track


# ───────────────────────── scoring ─────────────────────────
def load_gt():
    gt = json.load(open(GT))
    return {f["frame"]: f["ball"] for f in gt["frames"]
            if f.get("verified") and f.get("visible", True) and f.get("ball")}


def score(track, gtbf, fw, fps, tol_frac=0.02, start_sec=4.0):
    tol = tol_frac * fw
    start_n = int(start_sec * fps)
    cov = corr = n = 0
    cov_s = corr_s = n_s = 0
    for fidx, g in gtbf.items():
        if fidx >= len(track):
            continue
        n += 1
        is_s = fidx < start_n
        n_s += is_s
        t = track[fidx]
        if t is not None:
            cov += 1
            cov_s += is_s
            d = np.hypot(t[0] - g[0], t[1] - g[1])
            if d < tol:
                corr += 1
                corr_s += is_s
    return {
        "cov": cov, "corr": corr, "n": n,
        "cov_s": cov_s, "corr_s": corr_s, "n_s": n_s,
        "cov_pct": 100 * cov / max(1, n), "corr_pct": 100 * corr / max(1, n),
        "cov_s_pct": 100 * cov_s / max(1, n_s),
        "corr_s_pct": 100 * corr_s / max(1, n_s),
    }


def fmt(r):
    return (f"cov {r['cov']}/{r['n']}={r['cov_pct']:.1f}% "
            f"CORR {r['corr']}/{r['n']}={r['corr_pct']:.1f}% | "
            f"START cov {r['cov_s']}/{r['n_s']}={r['cov_s_pct']:.1f}% "
            f"CORR {r['corr_s']}/{r['n_s']}={r['corr_s_pct']:.1f}%")


def build_track(strat, frames, conf, fw, fh, fps, **kw):
    if strat == "highscore":
        return select_highscore(frames, conf, fw, fh)
    if strat == "kalman":
        return select_kalman(frames, conf, fw, fh, fps, **kw)
    raise ValueError(strat)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--strat", default="kalman", choices=["highscore", "kalman"])
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--init-min-conf", type=float, default=None)
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--tol-frac", type=float, default=0.02)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep a conf grid for both strategies")
    args = ap.parse_args()

    cache = json.loads(Path(args.cache).read_text())
    frames = cache["frames"]
    fw, fh, fps = cache["width"], cache["height"], cache["fps"]
    gtbf = load_gt()
    print(f"Cache {Path(args.cache).name}: imgsz={cache['imgsz']} "
          f"floor={cache['conf_floor']} {len(frames)} frames | GT {len(gtbf)} pts",
          flush=True)

    if args.sweep:
        grid = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25]
        for strat in ("highscore", "kalman"):
            print(f"\n--- strat={strat} ---", flush=True)
            for conf in grid:
                kw = {} if strat == "highscore" else {"no_static": args.no_static}
                t = build_track(strat, frames, conf, fw, fh, fps, **kw)
                r = score(t, gtbf, fw, fps, args.tol_frac)
                print(f"  conf={conf:.2f}  {fmt(r)}", flush=True)
        return

    kw = {}
    if args.strat == "kalman":
        kw = {"no_static": args.no_static, "init_min_conf": args.init_min_conf}
    t = build_track(args.strat, frames, args.conf, fw, fh, fps, **kw)
    r = score(t, gtbf, fw, fps, args.tol_frac)
    print(f"strat={args.strat} conf={args.conf} init_min_conf={args.init_min_conf} "
          f"no_static={args.no_static}", flush=True)
    print("  " + fmt(r), flush=True)


if __name__ == "__main__":
    main()
