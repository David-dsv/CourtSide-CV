"""THROWAWAY: unified far-selection STRATEGY sweep over demo3-GT AND felix-proxy-GT.

The two clips have OPPOSITE far-player geometries:
  demo3 (steep broadcast): far player CENTRAL (xr~0.42), TINY, FAILS keypoint gate.
  felix (corner/grazing):  far player OFF-CENTRE (xr~0.74), bigger, PASSES keypoint gate.
A pure-centrality score wins demo3 but breaks felix; a pure keypoint gate wins felix
but breaks demo3. We need ONE selector that wins both.

This caches posed far candidates for both clips ONCE (slow), then tests several
SELECTION STRATEGIES cheaply (not just linear weight blends), reporting demo3%,
felix%, min(both). The winner becomes the production far-selection block.

  python tools/pose_gt/_strat_sweep.py
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


CACHE_F = "/tmp/strat_cache.json"


def collect(clip, gtbf, det, pm):
    """For each GT frame: detect persons, keep above-net candidates, pose each,
    cache geometric + skeleton features. Returns {fidx: [row,...]}, fw, fh, net_y.
    Decodes ONLY the GT frames (seek), not the whole video, to stay fast on felix."""
    r = VideoReader(str(clip))
    fw, fh = r.width, r.height
    ny = 0.5 * fh
    want = set(int(i) for i in gtbf)
    maxf = max(want)
    frames = {}
    for i, f in enumerate(r.iter_frames()):
        if i in want:
            frames[i] = f
        if i >= maxf:
            break
    r.release()
    cache = {}
    for fidx in gtbf:
        if fidx not in frames:
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
            if cy >= ny:                     # centre-based far bucket
                continue
            w, h = x2 - x1, y2 - y1
            if h < 0.012 * fh:
                continue
            res = P.pose_on_crop(pm, f, b, "cpu", imgsz=960, conf=0.15)
            kcf = res[1] if res else None
            valid = P.is_valid_player(b, kcf, fw, fh, None, band=None)
            rows.append(dict(cx=float(cx), cy=float(cy),
                             asp=float(h / max(w, 1.0)), conf=float(c),
                             valid=int(valid), h=float(h), w=float(w)))
        cache[fidx] = rows
    return cache, fw, fh, ny


# ---- selection strategies. each returns the chosen row (or None) ----

def s_central(rows, fw, fh, ny):
    """Pure camera-geometry _far_score (committed c50b00b). Wins demo3, breaks felix."""
    if not rows:
        return None
    return max(rows, key=lambda r: _geom(r, fw, ny))


def s_valid_then_central(rows, fw, fh, ny):
    """Tier1 = valid+on-court (rank h*conf); Tier2 = geometry. The committed broken
    one — Tier1 admits a lower-stands spectator on demo3."""
    knee = 0.25 * ny
    t1 = [r for r in rows if r["valid"] and r["cy"] >= knee and r["h"] >= 0.04 * fh]
    if t1:
        return max(t1, key=lambda r: r["conf"] * r["h"])
    return max(rows, key=lambda r: _geom(r, fw, ny)) if rows else None


def s_valid_central_then_central(rows, fw, fh, ny):
    """Tier1 = valid + on-court + NOT-lateral (the demo3 spectator leak was a lateral
    lower-stands box at xr~0.80); Tier2 = geometry. The lateral gate is what should
    fix demo3 while keeping felix (felix far player xr~0.74 — borderline, so use a
    LOOSE lateral cut: reject only xr>0.85 or xr<0.15)."""
    knee = 0.25 * ny
    t1 = [r for r in rows
          if r["valid"] and r["cy"] >= knee and r["h"] >= 0.04 * fh
          and 0.15 < r["cx"] / fw < 0.85]
    if t1:
        return max(t1, key=lambda r: r["conf"] * r["h"])
    return max(rows, key=lambda r: _geom(r, fw, ny)) if rows else None


def s_blend(wv):
    """Single unified score: geometry + wv*(valid & on-court & not-lateral)."""
    def f(rows, fw, fh, ny):
        if not rows:
            return None
        knee = 0.25 * ny
        def sc(r):
            s = _geom(r, fw, ny)
            if (r["valid"] and r["cy"] >= knee and r["h"] >= 0.04 * fh
                    and 0.15 < r["cx"] / fw < 0.85):
                s += wv
            return s
        return max(rows, key=sc)
    return f


def _geom(r, fw, ny):
    central = 1.0 - abs(r["cx"] / fw - 0.5) * 2.0
    human = 1.0 if r["asp"] <= 2.2 else max(0.0, 1.0 - (r["asp"] - 2.2) * 0.5)
    s = central + 0.5 * human + 0.2 * r["conf"]
    knee = 0.25 * ny
    if r["cy"] < knee:
        s -= 1.0 * (1.0 - r["cy"] / knee)
    return s


# The cross-geometry insight (from _probe_cache): on BOTH clips the real far player
# is the TALLEST on-court above-net box. felix GT h~0.06-0.07H beats its central
# structure villain (0.03H); demo3 GT h~0.11H beats its lateral spectator (0.07H).
# Height is robust because a distant *player* is bigger than a distant *spectator*
# (seated, further back). Centrality only helps demo3 and actively breaks felix.

def _oncourt(r, fw, ny):
    """On the playing surface: below the top-stands knee, inside the lateral band."""
    return r["cy"] >= 0.25 * ny and 0.12 < r["cx"] / fw < 0.88


def s_height(rows, fw, fh, ny):
    """Pick the TALLEST on-court above-net box. Pure size, no centrality."""
    c = [r for r in rows if _oncourt(r, fw, ny)]
    if not c:
        c = rows
    return max(c, key=lambda r: r["h"]) if c else None


def s_height_conf(rows, fw, fh, ny):
    """Tallest * confidence, on-court. Conf breaks ties toward the real subject."""
    c = [r for r in rows if _oncourt(r, fw, ny)]
    if not c:
        c = rows
    return max(c, key=lambda r: r["h"] * (0.5 + r["conf"])) if c else None


def s_hcv(wc, wconf=0.3, wval=0.3, wh=1.0):
    """Unified: wh*height(norm) + wconf*conf + wc*centrality + wval*valid, on-court.
    Sweep the weights to find the cross-geometry sweet spot."""
    def f(rows, fw, fh, ny):
        c = [r for r in rows if _oncourt(r, fw, ny)]
        if not c:
            c = rows
        if not c:
            return None
        hmax = max(r["h"] for r in c) or 1.0
        def sc(r):
            central = 1.0 - abs(r["cx"] / fw - 0.5) * 2.0
            return (wh * r["h"] / hmax + wconf * r["conf"] + wc * central
                    + wval * r["valid"])
        return max(c, key=sc)
    return f


def s_hv(wv):
    """Height(norm) + wv*valid + small conf, on-court. No centrality at all —
    tests whether height+valid alone wins both (the truly geometry-free selector)."""
    def f(rows, fw, fh, ny):
        c = [r for r in rows if _oncourt(r, fw, ny)]
        if not c:
            c = rows
        if not c:
            return None
        hmax = max(r["h"] for r in c) or 1.0
        return max(c, key=lambda r: r["h"] / hmax + wv * r["valid"] + 0.2 * r["conf"])
    return f


def evalc(cache, gtbf, fw, ny, strat):
    ok = n = 0
    for fidx, (gx, gy) in gtbf.items():
        rows = cache.get(fidx, [])
        if not rows:
            continue
        n += 1
        best = strat(rows, fw, 2 * ny, ny)     # fh = 2*ny (ny = 0.5*fh)
        if best and ((best["cx"] - gx) ** 2 + (best["cy"] - gy) ** 2) ** 0.5 < 0.10 * fw:
            ok += 1
    return ok, n


def main():
    dgt = json.load(open(DEMO_GT))
    dgtbf = {f["frame"]: f["p1_far"]["center"] for f in dgt["frames"]
             if f["p1_far"].get("verified") and f["p1_far"].get("center")}
    fgt = {int(k): v for k, v in json.load(open(FELIX_GT)).items()}
    print(f"demo3 GT frames={len(dgtbf)}  felix GT frames={len(fgt)}", flush=True)

    if "--use-cache" in sys.argv and Path(CACHE_F).exists():
        c = json.load(open(CACHE_F))
        dcache = {int(k): v for k, v in c["demo"].items()}
        fcache = {int(k): v for k, v in c["felix"].items()}
        dfw, dny = c["dfw"], c["dny"]
        ffw, fny = c["ffw"], c["fny"]
        print("loaded posed cache from disk (instant)", flush=True)
    else:
        det = TennisYOLODetector(device="cpu")
        pm = YOLO("yolov8m-pose.pt")
        print("collecting demo3...", flush=True)
        dcache, dfw, dfh, dny = collect(DEMO, dgtbf, det, pm)
        print("collecting felix...", flush=True)
        fcache, ffw, ffh, fny = collect(FELIX, fgt, det, pm)
        try:                                   # cache write must NEVER lose results
            json.dump({"demo": dcache, "felix": fcache, "dfw": dfw, "dny": dny,
                       "ffw": ffw, "fny": fny}, open(CACHE_F, "w"))
            print(f"saved posed cache -> {CACHE_F}", flush=True)
        except Exception as e:
            print(f"WARN: cache write failed ({e}); continuing to eval", flush=True)

    # ROBUSTNESS GRID around the wc=1.2 plateau: vary each weight ±, confirm the
    # min stays high across a neighbourhood (not a knife-edge = not overfit).
    strats = [
        ("central (c50b00b)", s_central),
        ("hcv wc=1.2 (pick)", s_hcv(1.2)),
        ("  wc=1.1", s_hcv(1.1)),
        ("  wc=1.25", s_hcv(1.25)),
        ("  wc=1.2 wh=0.8", s_hcv(1.2, wh=0.8)),
        ("  wc=1.2 wh=1.2", s_hcv(1.2, wh=1.2)),
        ("  wc=1.2 wconf=0.2", s_hcv(1.2, wconf=0.2)),
        ("  wc=1.2 wconf=0.4", s_hcv(1.2, wconf=0.4)),
        ("  wc=1.2 wval=0.2", s_hcv(1.2, wval=0.2)),
        ("  wc=1.2 wval=0.4", s_hcv(1.2, wval=0.4)),
        ("  wc=1.2 wval=0.0", s_hcv(1.2, wval=0.0)),
    ]
    print(f"\n  {'strategy':<28} demo3%   felix%   min")
    best = None
    for name, strat in strats:
        do, dn = evalc(dcache, dgtbf, dfw, dny, strat)
        fo, fn = evalc(fcache, fgt, ffw, fny, strat)
        dp = 100 * do / max(1, dn)
        fp = 100 * fo / max(1, fn)
        m = min(dp, fp)
        flag = ""
        if best is None or m > best[0]:
            best = (m, name); flag = "  <-- best min"
        print(f"  {name:<28} {dp:5.1f}%  {fp:5.1f}%  {m:5.1f}%{flag}")
    print(f"\nBEST: {best[1]} (min {best[0]:.1f}%)")


if __name__ == "__main__":
    main()
