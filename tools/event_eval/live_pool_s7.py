"""S7 worktree replay — live_pool.py but FORCE-LOADING the worktree's vision.events
so it measures THIS branch's detect_near_court_knees (import trap #2:
courtside-edit-targets-main-tree). Byte-identical logic to tools/event_eval/
live_pool.py otherwise; adds the knee generator to the pool.

Run:
  <main-venv>/bin/python tools/event_eval/live_pool_s7.py <dump.json>
from inside the worktree. Prints the loaded events.py path + asserts the new
symbol is present so we never silently measure the main tree.
"""
import sys
import json
import argparse
import importlib.util
from pathlib import Path

WT = Path(__file__).resolve().parents[2]   # the worktree root
sys.path.insert(0, str(WT))

# force-load the WORKTREE vision.events by file path (so detect_near_court_knees
# from THIS branch is what we measure, not the main tree's package).
import vision  # noqa: E402  (establish the package)
_spec = importlib.util.spec_from_file_location(
    "vision.events", str(WT / "vision" / "events.py"))
_E = importlib.util.module_from_spec(_spec)
sys.modules["vision.events"] = _E
_spec.loader.exec_module(_E)
assert _E.__file__.startswith(str(WT)), f"events.py not from worktree: {_E.__file__}"
assert hasattr(_E, "detect_near_court_knees"), "knee generator missing — wrong module"
classify_events = _E.classify_events
detect_turning_points = _E.detect_turning_points
detect_sharp_turns = _E.detect_sharp_turns
detect_near_court_knees = _E.detect_near_court_knees

from vision.bounce import (  # noqa: E402
    detect_bounces_robust)
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]
WIN = 8


def load(path):
    d = json.loads(Path(path).read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    is_real = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    hits = d["hits"]
    ppf = d["methodo_ppf"]
    return d, fps, fw, fh, raw, is_real, ac, hits, ppf


def _near(f, pool, win=WIN):
    cands = [p for p in pool if abs(f - p) <= win]
    return min(cands, key=lambda p: abs(p - f)) if cands else None


def generators(raw, is_real, ac, ppf, hits, fps, fw, fh):
    b_cands, _re, _nd = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = detect_turning_points(ac, fps, fh)
    sharp = detect_sharp_turns(raw, is_real, fps, fw, fh)
    knees = detect_near_court_knees(raw, is_real, fps, fw, fh)
    hit_fr = [int(h["frame"]) for h in hits]
    return {
        "robust_bounce": sorted(int(f) for f, *_ in b_cands),
        "turning_points": sorted(int(f) for f in turns),
        "sharp_turns": sorted(int(f) for f in sharp),
        "near_knees": sorted(int(f) for f in knees),
        "detect_hits": sorted(hit_fr),
    }, b_cands


def report(path):
    d, fps, fw, fh, raw, is_real, ac, hits, ppf = load(path)
    print(f"[events.py] {_E.__file__}")
    n = len(raw)
    det = sum(1 for c in raw if c is not None)
    print(f"\n=== LIVE POOL THERMOMETER S7 ({Path(path).name}) ===")
    print(f"{n} frames @ {fps}fps {fw}x{fh} | raw ball {det}/{n} ({100*det/n:.0f}%)")

    gens, b_cands = generators(raw, is_real, ac, ppf, hits, fps, fw, fh)
    pool = sorted(set().union(*[set(v) for v in gens.values()]))
    print(f"\npool size = {len(pool)}; per-generator: "
          + ", ".join(f"{k}={len(v)}" for k, v in gens.items()))
    print(f"  near_knees frames = {gens['near_knees']}")

    print("\n── PER-GT POOL RECALL ──")
    hit_b = hit_h = 0
    for label, gts in (("BOUNCE", GT_B), ("HIT", GT_H)):
        for g in gts:
            srcs = [k for k, v in gens.items() if _near(g, v) is not None]
            present = bool(srcs)
            hit_b += present if label == "BOUNCE" else 0
            hit_h += present if label == "HIT" else 0
            tag = "OK " if present else "MISS"
            print(f"  GT {label:<6} {g:>3}: {tag}  src={srcs if srcs else '— (NO CANDIDATE)'}")
    print(f"\nPOOL RECALL = {hit_b + hit_h}/17  (bounces {hit_b}/9, hits {hit_h}/8)")

    turns = sorted(set(gens["turning_points"]) | set(gens["sharp_turns"])
                   | set(gens["near_knees"]))
    events, _rep = classify_events(
        b_cands, hits, raw, is_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=ac,
        wasb_centers=None, wasb_is_real=None)
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    tol = round(0.15 * fps)
    res = match_events(ev, GT_B, GT_H, tol)
    s = scores(res)
    bev = sorted(e["frame"] for e in events if e["label"] == "BOUNCE")
    hev = sorted(e["frame"] for e in events if e["label"] == "HIT")
    print(f"\n── CLASSIFY_EVENTS (replayed, S7 pool) ──")
    print(f"  pred BOUNCE: {bev}")
    print(f"  pred HIT   : {hev}")
    print(f"  confusion H->B={res['conf_HtoB']} B->H={res['conf_BtoH']} | "
          f"bounce F1={s['bounce']['F1']:.3f} (R={s['bounce']['R']:.2f}) | "
          f"hit F1={s['hit']['F1']:.3f} (R={s['hit']['R']:.2f})")
    print(f"  bounce misses: {res['FN_b']}  hit misses: {res['FN_h']}")
    print(f"  spurious: {res['spurious']}")
    return {"pool_recall": hit_b + hit_h, "res": res, "scores": s, "gens": gens}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    report(ap.parse_args().inputs)
