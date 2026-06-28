"""LIVE candidate-pool thermometer — per-generator recall on the REAL pipeline
track, replayed from a COURTSIDE_DUMP_METHODO_INPUTS dump (no re-run, instant).

The committed demo3 cache shows pool recall 17/17, but the PROMPT (S3) says the
LIVE bounce recall is ~5/9 — the cache LIES (its frozen Kalman track differs from
what the live pipeline produces today). This tool answers, on the EXACT live
inputs the pipeline serialized, the one question that is the recall ceiling:

  Of the 17 GT events (9 bounces + 8 hits), how many have AT LEAST ONE candidate
  within ±8f — and which GENERATOR (robust-bounce / turning-points / sharp-turns /
  detect-hits) contributes each? The misses are the candidate-generation gap.

It re-derives the candidate sets from the dumped RAW + spline tracks using the
SAME calls the pipeline makes, so adding/changing a generator here measures the
live pool delta with zero pipeline cost. It then replays classify_events to show
the end-to-end confusion + F1 (which is what score_stats.py reports on _stats.json).

Run:
  venv/bin/python tools/event_eval/live_pool.py /tmp/s3/live_inputs_run1.json
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import (  # noqa: E402
    smooth_ball_trajectory, detect_bounces_robust)
from vision.shots import detect_hits  # noqa: E402
from vision.events import (  # noqa: E402
    classify_events, detect_turning_points, detect_sharp_turns)
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
    """Re-derive each candidate set on the LIVE tracks, exactly as the pipeline."""
    b_cands, _re, _nd = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = detect_turning_points(ac, fps, fh)
    sharp = detect_sharp_turns(raw, is_real, fps, fw, fh)
    hit_fr = [int(h["frame"]) for h in hits]
    return {
        "robust_bounce": sorted(int(f) for f, *_ in b_cands),
        "turning_points": sorted(int(f) for f in turns),
        "sharp_turns": sorted(int(f) for f in sharp),
        "detect_hits": sorted(hit_fr),
    }, b_cands


def report(path):
    d, fps, fw, fh, raw, is_real, ac, hits, ppf = load(path)
    n = len(raw)
    det = sum(1 for c in raw if c is not None)
    any_pose = sum(1 for s in ppf if len(s) >= 1)
    far_pose = sum(1 for s in ppf if len(s) >= 2)
    print(f"\n=== LIVE POOL THERMOMETER ({Path(path).name}) ===")
    print(f"{n} frames @ {fps}fps {fw}x{fh} | raw ball {det}/{n} ({100*det/n:.0f}%) "
          f"| pose>=1 {100*any_pose/n:.0f}% / far(>=2) {100*far_pose/n:.0f}%")

    gens, b_cands = generators(raw, is_real, ac, ppf, hits, fps, fw, fh)
    pool = sorted(set().union(*[set(v) for v in gens.values()]))
    print(f"\npool size = {len(pool)} candidates; per-generator counts: "
          + ", ".join(f"{k}={len(v)}" for k, v in gens.items()))

    print("\n── PER-GT POOL RECALL (which generator supplies each GT?) ──")
    hit_b = hit_h = 0
    for label, gts in (("BOUNCE", GT_B), ("HIT", GT_H)):
        for g in gts:
            srcs = [k for k, v in gens.items() if _near(g, v) is not None]
            present = bool(srcs)
            if label == "BOUNCE":
                hit_b += present
            else:
                hit_h += present
            tag = "OK " if present else "MISS"
            print(f"  GT {label:<6} {g:>3}: {tag}  src={srcs if srcs else '— (NO CANDIDATE)'}")
    print(f"\nPOOL RECALL = {hit_b + hit_h}/17  (bounces {hit_b}/9, hits {hit_h}/8)")

    # end-to-end replay (the score_stats.py number)
    turns = sorted(set(gens["turning_points"]) | set(gens["sharp_turns"]))
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
    print(f"\n── CLASSIFY_EVENTS (replayed) ──")
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
    ap.add_argument("inputs", help="live_inputs.json from COURTSIDE_DUMP_METHODO_INPUTS")
    report(ap.parse_args().inputs)
