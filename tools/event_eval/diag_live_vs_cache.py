"""
DIAGNOSTIC — replay the LIVE classify_events inputs (dumped via
COURTSIDE_DUMP_METHODO_INPUTS) offline and compare, candidate-by-candidate, to
the frozen demo3 cache + the GT. Pinpoints WHERE the prod-vs-cache event
divergence comes from (ball-track candidates? anchors? alternation phase?).

This does NOT re-run the pipeline; it consumes the JSON the live run already
serialized, so iteration is instant.

Run:
  venv/bin/python tools/event_eval/diag_live_vs_cache.py /tmp/methodoprod/live_inputs.json
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.events import classify_events  # noqa: E402
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]


def near(frame, pool, tol):
    """Return the nearest GT-ish frame within tol, or None."""
    hits = [f for f in pool if abs(f - frame) <= tol]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", help="live_inputs.json from a --event-methodo run")
    args = ap.parse_args()
    d = json.loads(Path(args.inputs).read_text())
    fps = d["fps"]
    fw, fh = d["width"], d["height"]
    tol = round(0.15 * fps)

    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    is_real = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    b_cands = [(int(f), int(x), int(y)) for f, x, y in d["b_cands"]]
    turns = sorted(int(t) for t in d["turns"])
    hits = d["hits"]
    ppf = d["methodo_ppf"]

    det = sum(1 for c in raw if c is not None)
    n = len(raw)
    print(f"LIVE inputs: {n} frames @ {fps}fps {fw}x{fh}")
    print(f"  raw ball detected : {det}/{n} = {100*det/n:.0f}%")
    far_present = sum(1 for slot in ppf if len(slot) >= 2)
    any_present = sum(1 for slot in ppf if len(slot) >= 1)
    print(f"  ppf frames w/ >=1 pose: {any_present}/{n} ({100*any_present/n:.0f}%); "
          f">=2 (near+far): {far_present}/{n} ({100*far_present/n:.0f}%)")
    print(f"  b_cands frames    : {sorted(f for f, _, _ in b_cands)}")
    print(f"  turns frames      : {turns}")
    print(f"  detect_hits frames: {sorted(int(h['frame']) for h in hits)}")
    print(f"  GT bounces : {GT_B}")
    print(f"  GT shots   : {GT_H}")

    # candidate-pool recall vs GT (which GT events have ANY candidate nearby?)
    pool = sorted(set([f for f, _, _ in b_cands]) | set(turns)
                  | set(int(h["frame"]) for h in hits))
    print("\n── POOL RECALL (does a candidate exist near each GT?) ──")
    for g in GT_B:
        print(f"  GT bounce {g:>3}: nearest pool cand = {near(g, pool, tol)}")
    for g in GT_H:
        print(f"  GT hit    {g:>3}: nearest pool cand = {near(g, pool, tol)} "
              f"(detect_hits? {near(g, [int(h['frame']) for h in hits], tol)})")

    # run the classifier exactly as prod does
    events, report = classify_events(
        b_cands, hits, raw, is_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=ac,
        wasb_centers=None, wasb_is_real=None)
    bev = sorted(e["frame"] for e in events if e["label"] == "BOUNCE")
    hev = sorted(e["frame"] for e in events if e["label"] == "HIT")
    print(f"\n── CLASSIFY_EVENTS (replayed offline) ──")
    print(f"  BOUNCE: {bev}")
    print(f"  HIT   : {hev}")

    ev_list = [{"frame": e["frame"], "label": e["label"]} for e in events]
    res = match_events(ev_list, GT_B, GT_H, tol)
    s = scores(res)
    print(f"\n  confusion_H->B = {res['conf_HtoB']}  confusion_B->H = {res['conf_BtoH']}")
    print(f"  bounce F1 = {s['bounce']['F1']:.3f}  hit F1 = {s['hit']['F1']:.3f}")

    # WHY each confusion: for each predicted BOUNCE on a GT-hit, was it hit_like?
    print("\n── CONFUSION ATTRIBUTION (predicted BOUNCE landing on a GT HIT) ──")
    for e in events:
        if e["label"] != "BOUNCE":
            continue
        gh = near(e["frame"], GT_H, tol)
        if gh is not None:
            f = e["frame"]
            had_hit = near(f, [int(h["frame"]) for h in hits], tol)
            slot = ppf[f] if 0 <= f < len(ppf) else []
            print(f"  pred BOUNCE@{f} hits GT HIT {gh}: detect_hits near? {had_hit}; "
                  f"poses@frame={len(slot)}; why={e.get('why')}; "
                  f"dist_norm={e['features'].get('dist_norm')}")


if __name__ == "__main__":
    main()
