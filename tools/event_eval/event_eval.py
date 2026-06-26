"""
Bounce<->Shot CONFUSION evaluator — the thermometer for bounce-vs-hit accuracy.

The bounce evaluator (tools/bounce_eval) scores bounces against bounce GT in
isolation. But the user's bug is CONFUSION: a racket HIT emitted as a BOUNCE (and
vice-versa). Measuring that needs BOTH ground truths at once and a single cross
matcher, so a predicted "bounce" landing on a GT-strike counts as confusion_H->B,
not as an innocent false positive.

This is a PURE comparator: it takes a list of predicted events (each tagged
"BOUNCE" or "HIT" with a frame) and the two GT lists, runs ONE greedy 1-1 matcher
across all of them, and reports the 2x3 confusion matrix:

                 Pred BOUNCE   Pred HIT     Missed
    GT BOUNCE       TP_b      confusion B->H  FN_b
    GT HIT      confusion H->B    TP_h        FN_h

  confusion_H->B  = HIT GT matched by a predicted BOUNCE  <-- the user's bug (#1)
  confusion_B->H  = BOUNCE GT matched by a predicted HIT
  spurious        = predicted events that matched no GT at all (pure FP)

Disambiguation rule (docs/research/bounce-vs-shot-direction.md §5.2), applied so
the matrix is deterministic when a prediction is within tolerance of BOTH a GT
bounce and a GT strike:
  1. Closest frame wins.
  2. On a distance TIE, the predicted label decides which GT it pairs to first
     (a predicted BOUNCE pairs to the GT bounce; a predicted HIT to the GT shot).
  3. One-to-one greedy, nearest first: each GT and each prediction is used once.

Default tolerance = round(0.15 * fps), matching tools/bounce_eval/bounce_io.py.
"""
import sys
import json
import argparse
from pathlib import Path

BOUNCE, HIT = "BOUNCE", "HIT"


def _tie_break_key(d, pred_label, gt_class):
    """Sort key for candidate (distance, pred, gt) pairs implementing §5.2.

    Primary: distance (closest first). Secondary: on a tie, a same-class pairing
    (pred BOUNCE -> GT bounce, pred HIT -> GT shot) wins over a cross pairing, so
    the prediction's own label decides its first claim. Tertiary: stable by frames.
    """
    same_class = 0 if pred_label == gt_class else 1
    return (d, same_class)


def match_events(pred_events, gt_bounce_frames, gt_shot_frames, tolerance):
    """One greedy 1-1 cross-match of predicted events vs both GTs.

    Args:
        pred_events: list of dicts {"frame": int, "label": "BOUNCE"|"HIT"}.
        gt_bounce_frames: list[int] GT bounce frames.
        gt_shot_frames:   list[int] GT shot frames.
        tolerance: match window in frames.

    Returns a result dict with the confusion counts + the matched pairs and the
    unmatched predictions/GT (for diagnostics).
    """
    # GT pool: (class, frame, uid). uid keeps duplicates distinct.
    gt = ([("bounce", f, ("b", i)) for i, f in enumerate(gt_bounce_frames)]
          + [("shot", f, ("s", i)) for i, f in enumerate(gt_shot_frames)])
    pred = [(e["label"], int(e["frame"]), ("p", i)) for i, e in enumerate(pred_events)]

    # all candidate pairs within tolerance
    cand = []
    for pl, pf, puid in pred:
        for gc, gf, guid in gt:
            d = abs(pf - gf)
            if d <= tolerance:
                cand.append((d, pl, pf, puid, gc, gf, guid))
    # sort: distance, then §5.2 tie-break (same-class first), then frames/uids stable
    cand.sort(key=lambda c: (_tie_break_key(c[0], c[1], c[4]), c[2], c[5], c[3], c[6]))

    used_pred, used_gt, pairs = set(), set(), []
    for d, pl, pf, puid, gc, gf, guid in cand:
        if puid in used_pred or guid in used_gt:
            continue
        used_pred.add(puid)
        used_gt.add(guid)
        pairs.append({"pred_label": pl, "pred_frame": pf,
                      "gt_class": gc, "gt_frame": gf, "dist": d})

    # tally confusion matrix
    m = {"TP_b": 0, "TP_h": 0, "conf_BtoH": 0, "conf_HtoB": 0}
    for p in pairs:
        if p["gt_class"] == "bounce":
            m["TP_b" if p["pred_label"] == BOUNCE else "conf_BtoH"] += 1
        else:  # shot
            m["TP_h" if p["pred_label"] == HIT else "conf_HtoB"] += 1

    matched_gt = {(p["gt_class"], p["gt_frame"]) for p in pairs}
    # NOTE: gt_frame collisions across same class are rare in our GT; keep simple.
    used_gt_uids = used_gt
    fn_b = sorted(f for (gc, f, guid) in gt if gc == "bounce" and guid not in used_gt_uids)
    fn_h = sorted(f for (gc, f, guid) in gt if gc == "shot" and guid not in used_gt_uids)
    spurious = sorted((pf, pl) for (pl, pf, puid) in pred if puid not in used_pred)

    return {
        "pairs": pairs,
        "TP_b": m["TP_b"], "TP_h": m["TP_h"],
        "conf_BtoH": m["conf_BtoH"], "conf_HtoB": m["conf_HtoB"],
        "FN_b": fn_b, "FN_h": fn_h,
        "spurious": spurious,
        "n_gt_b": len(gt_bounce_frames), "n_gt_h": len(gt_shot_frames),
    }


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def scores(res):
    """Per-class precision/recall/F1 from a match_events result.

    For BOUNCE: a predicted bounce is correct only if it lands on a GT bounce
    (TP_b). A predicted bounce on a GT shot (conf_HtoB) is a bounce FP; a spurious
    predicted bounce is a bounce FP; a GT bounce missed or stolen by a HIT pred
    (FN_b + conf_BtoH) is a bounce FN. Symmetric for HIT.
    """
    spur_b = sum(1 for _, lab in res["spurious"] if lab == BOUNCE)
    spur_h = sum(1 for _, lab in res["spurious"] if lab == HIT)
    # bounce
    b_tp = res["TP_b"]
    b_fp = res["conf_HtoB"] + spur_b            # pred BOUNCE that wasn't a GT bounce
    b_fn = len(res["FN_b"]) + res["conf_BtoH"]  # GT bounce missed or taken by a HIT pred
    # hit
    h_tp = res["TP_h"]
    h_fp = res["conf_BtoH"] + spur_h
    h_fn = len(res["FN_h"]) + res["conf_HtoB"]
    bp, br, bf1 = _prf(b_tp, b_fp, b_fn)
    hp, hr, hf1 = _prf(h_tp, h_fp, h_fn)
    return {
        "bounce": {"tp": b_tp, "fp": b_fp, "fn": b_fn, "P": bp, "R": br, "F1": bf1},
        "hit": {"tp": h_tp, "fp": h_fp, "fn": h_fn, "P": hp, "R": hr, "F1": hf1},
    }


def print_report(res, fps=None, tol=None, title=""):
    s = scores(res)
    if title:
        print(f"── {title} ──")
    if tol is not None:
        tol_s = f" (±{tol}f" + (f", {tol/fps:.2f}s @ {fps:.0f}fps)" if fps else ")")
        print(f"tolerance:{tol_s}")
    print(f"GT bounces: {res['n_gt_b']} | GT shots: {res['n_gt_h']}")
    print("CONFUSION MATRIX")
    print("                 Pred BOUNCE   Pred HIT     Missed")
    print(f"  GT BOUNCE        {res['TP_b']:>5}      {res['conf_BtoH']:>7}     "
          f"{len(res['FN_b']):>5}")
    print(f"  GT HIT       {res['conf_HtoB']:>7}        {res['TP_h']:>5}     "
          f"{len(res['FN_h']):>5}")
    print(f"  >>> confusion_H->B = {res['conf_HtoB']}  (HITs shown as bounces — the bug)")
    print(f"  >>> confusion_B->H = {res['conf_BtoH']}  (bounces shown as hits)")
    if res["spurious"]:
        print(f"  spurious preds (no GT): {res['spurious']}")
    print(f"  BOUNCE  P={s['bounce']['P']:.3f} R={s['bounce']['R']:.3f} "
          f"F1={s['bounce']['F1']:.3f}  (TP={s['bounce']['tp']} FP={s['bounce']['fp']} FN={s['bounce']['fn']})")
    print(f"  HIT     P={s['hit']['P']:.3f} R={s['hit']['R']:.3f} "
          f"F1={s['hit']['F1']:.3f}  (TP={s['hit']['tp']} FP={s['hit']['fp']} FN={s['hit']['fn']})")
    if res["FN_b"]:
        print(f"  bounce misses: {res['FN_b']}")
    if res["FN_h"]:
        print(f"  hit misses:    {res['FN_h']}")


def load_pred_events(path):
    """Load a predicted-events file: {fps, events:[{frame,label}]} OR the union of
    a bounce file + a shots file isn't done here — predictions carry their own
    label. Accepts either 'events' or separate 'bounces'/'shots' arrays."""
    doc = json.loads(Path(path).read_text())
    if "events" in doc:
        return doc, [{"frame": int(e["frame"]), "label": e["label"]} for e in doc["events"]]
    ev = []
    for b in doc.get("bounces", []):
        ev.append({"frame": int(b["frame"]), "label": BOUNCE})
    for s in doc.get("shots", []):
        ev.append({"frame": int(s["frame"]), "label": HIT})
    return doc, ev


def parse_args():
    p = argparse.ArgumentParser(description="Bounce<->shot confusion evaluator.")
    p.add_argument("--pred", required=True,
                   help="Predicted events JSON ({events:[{frame,label}]} or bounces/shots).")
    p.add_argument("--truth-bounces", required=True, help="GT bounces JSON (bounce_io format).")
    p.add_argument("--truth-shots", required=True, help="GT shots JSON ({shots:[{frame,...}]}).")
    p.add_argument("--tolerance-frames", type=int, default=None,
                   help="Match window (default round(0.15*fps)).")
    return p.parse_args()


def main():
    args = parse_args()
    pred_doc, pred_events = load_pred_events(args.pred)
    tb = json.loads(Path(args.truth_bounces).read_text())
    ts = json.loads(Path(args.truth_shots).read_text())
    fps = tb.get("fps") or ts.get("fps") or pred_doc.get("fps") or 0.0
    tol = args.tolerance_frames if args.tolerance_frames is not None else (
        round(0.15 * fps) if fps else 4)
    gt_b = [int(b["frame"]) for b in tb["bounces"]]
    gt_h = [int(s["frame"]) for s in ts["shots"]]
    res = match_events(pred_events, gt_b, gt_h, tol)
    print_report(res, fps=fps, tol=tol, title=Path(args.pred).name)


if __name__ == "__main__":
    sys.exit(main())
