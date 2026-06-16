"""
Bounce-detection evaluation — the "thermometer".

Compares predicted bounces against hand-annotated ground truth for the same
clip and reports precision / recall / F1. This is a PURE comparator: it reads
two JSON files (same shared bounce format) and does no video decoding or model
inference, so it runs in milliseconds and is fully decoupled from the pipeline.

Predictions are expected in the same format as ground truth (see bounce_io.py);
they will be produced by a future, gated `--dump-bounces` export step in
run_pipeline_8s.py. Until then you can hand-write a predictions file to test.

Matching rule
-------------
A predicted bounce is a TRUE POSITIVE if it lies within ±tolerance frames of a
ground-truth bounce. Matching is one-to-one and greedy by smallest frame
distance: each GT bounce matches at most one prediction and vice versa, so
duplicate predictions near one real bounce count as false positives (not free
extra credit). Unmatched predictions = false positives; unmatched GT = misses.

Default tolerance = round(0.15 * fps) (~150 ms). Override with
--tolerance-frames. fps is read from the ground-truth file.

Usage
-----
    python tools/bounce_eval/eval_bounces.py \\
        --pred  data/preds/match1.bounces.json \\
        --truth tests/fixtures/bounces/match1.bounces.json

    # fixed tolerance regardless of fps
    python tools/bounce_eval/eval_bounces.py --pred ... --truth ... \\
        --tolerance-frames 4
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from tools.bounce_eval.bounce_io import load_bounce_file, absolute_frames  # noqa: E402


def match_bounces(pred_frames, truth_frames, tolerance):
    """
    One-to-one greedy nearest-frame matching within ±tolerance.

    Returns (matches, false_positives, false_negatives) where matches is a list
    of (pred_frame, truth_frame) pairs.
    """
    # All candidate (distance, pred, truth) pairs within tolerance, nearest first.
    candidates = []
    for p in pred_frames:
        for t in truth_frames:
            d = abs(p - t)
            if d <= tolerance:
                candidates.append((d, p, t))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))

    used_pred, used_truth, matches = set(), set(), []
    for d, p, t in candidates:
        if p in used_pred or t in used_truth:
            continue
        used_pred.add(p)
        used_truth.add(t)
        matches.append((p, t))

    fp = sorted(p for p in pred_frames if p not in used_pred)
    fn = sorted(t for t in truth_frames if t not in used_truth)
    return matches, fp, fn


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate bounce predictions vs ground truth.")
    p.add_argument("--pred", required=True, help="Predictions JSON")
    p.add_argument("--truth", required=True, help="Ground-truth JSON")
    p.add_argument("--tolerance-frames", type=int, default=None,
                   help="Match window in frames (default: round(0.15*fps))")
    return p.parse_args()


def main():
    args = parse_args()
    pred_doc = load_bounce_file(args.pred)
    truth_doc = load_bounce_file(args.truth)

    fps = truth_doc.get("fps") or pred_doc.get("fps") or 0.0
    if args.tolerance_frames is not None:
        tol = args.tolerance_frames
    elif fps > 0:
        tol = round(0.15 * fps)
    else:
        tol = 4
        print("WARNING: no fps in either file; defaulting tolerance to 4 frames.")

    pred_frames = absolute_frames(pred_doc)
    truth_frames = absolute_frames(truth_doc)

    matches, fp, fn = match_bounces(pred_frames, truth_frames, tol)
    tp = len(matches)
    precision, recall, f1 = prf(tp, len(fp), len(fn))

    tol_s = f"{tol/fps:.2f}s @ {fps:.0f}fps" if fps > 0 else "fps unknown"
    print(f"Clip: {Path(args.truth).name}  | tolerance: ±{tol} frames ({tol_s})")
    print(f"GT bounces: {len(truth_frames)} | Predicted: {len(pred_frames)}")
    print(f"True positives: {tp} | False positives: {len(fp)} | False negatives: {len(fn)}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    if fn:
        print(f"Misses (GT frames):  {fn}")
    if fp:
        print(f"False alarms (pred): {fp}")

    return f1


if __name__ == "__main__":
    main()
