"""
Forehand/Backhand SCORER — the thermometer for stroke-type accuracy.

The event evaluator (event_eval.py) scores WHEN a hit happens (bounce-vs-hit
confusion); this scorer scores WHAT a hit is (forehand vs backhand). It reads the
predicted strokes (the LIVE ``_stats.json`` written by run_pipeline_8s.py, or a
``--dump-shots`` JSON), pairs each predicted shot to the nearest ground-truth shot
within a frame tolerance (greedy 1-1), and reports the FH/BH metrics ON THE PAIRED
SHOTS ONLY:

    accuracy_called = correct / called          # called = matched AND not 'unknown'
    abstain_rate    = unknown / matched         # an abstention is neither right nor
                                                #   wrong — it's an honest refusal
    confusion FH<->BH:  FH->BH (GT fh shown bh) + BH->FH (GT bh shown fh)
    matched / missed (GT with no prediction) / extra (prediction with no GT)

Why measure on the LIVE _stats.json and not a frozen cache: a cache can pass while
prod diverges (the --event-methodo prod gap lesson). The number that matters is the
one the product actually emits, so this scorer's default input IS the _stats.json.

GT format: tests/fixtures/shots/<clip>.shots.json — frames are CLIP-LOCAL (0-based
within the -s/-d window). The _stats.json shots carry ABSOLUTE frames
(start_frame + clip_local); we read frame_range[0] from the stats to convert GT to
absolute before matching (no hardcoded offset).

Matching tolerance defaults to round(0.15*fps) — the wrist-speed peak lags the true
contact by a few frames (see vision/shots.py:detect_hits), so a moderate window is
needed; 0.15*fps ≈ 7 frames @50fps, the same order as the event matcher.

Usage:
    # score the LIVE stats directly
    python tools/event_eval/score_fhb.py \
        --stats data/output/tennis_demo3_annotated_stats.json \
        --truth tests/fixtures/shots/tennis_demo3.shots.json

    # or a --dump-shots predictions file (same shot schema, list under "shots")
    python tools/event_eval/score_fhb.py --pred preds.json --truth ... --fps 50
"""
import sys
import json
import argparse
from pathlib import Path

FH, BH, UNK = "forehand", "backhand", "unknown"


def _load_pred_shots(stats_path=None, pred_path=None):
    """Return (shots, fps, frame_offset). shots = [{frame:int, type:str}] in
    ABSOLUTE frames. Accepts a LIVE _stats.json (shots[].stroke, frame_range[0])
    or a --dump-shots predictions file (shots[].type|stroke, frame_offset|fps)."""
    path = stats_path or pred_path
    doc = json.loads(Path(path).read_text())
    raw = doc.get("shots", [])
    fps = doc.get("fps")
    # frame offset: _stats.json -> frame_range[0]; dump -> frame_offset; else 0
    offset = 0
    if "frame_range" in doc and doc["frame_range"]:
        offset = int(doc["frame_range"][0])
    elif "frame_offset" in doc:
        offset = int(doc["frame_offset"])
    shots = []
    for s in raw:
        t = s.get("stroke", s.get("type", s.get("fhb", UNK)))
        # frame in stats is already absolute; in a dump it may be local — but our
        # dump writer (run_pipeline_8s --dump-shots) writes absolute too, so trust
        # the field and only fall back to offset-adding if a 'local' flag is set.
        f = int(s["frame"])
        shots.append({"frame": f, "type": t})
    return shots, fps, offset


def _load_gt(truth_path):
    doc = json.loads(Path(truth_path).read_text())
    fps = doc.get("fps")
    # GT frames are CLIP-LOCAL; the absolute conversion is applied by the caller
    gt = [{"frame": int(s["frame"]), "type": s["type"]} for s in doc["shots"]]
    return gt, fps


def match_and_score(pred_shots, gt_shots, tolerance):
    """Greedy nearest 1-1 pairing of predicted shots to GT shots within tolerance,
    then FH/BH metrics on the paired shots. pred_shots/gt_shots frames must be in
    the SAME coordinate system (both absolute)."""
    cand = []  # (dist, pred_i, gt_i)
    for pi, p in enumerate(pred_shots):
        for gi, g in enumerate(gt_shots):
            d = abs(p["frame"] - g["frame"])
            if d <= tolerance:
                cand.append((d, pi, gi))
    cand.sort(key=lambda c: (c[0], c[1], c[2]))

    used_p, used_g, pairs = set(), set(), []
    for d, pi, gi in cand:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, d))

    matched = len(pairs)
    missed = len(gt_shots) - matched          # GT with no prediction
    extra = len(pred_shots) - matched         # prediction with no GT

    correct = called = abstain = 0
    conf_FHtoBH = conf_BHtoFH = 0
    detail = []
    for pi, gi, d in sorted(pairs, key=lambda x: x[1]):
        pt = pred_shots[pi]["type"]
        gt = gt_shots[gi]["type"]
        if pt == UNK:
            abstain += 1
            verdict = "abstain"
        else:
            called += 1
            if pt == gt:
                correct += 1
                verdict = "ok"
            else:
                verdict = "WRONG"
                if gt == FH and pt == BH:
                    conf_FHtoBH += 1
                elif gt == BH and pt == FH:
                    conf_BHtoFH += 1
        detail.append({
            "gt_frame": gt_shots[gi]["frame"], "pred_frame": pred_shots[pi]["frame"],
            "dist": d, "gt": gt, "pred": pt, "verdict": verdict})

    acc = (correct / called) if called else 0.0
    abstain_rate = (abstain / matched) if matched else 0.0
    return {
        "matched": matched, "missed": missed, "extra": extra,
        "called": called, "correct": correct, "abstain": abstain,
        "accuracy_called": acc, "abstain_rate": abstain_rate,
        "conf_FHtoBH": conf_FHtoBH, "conf_BHtoFH": conf_BHtoFH,
        "detail": detail,
    }


def main():
    ap = argparse.ArgumentParser(description="Score forehand/backhand vs GT.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stats", help="LIVE _stats.json from run_pipeline_8s.py")
    src.add_argument("--pred", help="A --dump-shots predictions JSON")
    ap.add_argument("--truth", required=True, help="GT shots fixture JSON")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override fps (else read from the stats/pred or GT)")
    ap.add_argument("--tol", type=int, default=None,
                    help="Match tolerance in frames (default round(0.15*fps))")
    ap.add_argument("--json", action="store_true", help="Emit metrics as JSON")
    args = ap.parse_args()

    pred_shots, pfps, offset = _load_pred_shots(args.stats, args.pred)
    gt_local, gfps = _load_gt(args.truth)
    fps = args.fps or pfps or gfps or 50.0
    tol = args.tol if args.tol is not None else round(0.15 * fps)

    # GT is clip-local; convert to absolute with the stats' frame offset so both
    # sides match in absolute coordinates.
    gt_abs = [{"frame": g["frame"] + offset, "type": g["type"]} for g in gt_local]

    res = match_and_score(pred_shots, gt_abs, tol)

    if args.json:
        out = {k: v for k, v in res.items() if k != "detail"}
        print(json.dumps(out, indent=2))
        return

    print(f"FH/BH scorer  (fps={fps:g}, tol=±{tol}f, offset={offset})")
    print(f"  predicted shots: {len(pred_shots)} | GT shots: {len(gt_abs)}")
    print(f"  matched={res['matched']}  missed={res['missed']}  extra={res['extra']}")
    print(f"  called={res['called']}  correct={res['correct']}  abstain={res['abstain']}")
    print(f"  accuracy (called) = {res['correct']}/{res['called']} = "
          f"{res['accuracy_called']:.3f}")
    print(f"  accuracy (matched, abstain=wrong floor) = {res['correct']}/{res['matched']} = "
          f"{(res['correct']/res['matched']) if res['matched'] else 0:.3f}")
    print(f"  abstain rate = {res['abstain']}/{res['matched']} = {res['abstain_rate']:.3f}")
    print(f"  confusion  FH->BH = {res['conf_FHtoBH']}   BH->FH = {res['conf_BHtoFH']}")
    print("  per-shot:")
    for d in res["detail"]:
        flag = {"ok": "✅", "WRONG": "❌", "abstain": "·"}[d["verdict"]]
        print(f"    {flag} GT f{d['gt_frame']} {d['gt']:8s} -> pred f{d['pred_frame']} "
              f"{d['pred']:8s} (Δ{d['dist']})")


if __name__ == "__main__":
    sys.exit(main())
