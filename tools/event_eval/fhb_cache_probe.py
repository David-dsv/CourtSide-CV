"""
FAST offline FH/BH probe on the committed demo3 pose cache.

Replays detect_hits + classify_forehand_backhand on tests/fixtures/cache/
demo3_pose.json (locked poses, ball centers) and scores FH/BH vs the GT fixture
with the SAME matcher as tools/event_eval/score_fhb.py. No GPU / no video decode —
instant, so the classifier logic can be iterated here, then VALIDATED on a LIVE
_stats.json (the cache can pass while prod diverges — always confirm live).

Usage:  python tools/event_eval/fhb_cache_probe.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.shots import detect_hits, classify_forehand_backhand  # noqa: E402
from tools.event_eval.score_fhb import match_and_score  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_pose.json"
GT = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def run():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    ball = [tuple(x) if x else None for x in c["ball_centers_smoothed"]]
    ppf = c["players_per_frame"]
    gt = json.loads(GT.read_text())["shots"]
    gt_shots = [{"frame": int(s["frame"]), "type": s["type"]} for s in gt]

    # Mirror the LEGACY prod call: locked poses, wrist-prox gate ON.
    hits = detect_hits(ball, ppf, fps, fh, fw, bounce_frames=[], wrist_prox_max=1.2)
    pred = []
    for h in hits:
        fhb = classify_forehand_backhand(h, ppf)
        pred.append({"frame": int(h["frame"]), "type": fhb,
                     "side": h["player_side"]})

    tol = round(0.15 * fps)
    res = match_and_score(pred, gt_shots, tol)
    return pred, gt_shots, res, tol


def main():
    pred, gt_shots, res, tol = run()
    print(f"[cache probe] hits={len(pred)} GT={len(gt_shots)} tol=±{tol}f")
    print(f"  matched={res['matched']} missed={res['missed']} extra={res['extra']}")
    print(f"  accuracy(called) = {res['correct']}/{res['called']} = "
          f"{res['accuracy_called']:.3f} | abstain={res['abstain']} "
          f"| FH->BH={res['conf_FHtoBH']} BH->FH={res['conf_BHtoFH']}")
    for d in res["detail"]:
        flag = {"ok": "OK", "WRONG": "XX", "abstain": ".."}[d["verdict"]]
        print(f"   {flag} GT f{d['gt_frame']:>4} {d['gt']:8s} -> pred f{d['pred_frame']:>4} "
              f"{d['pred']:8s} (d{d['dist']})")
    # show extras (predicted hits with no GT) for diagnostics
    matched_pf = {d["pred_frame"] for d in res["detail"]}
    extras = [p for p in pred if p["frame"] not in matched_pf]
    if extras:
        print("  EXTRA (no GT):", [(p["frame"], p["type"], p["side"]) for p in extras])


if __name__ == "__main__":
    main()
