"""
Replay the LIVE methodo FH/BH classification OFFLINE from a dumped inputs file.

run_pipeline_8s.py with COURTSIDE_DUMP_METHODO_INPUTS=PATH serializes the exact
classify-events inputs (the LOCKED tracks methodo_ppf + ball centers). This replays
the NEW methodo FH/BH path FAITHFULLY — a detect_hits pass on the locked tracks
(well-attributed near+far) → composite classify — and scores it vs the GT, so the
cache-vs-LIVE composite gap can be debugged without re-running the ~8min pipeline.

Usage:  python tools/event_eval/fhb_live_replay.py /tmp/s5_methodo_inputs.json
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.shots import (  # noqa: E402
    detect_hits, classify_forehand_backhand, estimate_handedness_prior)
from tools.event_eval.score_fhb import match_and_score  # noqa: E402

GT = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def main():
    dump_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/s5_methodo_inputs.json"
    d = json.loads(Path(dump_path).read_text())
    fps, fh, fw = d["fps"], d["height"], d["width"]
    start = d.get("start_frame", 0)
    ppf = d["methodo_ppf"]
    ball = [tuple(c) if c else None for c in d["all_ball_centers"]]

    # mirror prod: detect FH/BH contacts on the dense locked tracks, then composite
    contacts = detect_hits(ball, ppf, fps, fh, fw, bounce_frames=[],
                           wrist_prox_max=1.2, far_aware=True)
    handed = estimate_handedness_prior(contacts, ppf, fps, fh)
    print(f"handedness prior: {handed} | locked contacts={len(contacts)} "
          f"({sum(1 for c in contacts if c['player_side']=='far')} far)")

    pred = []
    for c in contacts:
        label = classify_forehand_backhand(
            c, ppf, players_per_frame_window=ppf, handedness=handed,
            fps=fps, frame_height=fh)
        pred.append({"frame": int(start + c["frame"]), "type": label,
                     "side": c["player_side"]})

    gt_local = json.loads(GT.read_text())["shots"]
    gt_abs = [{"frame": int(start + s["frame"]), "type": s["type"]}
              for s in gt_local]
    tol = round(0.15 * fps)
    res = match_and_score(pred, gt_abs, tol)
    print(f"[live replay] matched={res['matched']} missed={res['missed']} "
          f"extra={res['extra']} | acc(called)={res['correct']}/{res['called']}="
          f"{res['accuracy_called']:.3f} abstain={res['abstain']} "
          f"FH->BH={res['conf_FHtoBH']} BH->FH={res['conf_BHtoFH']}")
    for x in res["detail"]:
        flag = {"ok": "OK", "WRONG": "XX", "abstain": ".."}[x["verdict"]]
        print(f"   {flag} GT f{x['gt_frame']} {x['gt']:8s} -> f{x['pred_frame']} "
              f"{x['pred']:8s} (d{x['dist']})")


if __name__ == "__main__":
    main()
