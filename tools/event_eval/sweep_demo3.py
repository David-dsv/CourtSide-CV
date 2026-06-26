"""Parameter sweep on demo3: find configs maximizing (bounce F1 + hit F1) subject
to the HARD constraint confusion_H->B == 0 (and prefer confusion_B->H == 0).

Reports the best configs + the invariance check (how many configs hold 0/0).
"""
import sys
import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa
from vision.shots import detect_hits  # noqa
from vision.events import classify_events, detect_turning_points  # noqa
from tools.event_eval.event_eval import match_events, scores  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kr = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wr = c["wasb_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    allc = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    b = detect_bounces_robust(allc, [0.0] * len(allc), fps, fh, fw)[0]
    turns = detect_turning_points(allc, fps, fh)
    hits = detect_hits(allc, ppf, fps, fh, fw, bounce_frames=[])
    tol = round(0.15 * fps)

    grid = {
        "decoder": ["segment", "global"],
        "near_wrist": [0.45, 0.5, 0.55],
        "far_wrist": [1.0, 1.2],
        "keep": [0.0, 0.1, 0.3],
        "miss_cost": [1.0, 1.2, 1.5],
        "step0_frac": [0.45, 0.55, 0.65],
        "slot_rad": [0.4, 0.6],
    }
    keys = list(grid)
    best = []
    held = 0
    total = 0
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        ev, rep = classify_events(b, hits, kal, kr, ppf, fps, fw, fh,
                                  turning_frames=turns, smoothed_centers=allc,
                                  wasb_centers=wasb, wasb_is_real=wr, **kw)
        evs = [{"frame": e["frame"], "label": e["label"]} for e in ev]
        res = match_events(evs, gt_b, gt_h, tol)
        total += 1
        if res["conf_HtoB"] == 0 and res["conf_BtoH"] == 0:
            held += 1
        s = scores(res)
        score = s["bounce"]["F1"] + s["hit"]["F1"]
        # HARD constraint: H->B must be 0
        feasible = (res["conf_HtoB"] == 0)
        best.append((feasible, res["conf_HtoB"], res["conf_BtoH"],
                     round(s["bounce"]["F1"], 3), round(s["hit"]["F1"], 3),
                     round(score, 3), round(rep["consistency"], 2), kw))

    # rank: feasible first, then H->B asc, B->H asc, sum F1 desc
    best.sort(key=lambda r: (not r[0], r[1], r[2], -r[5]))
    print(f"swept {total} configs | held 0/0 confusion: {held}/{total} "
          f"({100*held/total:.0f}%)")
    print("\nTOP 12 (feasible H->B==0, ranked by B->H then sum-F1):")
    print(f"{'H->B':>4} {'B->H':>4} {'bF1':>5} {'hF1':>5} {'sum':>5} {'cons':>4}  config")
    for fea, hb, bh, bf1, hf1, sm, cons, kw in best[:12]:
        cfg = " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"{hb:>4} {bh:>4} {bf1:>5} {hf1:>5} {sm:>5} {cons:>4}  {cfg}")


if __name__ == "__main__":
    main()
