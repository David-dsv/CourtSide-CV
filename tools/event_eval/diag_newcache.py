"""One-shot re-baseline on whatever cache is committed: far coverage, the
far-wrist generator's output near f333/f525, the dmin at f174 with the (new)
far-pose, and the full confusion+F1 (far_wrist ON vs OFF)."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa
from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa
from vision.shots import detect_hits  # noqa
from vision.events import (  # noqa
    classify_events, detect_turning_points, detect_sharp_turns,
    detect_far_wrist_hits, _nearest_player_dist_norm)
from tools.event_eval.event_eval import match_events, scores  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def run(far_on, kal, kal_real, wasb, wasb_real, ppf, fps, fw, fh, gt_b, gt_h):
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    b, _r, _n = detect_bounces_robust(sm, [0.0] * len(sm), fps, fh, fw)
    turns = sorted(set(detect_turning_points(sm, fps, fh))
                   | set(detect_sharp_turns(kal, kal_real, fps, fw, fh)))
    hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[])
    ev, rep = classify_events(b, hits, kal, kal_real, ppf, fps, fw, fh,
                              turning_frames=turns, smoothed_centers=sm,
                              wasb_centers=wasb, wasb_is_real=wasb_real,
                              far_wrist_hits=far_on)
    res = match_events([{"frame": e["frame"], "label": e["label"]} for e in ev],
                       gt_b, gt_h, round(0.15 * fps))
    return res, scores(res), ev


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wasb_real = c["wasb_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))

    net = fh * 0.47
    far = sum(1 for fr in ppf for p in (fr or [])
              if p and len(p.get("box", [])) >= 4 and p["box"][3] < net)
    print(f"FAR coverage: {far}/{len(ppf)} = {100*far/len(ppf):.1f}%")

    fwh = detect_far_wrist_hits(ppf, sm, fps, fw, fh)
    print(f"detect_far_wrist_hits -> {[h['frame'] for h in fwh]}")
    for tgt in (333, 525):
        near = [h['frame'] for h in fwh if abs(h['frame'] - tgt) <= 8]
        print(f"  near GT hit {tgt}: {near}")

    print("\ndmin at f174 region (new far-pose):")
    for ff in range(170, 182):
        if sm[ff] is None:
            continue
        dn = _nearest_player_dist_norm(ff, sm[ff], ppf)
        print(f"  f{ff}: ball={tuple(int(v) for v in sm[ff])} dmin={dn:.2f}")

    for far_on in (False, True):
        res, s, ev = run(far_on, kal, kal_real, wasb, wasb_real, ppf, fps, fw, fh, gt_b, gt_h)
        print(f"\n=== far_wrist_hits={far_on} ===")
        print(f"  H->B={res['conf_HtoB']} B->H={res['conf_BtoH']} | "
              f"bounce F1={s['bounce']['F1']:.3f} (R={s['bounce']['R']:.2f}) | "
              f"hit F1={s['hit']['F1']:.3f} (R={s['hit']['R']:.2f})")
        print(f"  bounce misses: {res['FN_b']}  hit misses: {res['FN_h']}")
        print(f"  spurious: {res['spurious']}")


if __name__ == "__main__":
    main()
