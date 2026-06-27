"""Inspect the spurious-FP regions (head f60-90; tail f500-630) — every merged
event with features + final label, to understand the hit over-generation."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa
from vision.shots import detect_hits  # noqa
import vision.events as ev  # noqa
from vision.events import detect_turning_points, detect_sharp_turns  # noqa

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


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

    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    b_cands, _re, _nd = detect_bounces_robust(all_centers, speeds, fps, fh, fw)
    turns = sorted(set(detect_turning_points(all_centers, fps, fh))
                   | set(detect_sharp_turns(kal, kal_real, fps, fw, fh)))
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=[])

    orig = ev._global_alternation_decode
    cap = {}

    def spy(evs, step, keep, lam, miss_cost, anchor_bonus=3.0):
        cap["evs"] = evs
        return orig(evs, step, keep, lam, miss_cost, anchor_bonus)

    ev._global_alternation_decode = spy
    events, _ = ev.classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=all_centers,
        wasb_centers=wasb, wasb_is_real=wasb_real)
    ev._global_alternation_decode = orig
    final = {e["frame"]: e["label"] for e in events}

    def tag(f):
        gtb = min((abs(f - g) for g in gt_b), default=999)
        gth = min((abs(f - g) for g in gt_h), default=999)
        if gtb <= 8:
            return f"GTb(d{gtb})"
        if gth <= 8:
            return f"GTh(d{gth})"
        return "SPUR"

    for lo, hi in [(55, 95), (240, 300), (500, 630)]:
        print(f"\n=== merged events f{lo}..f{hi} ===")
        print(f"  {'frame':>5} {'src':<22} vy {'dmin':>5} h b t hl "
              f"{'bsc':>4} {'hsc':>4} {'anch':<6} -> {'label':<6} GT")
        for e in cap["evs"]:
            if not (lo <= e["frame"] <= hi):
                continue
            print(f"  f{e['frame']:>3} {str(e['src']):<22} "
                  f"{int(e['vy'])}  {e['dmin']:>5.2f} "
                  f"{int(e['hit'])} {int(e['bnc'])} {int(e['turn'])} {int(e['hit_like'])} "
                  f"{e['bscore']:>4.1f} {e['hscore']:>4.1f} "
                  f"{str(e['anchor']):<6} -> {str(final.get(e['frame'])):<6} {tag(e['frame'])}")


if __name__ == "__main__":
    main()
