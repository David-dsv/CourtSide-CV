"""Dump every classify_events candidate -> merged event with its full feature
record + final label, so we can see exactly why f174/f333/f525 land where they do.

Also probe detect_hits internals at f333 (why no hit candidate despite dense far
pose) by replicating its gates.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_robust  # noqa
from vision.shots import (  # noqa
    detect_hits, _wrist_speed_series, detect_handedness, _adaptive_threshold,
    _fwhm_at, _side_of, _player_anchor, _player_height)
from vision.events import (  # noqa
    detect_turning_points, detect_sharp_turns, classify_events)
import numpy as np  # noqa

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

    events, report = classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=all_centers,
        wasb_centers=wasb, wasb_is_real=wasb_real)

    print("=== FINAL EVENTS ===")
    for e in events:
        gtb = min((abs(e["frame"] - g) for g in gt_b), default=999)
        gth = min((abs(e["frame"] - g) for g in gt_h), default=999)
        tag = ""
        if gtb <= 8:
            tag = f"<-GTbounce(d{gtb})"
        elif gth <= 8:
            tag = f"<-GThit(d{gth})"
        else:
            tag = "<-SPURIOUS"
        print(f"  f{e['frame']:>3} {e['label']:<6} conf={e['confidence']:.2f} "
              f"why={e['why']:<12} {tag}")

    # ─── probe detect_hits gates at f333 ───
    print("\n=== detect_hits gate probe near f333 (far side) ===")
    net_y = fh * 0.47
    handed = detect_handedness(ppf, net_y)
    print(f"handed far={handed['far']} near={handed['near']}")
    speed_far, wxy_far = _wrist_speed_series(ppf, "far", handed["far"], net_y, fps)
    amp = _adaptive_threshold(None, fh, fps)
    fwhm_min = max(3, int(fps * 0.06))
    print(f"amp_floor={amp:.1f} fwhm_min={fwhm_min}")
    print("  f : speed_far  fwhm  is_localmax")
    for ff in range(322, 345):
        s = speed_far[ff]
        if np.isnan(s):
            print(f"  {ff}: nan")
            continue
        lm = (ff > 0 and ff < len(speed_far) - 1
              and not np.isnan(speed_far[ff - 1]) and not np.isnan(speed_far[ff + 1])
              and s >= speed_far[ff - 1] and s > speed_far[ff + 1])
        fw_ = _fwhm_at(speed_far, ff)
        flag = ""
        if lm and s >= amp:
            flag = f"LOCALMAX fwhm={fw_} {'PASS' if fw_ >= fwhm_min else 'FAIL-fwhm'}"
        print(f"  {ff}: {s:7.2f}  {fw_:>3}  {flag}")

    # which far hit cands did detect_hits actually emit?
    far_hits = [h for h in hits if h["player_side"] == "far"]
    print(f"\nfar hits emitted: {[h['frame'] for h in far_hits]}")
    print(f"all hit cands: {sorted(h['frame'] for h in hits)}")


if __name__ == "__main__":
    main()
