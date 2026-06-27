"""Deep diagnostic of the demo3 event misses — answer the structural questions:

  - Are the missed frames (bounce 174; hits 333, 525) IN the candidate pool?
    (a generation gap) or in the pool but mis-labeled / dropped (classification)?
  - What features does each candidate near a GT frame carry (vy, dmin, hit/bnc/turn,
    hit_like, anchor)?  -> tells us what lever recovers it.
  - How dense is the FAR-player pose near the missed FAR hits (f333/f525)?

Pure-cache, no video decode. Run with PYTHONPATH=<worktree>.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import (  # noqa: E402
    smooth_ball_trajectory, detect_bounces_robust)
from vision.shots import detect_hits  # noqa: E402
from vision.events import (  # noqa: E402
    detect_turning_points, detect_sharp_turns, _direction_features,
    _nearest_player_dist_norm)

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]

    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    b_cands, _re, _nd = detect_bounces_robust(all_centers, speeds, fps, fh, fw)
    b_frames = sorted(b[0] for b in b_cands)
    turns = detect_turning_points(all_centers, fps, fh)
    sharp = detect_sharp_turns(kal, kal_real, fps, fw, fh)
    turn_pool = sorted(set(turns) | set(sharp))
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=[])
    hit_frames = sorted(h["frame"] for h in hits)

    print(f"=== POOLS @ {fps}fps {fw}x{fh} ===")
    print(f"bounce_cands (detect_bounces_robust): {b_frames}")
    print(f"turn pool (turning_points U sharp_turns): {turn_pool}")
    print(f"  turning_points: {sorted(turns)}")
    print(f"  sharp_turns:    {sorted(sharp)}")
    print(f"hit_cands (detect_hits): {hit_frames}")
    print()

    def near_pool(f, pool, tol=8):
        return [p for p in pool if abs(p - f) <= tol]

    print("=== GT BOUNCE coverage in pools (tol 8) ===")
    for f in sorted(gt_b):
        inb = near_pool(f, b_frames)
        intr = near_pool(f, turn_pool)
        print(f"  b@{f}: bounce_cand={inb} turn={intr}")
    print()
    print("=== GT HIT coverage in pools (tol 8) ===")
    for f in sorted(gt_h):
        inh = near_pool(f, hit_frames)
        intr = near_pool(f, turn_pool)
        print(f"  h@{f}: hit_cand={inh} turn={intr}")
    print()

    # far-pose density near the missed far hits
    net_y = fh * 0.47
    def far_present(f):
        pl = ppf[f] if 0 <= f < len(ppf) else []
        out = []
        for p in (pl or []):
            if p is None:
                continue
            box = p.get("box")
            if box is None or len(box) < 4:
                continue
            side = "near" if box[3] >= net_y else "far"
            kc = p.get("kps_conf")
            nkp = sum(1 for v in (kc or []) if v and v >= 0.30)
            # wrist conf
            wl = (kc[9] if kc and len(kc) > 9 else 0) or 0
            wr = (kc[10] if kc and len(kc) > 10 else 0) or 0
            out.append((side, round(box[3] - box[1]), nkp, round(wl, 2), round(wr, 2)))
        return out

    print("=== FAR-pose density window around missed hits/bounce ===")
    for f in (174, 333, 525):
        print(f"  frame window {f}-4..{f}+4:")
        for ff in range(f - 4, f + 5):
            print(f"    f{ff}: {far_present(ff)}")
        print()

    # direction features at the missed frames + neighbours
    print("=== direction features at missed frames (best in tol window) ===")
    for label, f in [("bounce", 174), ("hit", 333), ("hit", 525)]:
        print(f"  {label}@{f}:")
        for ff in range(f - 6, f + 7):
            feat = _direction_features(ff, kal, kal_real, fps, fw)
            dn = _nearest_player_dist_norm(ff, all_centers[ff] if all_centers[ff] else (0, 0), ppf) if all_centers[ff] else float('inf')
            if feat["vx_flip"] is not None or feat["vy_flip"] or (dn < 90):
                print(f"    f{ff}: vx_flip={feat['vx_flip']} vy_flip={feat['vy_flip']} "
                      f"dmin={dn:.2f} ss={feat['speed_scale']:.1f}")
        print()


if __name__ == "__main__":
    main()
