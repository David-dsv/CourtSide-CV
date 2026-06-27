"""Diagnostic: WHY the legacy path misses bounces / over-fires hits on demo3.

Prints, for each GT bounce: is it in the smoothed trajectory? does it have a
y-extremum? what is its velocity-vector turn angle on the RAW pre-spline track?
And for each detected hit: distance ball<->nearest-wrist at contact.
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory
from vision.shots import detect_hits, _wrist_of, _side_of, _player_anchor, _player_height

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_event.json"
GT_B = ROOT / "tests" / "fixtures" / "bounces" / "tennis_demo3.bounces.json"
GT_S = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def turn_angle(raw, is_real, f, k):
    n = len(raw)
    if f - k < 0 or f + k >= n:
        return None
    if not (is_real[f - k] and is_real[f] and is_real[f + k]):
        return None
    c0, c1, c2 = raw[f - k], raw[f], raw[f + k]
    if c0 is None or c1 is None or c2 is None:
        return None
    ax, ay = c1[0] - c0[0], c1[1] - c0[1]
    bx, by = c2[0] - c1[0], c2[1] - c1[1]
    na = (ax*ax + ay*ay) ** .5; nb = (bx*bx + by*by) ** .5
    if na < 1 or nb < 1:
        return None
    cos = max(-1, min(1, (ax*bx + ay*by) / (na*nb)))
    vx_sign = (np.sign(ax), np.sign(bx))
    vy_sign = (np.sign(ay), np.sign(by))
    return np.degrees(np.arccos(cos)), vx_sign, vy_sign


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    ppf = c["players_per_frame"]
    gt_b = [int(b["frame"]) for b in json.loads(GT_B.read_text())["bounces"]]
    gt_h = [int(s["frame"]) for s in json.loads(GT_S.read_text())["shots"]]
    net_y = fh * 0.47

    sm = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    ys = np.array([p[1] if p else np.nan for p in sm])
    k = max(2, round(fps * 0.06))

    print("=== GT BOUNCES: trajectory presence + y-extremum + turn angle (raw) ===")
    for gt in gt_b:
        # tracked nearby?
        win = [i for i in range(gt-8, gt+9) if 0 <= i < len(sm) and sm[i]]
        present = len(win)
        # local y-extremum (is it a local max of y = lowest screen point)?
        lo, hi = max(0, gt-4), min(len(ys), gt+5)
        loc = ys[lo:hi]; loc = loc[~np.isnan(loc)]
        is_ymax = (not np.isnan(ys[gt])) and len(loc) and ys[gt] >= np.nanmax(loc) - 0.5
        # raw turn angle (try ±8 window for the best)
        best = None
        for ff in range(gt-6, gt+7):
            ta = turn_angle(kal, kal_real, ff, k)
            if ta and (best is None or ta[0] > best[1]):
                best = (ff, ta[0], ta[1], ta[2])
        ta_s = (f"angle={best[1]:.0f}deg@f{best[0]} vx{best[2]} vy{best[3]}"
                if best else "no-raw-turn")
        print(f"  GT b{gt}: tracked {present}/17, y-extremum={is_ymax}, {ta_s}")

    b_cands = detect_bounces_from_trajectory(sm, [0.0]*len(sm), fps, fh, fw)
    print(f"\n  legacy bounce preds: {[b[0] for b in b_cands]}")

    print("\n=== DETECTED HITS: ball<->wrist proximity at contact ===")
    hits = detect_hits(sm, ppf, fps, fh, fw, bounce_frames=[b[0] for b in b_cands])
    for h in hits:
        f = h["frame"]
        side = h["player_side"]
        ball = sm[f] if f < len(sm) and sm[f] else None
        # find the on-side player + their racket wrist
        wd = None
        players = ppf[f] if f < len(ppf) else None
        if players and ball:
            me = None
            for p in players:
                if _side_of(p, net_y) == side:
                    if me is None or _player_height(p) > _player_height(me):
                        me = p
            if me is not None:
                w = _wrist_of(me, h.get("right_handed", True))
                hh = max(_player_height(me), 1.0)
                if w is not None:
                    wd = float(np.hypot(ball[0]-w[0], ball[1]-w[1])) / hh
        near_gt = any(abs(f - g) <= 8 for g in gt_h)
        print(f"  HIT f{f} side={side} wspeed={h['wrist_speed']:.0f} "
              f"ball<->wrist/h={wd if wd is None else round(wd,2)} "
              f"{'<-NEAR GT shot' if near_gt else '<<< SPURIOUS (no GT shot)'}")


if __name__ == "__main__":
    main()
