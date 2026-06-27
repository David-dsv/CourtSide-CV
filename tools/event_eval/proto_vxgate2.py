"""
PROTOTYPE 2 — gate the vy BOUNCE anchor by the REPRESENTATIVE frame's vx_flip.

Prototype 1 patched _direction_features per-frame, but the cluster's vy is read
on a MEMBER frame (e.g. 79, vx=None) while the committed REP frame (80) carries
vx_flip=True. So the gate must apply on the rep feature. Here we reimplement the
PASS-0/1 merge inline (faithful copy of vision.events.classify_events) with ONE
change: a vy-flip whose REP-frame vx_flip is True (vertical AND horizontal
reversal = redirect = HIT) is NOT treated as a pure floor-bounce anchor — vy is
cleared for that event so the firewall + alternation arbitrate it normally.

Scores LIVE + CACHE, OFF vs ON. Must: LIVE H->B 4->0, CACHE stays 0/0 F1 0.842.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import vision.events as V  # noqa: E402
from vision.events import (BOUNCE, HIT, _nearest_player_dist_norm,  # noqa: E402
                           _global_alternation_decode, _wasb_consistency)
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]


def classify_vxgate(bounce_cands, hit_cands, raw_centers, is_real,
                    players_per_frame, fps, frame_width, frame_height, *,
                    vx_gate=True,
                    half_frac=0.10, gap_frac=0.05, eps_rel=0.15,
                    eps_floor_frac=0.0015, min_netdx_frac=0.004,
                    near_wrist=0.5, far_wrist=1.2, merge_frac=0.12,
                    step0_frac=0.55, keep=0.3, lam=0.5, miss_cost=1.0,
                    turning_frames=None, smoothed_centers=None,
                    wasb_centers=None, wasb_is_real=None):
    merge_gap = max(1, round(fps * merge_frac))
    near, far = near_wrist, far_wrist
    step0 = max(1.0, step0_frac * fps)

    def _xy_at(fr):
        if smoothed_centers is not None and 0 <= fr < len(smoothed_centers) \
                and smoothed_centers[fr] is not None:
            return int(smoothed_centers[fr][0]), int(smoothed_centers[fr][1])
        return None

    def _feat(fr):
        return V._direction_features(
            fr, raw_centers, is_real, fps, frame_width, half_frac=half_frac,
            gap_frac=gap_frac, eps_rel=eps_rel, eps_floor_frac=eps_floor_frac,
            min_netdx_frac=min_netdx_frac)

    cand = []
    for (fr, x, y) in bounce_cands:
        cand.append((int(fr), "bounce", (int(x), int(y)), None))
    for fr in (turning_frames or []):
        xy = _xy_at(int(fr))
        if xy is not None:
            cand.append((int(fr), "turn", xy, None))
    for h in hit_cands:
        cand.append((int(h["frame"]), "hit", (int(h["x"]), int(h["y"])), h))
    cand.sort(key=lambda c: c[0])

    clusters = []
    cur = None
    for fr, src, xy, hm in cand:
        if cur is None or (fr - cur["start"]) > merge_gap:
            cur = {"start": fr, "members": []}
            clusters.append(cur)
        cur["members"].append((fr, src, xy, hm))

    evs = []
    for cl in clusters:
        members = cl["members"]
        hit = any(s == "hit" for _, s, _, _ in members)
        bnc = any(s == "bounce" for _, s, _, _ in members)
        turn = any(s == "turn" for _, s, _, _ in members)
        mfeat = {fr: _feat(fr) for fr, _, _, _ in members}
        vy = any(mfeat[fr]["vy_flip"] for fr, _, _, _ in members)
        dmins = [_nearest_player_dist_norm(fr, xy, players_per_frame)
                 for fr, _, xy, _ in members]
        dmins = [dd for dd in dmins if dd < 90]
        dmin = min(dmins) if dmins else float("inf")
        rep = None
        for fr, s, xy, hm in members:
            if mfeat[fr]["vy_flip"]:
                rep = (fr, xy, hm)
                break
        if rep is None:
            for fr, s, xy, hm in members:
                dn = _nearest_player_dist_norm(fr, xy, players_per_frame)
                if s == "hit" and dn < near:
                    rep = (fr, xy, hm)
                    break
        if rep is None:
            best = min(members, key=lambda m: _nearest_player_dist_norm(
                m[0], m[2], players_per_frame))
            rep = (best[0], best[2], best[3])
        rep_fr, rep_xy, rep_hm = rep
        feat = dict(mfeat[rep_fr])
        feat["dist_norm"] = dmin if dmin != float("inf") else 99.0

        # ── THE GATE: a vertical reversal that ALSO reverses horizontal (vx_flip
        # True at the rep frame) is a racket REDIRECT, not a floor bounce. Clear vy
        # for this event so the firewall + alternation arbitrate it (it will be
        # caught as hit_like via proximity, or alternation-placed as a HIT). vx_flip
        # False (preserved) or None (unreadable) keeps vy = a true floor bounce.
        vy_is_bounce = vy
        _vxf = feat.get("vx_flip")
        if vx_gate and vy and _vxf is not None and bool(_vxf):
            vy_is_bounce = False

        hit_like = (hit or dmin < near) and not vy_is_bounce
        bscore = (1.0 * vy_is_bounce + 0.6 * bnc + 0.6 * (turn and dmin >= near))
        hscore = (1.0 * hit + 1.0 * (dmin < near) + 0.4 * (near <= dmin < far))
        anchor = None
        if vy_is_bounce:
            anchor = BOUNCE
        elif hit and dmin < near:
            anchor = HIT
        evs.append({
            "frame": rep_fr, "x": rep_xy[0], "y": rep_xy[1],
            "src": sorted(set(s for _, s, _, _ in members)),
            "hit_meta": rep_hm, "features": feat,
            "hit": hit, "bnc": bnc, "turn": turn, "vy": vy_is_bounce, "dmin": dmin,
            "hit_like": hit_like, "bscore": bscore, "hscore": hscore,
            "anchor": anchor, "label": None,
        })
    evs.sort(key=lambda e: e["frame"])

    anchors = [e for e in evs if e["anchor"] is not None]
    gaps = [anchors[i]["frame"] - anchors[i - 1]["frame"]
            for i in range(1, len(anchors))]
    units = [g / max(1, round(g / step0)) for g in gaps if g > 0]
    units = [u for u in units if u > merge_gap]
    step = float(np.median(units)) if units else step0
    step = max(step, 1.0)

    _global_alternation_decode(evs, step, keep, lam, miss_cost)
    kept = [e for e in evs if e["label"] is not None]
    for e in kept:
        e["confidence"] = min(1.0, (e["bscore"] if e["label"] == BOUNCE
                                    else e["hscore"]) / 1.5)
        if "why" not in e:
            e["why"] = ("vy_flip" if e["anchor"] == BOUNCE
                        else ("hit@wrist" if e["anchor"] == HIT else "parity_fill"))
    events = [{"frame": e["frame"], "x": e["x"], "y": e["y"],
               "label": e["label"], "why": e.get("why")} for e in kept]
    return events


def score(events, tol):
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    res = match_events(ev, GT_B, GT_H, tol)
    return res, scores(res)


def load_live():
    d = json.loads(Path("/tmp/methodoprod/live_inputs.json").read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    isr = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    bc = [(int(f), int(x), int(y)) for f, x, y in d["b_cands"]]
    turns = sorted(int(t) for t in d["turns"])
    return dict(bc=bc, hits=d["hits"], raw=raw, isr=isr, ppf=d["methodo_ppf"],
                fps=fps, fw=fw, fh=fh, ac=ac, turns=turns)


def load_cache():
    from vision.bounce import smooth_ball_trajectory, detect_bounces_robust
    from vision.shots import detect_hits
    from vision.events import detect_turning_points, detect_sharp_turns
    c = json.loads((ROOT / "tests/fixtures/cache/demo3_event.json").read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kr = c["kalman_is_real"]
    ppf = c["players_per_frame"]
    ac = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    bc, _, _ = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = set(detect_turning_points(ac, fps, fh)) | set(
        detect_sharp_turns(kal, kr, fps, fw, fh))
    hits = detect_hits(ac, ppf, fps, fh, fw, bounce_frames=[])
    return dict(bc=bc, hits=hits, raw=kal, isr=kr, ppf=ppf, fps=fps, fw=fw,
                fh=fh, ac=ac, turns=sorted(turns))


def run(data, vx_gate):
    events = classify_vxgate(
        data["bc"], data["hits"], data["raw"], data["isr"], data["ppf"],
        data["fps"], data["fw"], data["fh"], vx_gate=vx_gate,
        turning_frames=data["turns"], smoothed_centers=data["ac"])
    return score(events, round(0.15 * data["fps"])), events


def line(name, sc):
    res, s = sc
    print(f"  {name:18}: H->B={res['conf_HtoB']} B->H={res['conf_BtoH']}  "
          f"bounceF1={s['bounce']['F1']:.3f} hitF1={s['hit']['F1']:.3f}  "
          f"(b R={s['bounce']['R']:.2f} P={s['bounce']['P']:.2f})")


if __name__ == "__main__":
    live = load_live()
    cache = load_cache()
    print("LIVE inputs (prod ball track):")
    (rOff, _), evOff = run(live, False)
    (rOn, _), evOn = run(live, True)
    line("OFF (current)", (rOff, _))
    line("ON  (vx-gate)", (rOn, _))
    print("    ON bounces:", sorted(e["frame"] for e in evOn if e["label"] == "BOUNCE"))
    print("    ON hits   :", sorted(e["frame"] for e in evOn if e["label"] == "HIT"))
    print("\nCACHE (frozen demo3):")
    line("OFF (current)", run(cache, False)[0])
    line("ON  (vx-gate)", run(cache, True)[0])
