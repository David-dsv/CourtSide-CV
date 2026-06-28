"""S2 dual thermometer — score events.py on BOTH the committed cache AND the
canonical LIVE post-S1 dump in one shot, with explicit module force-loading
(worktree trap #2). Prints a compact machine-readable summary per source.

Both are scored with the SAME pipeline calls used by the regression test
(cache) and live_pool (live), so the numbers match those gates exactly.

Run:
  venv/bin/python tools/event_eval/dual_score.py --events <wt>/vision/events.py \
      [--decoder global|anchor] [--kw merge_frac=0.18,near_wrist=0.5]
"""
import sys
import json
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]

LIVE_DUMP = ROOT / "tests/fixtures/methodo/demo3_methodo_inputs_fullvideo_postS1.json"
CACHE = ROOT / "tests/fixtures/cache/demo3_event.json"


def load_events(path):
    if path is None:
        from vision import events as E
        return E
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("vision.events", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vision.events"] = mod
    spec.loader.exec_module(mod)
    return mod


def _conf_pairs(res):
    htb = btoh = None
    for p in res["pairs"]:
        if p["gt_class"] == "shot" and p["pred_label"] == "BOUNCE":
            htb = (p["pred_frame"], p["gt_frame"])
        if p["gt_class"] == "bounce" and p["pred_label"] == "HIT":
            btoh = (p["pred_frame"], p["gt_frame"])
    return htb, btoh


def score_cache(E, decoder, kw):
    from vision.bounce import smooth_ball_trajectory, detect_bounces_robust
    from vision.shots import detect_hits
    from tools.event_eval.event_eval import match_events, scores
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wasb_real = c["wasb_is_real"]
    ppf = c["players_per_frame"]
    ac = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    b_cands, _, _ = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = sorted(set(E.detect_turning_points(ac, fps, fh))
                   | set(E.detect_sharp_turns(kal, kal_real, fps, fw, fh)))
    hits = detect_hits(ac, ppf, fps, fh, fw, bounce_frames=[])
    events, _ = E.classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=ac,
        wasb_centers=wasb, wasb_is_real=wasb_real, decoder=decoder, **kw)
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    res = match_events(ev, GT_B, GT_H, round(0.15 * fps))
    return res, scores(res), events


def score_live(E, decoder, kw):
    from vision.bounce import detect_bounces_robust
    from tools.event_eval.event_eval import match_events, scores
    d = json.loads(LIVE_DUMP.read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    is_real = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    hits = d["hits"]
    ppf = d["methodo_ppf"]
    b_cands, _, _ = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = sorted(set(E.detect_turning_points(ac, fps, fh))
                   | set(E.detect_sharp_turns(raw, is_real, fps, fw, fh)))
    events, _ = E.classify_events(
        b_cands, hits, raw, is_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=ac,
        wasb_centers=None, wasb_is_real=None, decoder=decoder, **kw)
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    res = match_events(ev, GT_B, GT_H, round(0.15 * fps))
    return res, scores(res), events


def _line(tag, res, s, events):
    htb, btoh = _conf_pairs(res)
    bev = sorted(e["frame"] for e in events if e["label"] == "BOUNCE")
    hev = sorted(e["frame"] for e in events if e["label"] == "HIT")
    print(f"[{tag}] H->B={res['conf_HtoB']} B->H={res['conf_BtoH']} "
          f"bF1={s['bounce']['F1']:.3f} hF1={s['hit']['F1']:.3f} "
          f"(bR={s['bounce']['R']:.2f} hR={s['hit']['R']:.2f}) "
          f"spur={len(res['spurious'])} FNb={res['FN_b']} FNh={res['FN_h']}")
    if htb:
        print(f"      conf_H->B: pred BOUNCE@{htb[0]} stole GT_H@{htb[1]}")
    if btoh:
        print(f"      conf_B->H: pred HIT@{btoh[0]} stole GT_B@{btoh[1]}")
    print(f"      B={bev}")
    print(f"      H={hev}")


def parse_kw(s):
    if not s:
        return {}
    out = {}
    for tok in s.split(","):
        k, v = tok.split("=")
        try:
            out[k] = float(v) if ("." in v or "e" in v) else int(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=None)
    ap.add_argument("--decoder", default="global")
    ap.add_argument("--kw", default="")
    ap.add_argument("--only", default="both", choices=["both", "cache", "live"])
    args = ap.parse_args()
    E = load_events(args.events)
    kw = parse_kw(args.kw)
    print(f"events: {Path(E.__file__).resolve()}  decoder={args.decoder}  kw={kw}")
    cache_pass = None
    if args.only in ("both", "cache"):
        r, s, ev = score_cache(E, args.decoder, kw)
        _line("CACHE", r, s, ev)
        cache_pass = (r["conf_HtoB"] == 0 and r["conf_BtoH"] == 0
                      and s["bounce"]["F1"] >= 0.80 and s["hit"]["F1"] >= 0.60)
        print(f"      CACHE GATE: {'PASS' if cache_pass else 'FAIL'} "
              f"(need H->B=0,B->H=0,bF1>=0.80,hF1>=0.60)")
    if args.only in ("both", "live"):
        r, s, ev = score_live(E, args.decoder, kw)
        _line("LIVE ", r, s, ev)
    if cache_pass is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
