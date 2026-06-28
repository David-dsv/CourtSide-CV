"""S2 decoder probe — apples-to-apples global vs anchor on ONE dump.

Force-loads vision.events from a CHOSEN tree (worktree trap #2 fix): pass the
events.py path so we measure the code we just edited, not the MAIN tree's copy.

It runs classify_events twice (decoder="global" and decoder="anchor") on the same
canonical dump, prints the per-event internal state (anchor/vy/hit_like/label/why)
and the end-to-end confusion+F1 for each, so the residual @308 / @271 / @65 can be
attributed exactly.

Run:
  venv/bin/python tools/event_eval/decoder_probe.py \
      tests/fixtures/methodo/demo3_methodo_inputs_fullvideo_postS1.json \
      --events <worktree>/vision/events.py
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

_CAPTURE = {"evs": None}


def _install_capture(E):
    """Wrap both decoders to snapshot the pre-decode evs list (anchor/hit_like/
    vy/scores) without touching events.py — pure measurement hook."""
    g0, a0 = E._global_alternation_decode, E._anchor_parity_decode

    def gwrap(evs, *a, **k):
        _CAPTURE["evs"] = [dict(e) for e in evs]
        return g0(evs, *a, **k)

    def awrap(evs, *a, **k):
        _CAPTURE["evs"] = [dict(e) for e in evs]
        return a0(evs, *a, **k)

    E._global_alternation_decode = gwrap
    E._anchor_parity_decode = awrap


def load_events_module(events_path):
    """Force-load vision.events from an explicit path and register it in
    sys.modules BEFORE any other import resolves it (trap #2)."""
    if events_path is None:
        from vision import events as E
        return E
    p = Path(events_path).resolve()
    spec = importlib.util.spec_from_file_location("vision.events", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vision.events"] = mod
    spec.loader.exec_module(mod)
    return mod


def load(path):
    d = json.loads(Path(path).read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    is_real = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    hits = d["hits"]
    ppf = d["methodo_ppf"]
    return d, fps, fw, fh, raw, is_real, ac, hits, ppf


def run(E, decoder, path, verbose=False, focus=(), **kw):
    from vision.bounce import smooth_ball_trajectory, detect_bounces_robust
    from vision.shots import detect_hits
    from tools.event_eval.event_eval import match_events, scores
    d, fps, fw, fh, raw, is_real, ac, hits, ppf = load(path)
    b_cands, _re, _nd = detect_bounces_robust(ac, [0.0] * len(ac), fps, fh, fw)
    turns = E.detect_turning_points(ac, fps, fh)
    sharp = E.detect_sharp_turns(raw, is_real, fps, fw, fh)
    turns = sorted(set(turns) | set(sharp))
    events, _rep = E.classify_events(
        b_cands, hits, raw, is_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=ac,
        wasb_centers=None, wasb_is_real=None, decoder=decoder, **kw)
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    tol = round(0.15 * fps)
    res = match_events(ev, GT_B, GT_H, tol)
    s = scores(res)
    bev = sorted(e["frame"] for e in events if e["label"] == "BOUNCE")
    hev = sorted(e["frame"] for e in events if e["label"] == "HIT")
    print(f"\n══ decoder={decoder}  (events.py: {Path(E.__file__).resolve()}) ══")
    print(f"  pred BOUNCE: {bev}")
    print(f"  pred HIT   : {hev}")
    print(f"  confusion H->B={res['conf_HtoB']} B->H={res['conf_BtoH']} | "
          f"bounce F1={s['bounce']['F1']:.3f} (R={s['bounce']['R']:.2f} P={s['bounce']['P']:.2f}) | "
          f"hit F1={s['hit']['F1']:.3f} (R={s['hit']['R']:.2f} P={s['hit']['P']:.2f})")
    print(f"  bounce misses: {res['FN_b']}  hit misses: {res['FN_h']}")
    print(f"  spurious: {res['spurious']}")
    if focus == ("all",) and _CAPTURE.get("evs") is not None:
        print("  ── ALL internal evs ──")
        for ev_ in _CAPTURE["evs"]:
            a = ev_["anchor"]
            mand = "MAND" if ev_["vy"] else "    "
            print(f"    f{ev_['frame']:>3} {mand} src={str(ev_['src']):<22} "
                  f"anchor={str(a):<6} hit_like={int(ev_['hit_like'])} "
                  f"vy={int(ev_['vy'])} h={int(ev_['hit'])} b={int(ev_['bnc'])} t={int(ev_['turn'])} "
                  f"dmin={ev_['dmin']:.2f} bs={ev_['bscore']:.2f} hs={ev_['hscore']:.2f} "
                  f"vxf={ev_['features'].get('vx_flip')} -> {ev_['label']}")
    elif focus and _CAPTURE.get("evs") is not None:
        print("  ── internal evs near focus frames ──")
        for ev_ in _CAPTURE["evs"]:
            if any(abs(ev_["frame"] - f) <= 10 for f in focus):
                print(f"    f{ev_['frame']:>3} src={ev_['src']} anchor={ev_['anchor']} "
                      f"vy={ev_['vy']} hit={ev_['hit']} bnc={ev_['bnc']} turn={ev_['turn']} "
                      f"hit_like={ev_['hit_like']} dmin={ev_['dmin']:.2f} "
                      f"bscore={ev_['bscore']:.2f} hscore={ev_['hscore']:.2f} "
                      f"vx_flip={ev_['features'].get('vx_flip')} -> label={ev_['label']}")
    # which pairs are the confusions
    for p in res["pairs"]:
        if (p["gt_class"] == "shot" and p["pred_label"] == "BOUNCE"):
            print(f"  >> conf_H->B: pred BOUNCE@{p['pred_frame']} stole GT_H@{p['gt_frame']}")
        if (p["gt_class"] == "bounce" and p["pred_label"] == "HIT"):
            print(f"  >> conf_B->H: pred HIT@{p['pred_frame']} stole GT_B@{p['gt_frame']}")
    return res, s, events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    ap.add_argument("--events", default=None, help="path to events.py to force-load")
    ap.add_argument("--decoders", default="global,anchor")
    ap.add_argument("--focus", default="62,268,308,174")
    args = ap.parse_args()
    E = load_events_module(args.events)
    _install_capture(E)
    # sanity: prove which module + which symbols are present
    print(f"LOADED events from: {Path(E.__file__).resolve()}")
    if args.focus == "all":
        focus = ("all",)
    else:
        focus = tuple(int(x) for x in args.focus.split(",")) if args.focus else ()
    for d in args.decoders.split(","):
        run(E, d.strip(), args.inputs, focus=focus)


if __name__ == "__main__":
    main()
