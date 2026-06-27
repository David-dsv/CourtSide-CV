"""
PROTOTYPE — test the hypothesis that gating the vy-flip BOUNCE anchor by vx
(a vy-flip that ALSO inverts vx is a redirect=HIT, not a floor bounce) fixes the
live prod confusion WITHOUT breaking the demo3 cache or felix.

We monkeypatch vision.events.classify_events's notion of "vy" so that a vertical
reversal coinciding with a horizontal reversal (vx_flip True) is NOT treated as a
pure floor bounce. We re-derive events on BOTH:
  - the LIVE inputs dump (/tmp/methodoprod/live_inputs.json)
  - the frozen demo3 cache (run_demo3.methodo path)
and score both vs GT.

This does not edit vision/events.py — it imports a patched variant so we can
measure before committing.

Run: venv/bin/python tools/event_eval/proto_vxgate.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import vision.events as V  # noqa: E402
from tools.event_eval.event_eval import match_events, scores  # noqa: E402

GT_B = [62, 120, 174, 255, 308, 369, 456, 511, 569]
GT_H = [81, 141, 196, 268, 333, 394, 468, 525]


def patched_classify_events(bounce_cands, hit_cands, raw_centers, is_real,
                            players_per_frame, fps, frame_width, frame_height,
                            *, vx_gate=True, **kw):
    """A copy of classify_events PASS-0/1 with the vy anchor gated by vx, then it
    delegates to the same decode. To avoid duplicating the whole 200-line function
    we instead call the real one but with a wrapper around _direction_features that
    suppresses vy_flip when vx_flip is True (a redirect)."""
    orig = V._direction_features

    def gated(f, raw, real, fps_, fw_, **kk):
        ft = orig(f, raw, real, fps_, fw_, **kk)
        if vx_gate and ft.get("vy_flip") and ft.get("vx_flip") is True:
            # vertical AND horizontal reversal => redirect (HIT), not a floor bounce
            ft = dict(ft)
            ft["vy_flip"] = False
        return ft

    V._direction_features = gated
    try:
        return V.classify_events(
            bounce_cands, hit_cands, raw_centers, is_real, players_per_frame,
            fps, frame_width, frame_height, **kw)
    finally:
        V._direction_features = orig


def score(events, tol):
    ev = [{"frame": e["frame"], "label": e["label"]} for e in events]
    res = match_events(ev, GT_B, GT_H, tol)
    s = scores(res)
    return res, s


def run_live(vx_gate):
    d = json.loads(Path("/tmp/methodoprod/live_inputs.json").read_text())
    fps, fw, fh = d["fps"], d["width"], d["height"]
    raw = [tuple(c) if c is not None else None for c in d["raw_ball_centers"]]
    isr = d["ball_is_real"]
    ac = [tuple(c) if c is not None else None for c in d["all_ball_centers"]]
    bc = [(int(f), int(x), int(y)) for f, x, y in d["b_cands"]]
    turns = sorted(int(t) for t in d["turns"])
    hits = d["hits"]
    ppf = d["methodo_ppf"]
    fn = patched_classify_events if vx_gate else V.classify_events
    if vx_gate:
        events, _ = patched_classify_events(
            bc, hits, raw, isr, ppf, fps, fw, fh, vx_gate=True,
            turning_frames=turns, smoothed_centers=ac,
            wasb_centers=None, wasb_is_real=None)
    else:
        events, _ = V.classify_events(
            bc, hits, raw, isr, ppf, fps, fw, fh,
            turning_frames=turns, smoothed_centers=ac,
            wasb_centers=None, wasb_is_real=None)
    return score(events, round(0.15 * fps))


def run_cache(vx_gate):
    from vision.bounce import smooth_ball_trajectory, detect_bounces_robust
    from vision.shots import detect_hits
    from vision.events import detect_turning_points, detect_sharp_turns
    c = json.loads((ROOT / "tests/fixtures/cache/demo3_event.json").read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kr = c["kalman_is_real"]
    ppf = c["players_per_frame"]
    ac = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(ac)
    bc, _, _ = detect_bounces_robust(ac, speeds, fps, fh, fw)
    turns = set(detect_turning_points(ac, fps, fh)) | set(
        detect_sharp_turns(kal, kr, fps, fw, fh))
    hits = detect_hits(ac, ppf, fps, fh, fw, bounce_frames=[])
    if vx_gate:
        events, _ = patched_classify_events(
            bc, hits, kal, kr, ppf, fps, fw, fh, vx_gate=True,
            turning_frames=sorted(turns), smoothed_centers=ac,
            wasb_centers=None, wasb_is_real=None)
    else:
        events, _ = V.classify_events(
            bc, hits, kal, kr, ppf, fps, fw, fh,
            turning_frames=sorted(turns), smoothed_centers=ac,
            wasb_centers=None, wasb_is_real=None)
    return score(events, round(0.15 * fps))


def line(name, res, s):
    print(f"  {name:18}: H->B={res['conf_HtoB']} B->H={res['conf_BtoH']}  "
          f"bounceF1={s['bounce']['F1']:.3f} hitF1={s['hit']['F1']:.3f}  "
          f"(b R={s['bounce']['R']:.2f} P={s['bounce']['P']:.2f})")


if __name__ == "__main__":
    print("LIVE inputs (prod ball track):")
    line("OFF (current)", *run_live(False))
    line("ON  (vx-gate)", *run_live(True))
    print("\nCACHE (frozen demo3):")
    line("OFF (current)", *run_cache(False))
    line("ON  (vx-gate)", *run_cache(True))
