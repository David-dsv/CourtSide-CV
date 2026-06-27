"""Inspect the f120..f196 segment in detail: every merged event with its raw
feature record (vy, dmin, hit/bnc/turn, hit_like, bscore, anchor) so we see why the
DP picks f159 (spurious) over f174 (the GT bounce)."""
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


def main():
    c = json.loads(CACHE.read_text())
    fps, fw, fh = c["fps"], c["width"], c["height"]
    kal = [tuple(x) if x else None for x in c["kalman_centers"]]
    kal_real = c["kalman_is_real"]
    wasb = [tuple(x) if x else None for x in c["wasb_centers"]]
    wasb_real = c["wasb_is_real"]
    ppf = c["players_per_frame"]

    all_centers = smooth_ball_trajectory(kal, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    b_cands, _re, _nd = detect_bounces_robust(all_centers, speeds, fps, fh, fw)
    turns = sorted(set(detect_turning_points(all_centers, fps, fh))
                   | set(detect_sharp_turns(kal, kal_real, fps, fw, fh)))
    hits = detect_hits(all_centers, ppf, fps, fh, fw, bounce_frames=[])

    # Monkeypatch classify_events to capture the merged `evs` before decode by
    # re-implementing the merge with the same params (cheap: call internals).
    # Simpler: temporarily wrap _global_alternation_decode to print state.
    orig_decode = ev._global_alternation_decode
    captured = {}

    def spy(evs, step, keep, lam, miss_cost, anchor_bonus=3.0):
        captured["evs"] = evs
        captured["step"] = step
        return orig_decode(evs, step, keep, lam, miss_cost, anchor_bonus)

    ev._global_alternation_decode = spy
    events, report = ev.classify_events(
        b_cands, hits, kal, kal_real, ppf, fps, fw, fh,
        turning_frames=turns, smoothed_centers=all_centers,
        wasb_centers=wasb, wasb_is_real=wasb_real)
    ev._global_alternation_decode = orig_decode

    evs = captured["evs"]
    print(f"STEP = {captured['step']:.1f}f")
    print("merged events in f120..f200:")
    print(f"  {'frame':>5} {'src':<22} vy {'dmin':>5} hit bnc turn hit_like "
          f"{'bscore':>6} {'hscore':>6} {'anchor':<7} -> label")
    final = {e["frame"]: e["label"] for e in events}
    for e in evs:
        if not (110 <= e["frame"] <= 205):
            continue
        print(f"  f{e['frame']:>3} {str(e['src']):<22} "
              f"{int(e['vy'])}  {e['dmin']:>5.2f} "
              f"{int(e['hit'])}   {int(e['bnc'])}   {int(e['turn'])}    "
              f"{int(e['hit_like'])}     {e['bscore']:>6.2f} {e['hscore']:>6.2f} "
              f"{str(e['anchor']):<7} -> {final.get(e['frame'])}")


if __name__ == "__main__":
    main()
