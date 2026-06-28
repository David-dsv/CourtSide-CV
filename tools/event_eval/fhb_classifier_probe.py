"""
Isolate the FH/BH CLASSIFIER from hit-detection noise.

The doc's 2/8->8/8 measured classify_forehand_backhand at each GT CONTACT with the
GT-correct player (the on-court player whose body anchor is nearest the GT ball) —
decoupling stroke-type classification from detect_hits timing/attribution (a
separate, bounce/event-session problem). This probe reproduces that: for each GT
shot it builds a synthetic hit dict {frame, x, y, player_idx} pointing at the
nearest on-court player, runs classify_forehand_backhand, and scores vs the GT
label. Use this to iterate the classifier; then validate end-to-end LIVE.

Usage:  python tools/event_eval/fhb_classifier_probe.py
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vision.shots import (  # noqa: E402
    classify_forehand_backhand, estimate_handedness_prior,
    _player_anchor, _player_height, _side_of)

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_pose.json"
GT = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"


def nearest_player(players, bx, by, net_y):
    """The on-court player whose body anchor is nearest the ball (player-height
    units) — the GT-correct contact player. Returns (idx, player, side, d/h)."""
    best = None
    for idx, p in enumerate(players):
        ax, ay = _player_anchor(p)
        h = max(_player_height(p), 1.0)
        d = float(np.hypot(bx - ax, by - ay)) / h
        if best is None or d < best[3]:
            best = (idx, p, _side_of(p, net_y), d)
    return best


def run():
    c = json.loads(CACHE.read_text())
    fh = c["height"]
    fps = c["fps"]
    net_y = fh * 0.47
    ppf = c["players_per_frame"]
    gt = json.loads(GT.read_text())["shots"]

    # build synthetic hits at GT contacts with the GT-correct player, then vote the
    # handedness prior over them (exactly what classify_shots does in prod)
    synth_hits = []
    for s in gt:
        f, bx, by = s["frame"], s["x"], s["y"]
        players = ppf[f] if f < len(ppf) else []
        if not players:
            synth_hits.append(None)
            continue
        idx, p, side, dh = nearest_player(players, bx, by, net_y)
        synth_hits.append({"frame": f, "x": bx, "y": by, "player_idx": idx,
                           "player_side": side})
    handed = estimate_handedness_prior(
        [h for h in synth_hits if h], ppf, fps, fh)

    rows = []
    correct = called = abstain = cFHtoBH = cBHtoFH = 0
    for s, hit in zip(gt, synth_hits):
        f, label = s["frame"], s["type"]
        if hit is None:
            rows.append((f, label, "no-players", "?", "miss"))
            continue
        side = hit["player_side"]
        pred = classify_forehand_backhand(
            hit, ppf, players_per_frame_window=ppf, handedness=handed,
            fps=fps, frame_height=fh)
        if pred == "unknown":
            abstain += 1
            verdict = "abstain"
        else:
            called += 1
            if pred == label:
                correct += 1
                verdict = "ok"
            else:
                verdict = "WRONG"
                if label == "forehand":
                    cFHtoBH += 1
                else:
                    cBHtoFH += 1
        rows.append((f, label, pred, side, verdict))

    return rows, dict(correct=correct, called=called, abstain=abstain,
                      cFHtoBH=cFHtoBH, cBHtoFH=cBHtoFH, n=len(gt))


def main():
    rows, m = run()
    acc = m["correct"] / m["called"] if m["called"] else 0.0
    print(f"[classifier probe @GT contacts] {m['correct']}/{m['called']} called "
          f"(acc {acc:.3f}) | abstain {m['abstain']}/{m['n']} | "
          f"FH->BH {m['cFHtoBH']} BH->FH {m['cBHtoFH']}")
    for r in rows:
        f, label, pred, side = r[0], r[1], r[2], r[3]
        verdict = r[4]
        flag = {"ok": "OK", "WRONG": "XX", "abstain": "..", "miss": "--"}[verdict]
        print(f"   {flag} f{f:>4} GT {label:8s} -> {pred:8s} [{side}]")


if __name__ == "__main__":
    main()
