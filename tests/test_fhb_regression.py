"""
Forehand/Backhand CLASSIFIER regression test (fast, no GPU / no video decode).

Replays classify_forehand_backhand (the composite: ball-at-contact + swing-window
cross-body override + abstention + handedness prior) on the committed demo3 pose
cache (tests/fixtures/cache/demo3_pose.json) at each GT contact, with the
GT-correct player (nearest body anchor), and asserts the FH/BH metrics vs the GT
fixture do not regress:

  - confusion FH<->BH == 0   (NON-NEGOTIABLE: no stroke shown as its opposite)
  - accuracy on CALLED shots >= floor
  - abstention <= cap

This locks the CLASSIFIER quality (the doc's measured 8/8, 0/0, 0 abstain),
isolated from hit-DETECTION recall/attribution (a separate bounce/event-session
concern). It replays a cache, so it's fast and deterministic (no model inference).

Run:  venv/bin/python tests/test_fhb_regression.py
"""
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.shots import (  # noqa: E402
    classify_forehand_backhand, estimate_handedness_prior,
    _player_anchor, _player_height, _side_of)

CACHE = ROOT / "tests" / "fixtures" / "cache" / "demo3_pose.json"
GT = ROOT / "tests" / "fixtures" / "shots" / "tennis_demo3.shots.json"

# Floors locked at the measured composite numbers. confusion is a HARD 0 (a forehand
# must never be shown as a backhand and vice-versa) — that is the load-bearing
# guarantee. The composite calls 7/7 correct and ABSTAINS on the single genuine
# cross-body (f525): abstaining there (rather than the doc's swing-override, which
# introduced a far-court confusion live) is the deliberate choice that keeps
# confusion 0 on BOTH the cache and live. So: 7/7 called correct, ≤1 abstain, 0
# confusion. (A small margin so a benign pose-cache rebuild doesn't flake.)
CONF_MAX = 0
ACC_CALLED_FLOOR = 1.0      # every CALLED shot correct
ABSTAIN_MAX = 1            # the lone genuine cross-body (f525) is abstained, not guessed


def _nearest_player(players, bx, by, net_y):
    best = None
    for idx, p in enumerate(players):
        ax, ay = _player_anchor(p)
        h = max(_player_height(p), 1.0)
        d = float(np.hypot(bx - ax, by - ay)) / h
        if best is None or d < best[3]:
            best = (idx, p, _side_of(p, net_y), d)
    return best


def evaluate():
    c = json.loads(CACHE.read_text())
    fh, fps = c["height"], c["fps"]
    net_y = fh * 0.47
    ppf = c["players_per_frame"]
    gt = json.loads(GT.read_text())["shots"]

    synth = []
    for s in gt:
        f, bx, by = s["frame"], s["x"], s["y"]
        players = ppf[f] if f < len(ppf) else []
        if not players:
            synth.append(None)
            continue
        idx, p, side, _ = _nearest_player(players, bx, by, net_y)
        synth.append({"frame": f, "x": bx, "y": by, "player_idx": idx,
                      "player_side": side})
    handed = estimate_handedness_prior([h for h in synth if h], ppf, fps, fh)

    correct = called = abstain = conf = matched = 0
    for s, hit in zip(gt, synth):
        if hit is None:
            continue
        matched += 1
        pred = classify_forehand_backhand(
            hit, ppf, players_per_frame_window=ppf, handedness=handed,
            fps=fps, frame_height=fh)
        if pred == "unknown":
            abstain += 1
        else:
            called += 1
            if pred == s["type"]:
                correct += 1
            else:
                conf += 1
    acc = correct / called if called else 0.0
    return dict(matched=matched, called=called, correct=correct,
                abstain=abstain, conf=conf, acc=acc)


def test_no_fhbh_confusion():
    m = evaluate()
    assert m["conf"] <= CONF_MAX, (
        f"FH<->BH confusion = {m['conf']} > {CONF_MAX}: a stroke shown as its "
        f"opposite (the user's bug).")


def test_accuracy_and_abstention():
    m = evaluate()
    assert m["acc"] >= ACC_CALLED_FLOOR, (
        f"FH/BH accuracy(called) {m['correct']}/{m['called']} = {m['acc']:.3f} "
        f"< floor {ACC_CALLED_FLOOR}")
    assert m["abstain"] <= ABSTAIN_MAX, (
        f"abstentions {m['abstain']} > cap {ABSTAIN_MAX}")


if __name__ == "__main__":
    m = evaluate()
    print(f"demo3 FH/BH: {m['correct']}/{m['called']} called (acc {m['acc']:.3f}) "
          f"| abstain {m['abstain']}/{m['matched']} | confusion {m['conf']}")
    ok = (m["conf"] <= CONF_MAX and m["acc"] >= ACC_CALLED_FLOOR
          and m["abstain"] <= ABSTAIN_MAX)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
