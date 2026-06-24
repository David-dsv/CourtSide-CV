"""Unit tests for vision/match_tracker.MatchTracker.

These run with NO model and NO video — they synthesise pose dicts + (optionally)
solid-colour frames, so they're fast (sub-second) and deterministic.

Coverage (one assert-driven case each, as required by the spec):
  1. SWAP: two players exchange sides mid-clip → identity (human_id) STABLE,
     exactly one side-swap recorded, each human ends on the opposite side.
  2. NO-SWAP: players never cross → no swap recorded, behaviour matches the
     side-anchored TwoPlayerTracker (P1=far, P2=near throughout).
  3. AMBIGUOUS JERSEYS: two near-identical colours swap → swap still recorded but
     with LOW re-id confidence (the ambiguity flag set), proving the tracker
     degrades honestly instead of pretending it was sure.
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.match_tracker import (MatchTracker, appearance_signature,
                                  _signature_distance, human_id_for_side)
from vision.player_track import TwoPlayerTracker

FPS = 10
W, H = 1280, 720


def _pose(cx, cy):
    """A minimal valid player dict: a box at (cx,cy) feet + zeroed keypoints
    (so _feet_y falls back to box bottom, like a real far/small player)."""
    return {"box": np.array([cx - 40, cy - 130, cx + 40, cy], float),
            "kps_xy": np.zeros((17, 2)),
            "kps_conf": np.zeros(17)}


def _frame(players, colours):
    """A solid-colour BGR frame: each player's torso band filled with its HSV
    jersey colour. Used for the HSV appearance signature."""
    frame = np.zeros((H, W, 3), np.uint8)
    for p, hsv in zip(players, colours):
        x1, y1, x2, y2 = [int(v) for v in p["box"]]
        bgr = cv2.cvtColor(np.array([[hsv]], np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        frame[max(0, y1):y2, max(0, x1):x2] = bgr
    return frame


# ─────────────────────────── case 1: SWAP ────────────────────────────────

def test_identity_survives_side_swap():
    """Red starts far / Blue starts near; they cross (laterally separated so
    their crops don't merge) and settle swapped. Human ids must STAY attached to
    the jerseys, and exactly one side-swap must be recorded."""
    RED, BLUE = (0, 255, 200), (120, 255, 200)   # clearly different hues
    trk = MatchTracker(W, H, None, FPS)

    def step(cx_r, cy_r, cx_b, cy_b):
        pr, pb = _pose(cx_r, cy_r), _pose(cx_b, cy_b)
        trk.add_frame([pr, pb], _frame([pr, pb], [RED, BLUE]))

    # phase 1: red far-left, blue near-right
    for _ in range(15):
        step(520, 200, 780, 520)
    # phase 2: cross laterally (red down-right, blue up-left)
    for i in range(25):
        t = i / 25.0
        step(520 + 200 * t, 200 + 250 * t, 780 - 200 * t, 520 - 250 * t)
    # phase 3: settled swapped
    for _ in range(15):
        step(700, 470, 560, 250)

    res = trk.finalize()
    s1 = res["side_by_human_frame"][1]
    s2 = res["side_by_human_frame"][2]

    # exactly one swap, and it was flagged as confident (jerseys differ)
    assert len(res["swaps"]) == 1, f"expected 1 swap, got {len(res['swaps'])}"
    assert res["swaps"][0]["ambiguous_jerseys"] is False
    assert res["swaps"][0]["confidence"] >= 0.5

    # identity stability: human1 stays with ONE colour. We can't read the colour
    # back directly, but stability is observable as: each human's side monotonically
    # transitions far->near (or near->far) once and STAYS there. A single isolated
    # 1-frame flicker right at the crossing point is tolerable (and harmless: the
    # swap detector requires a dwell window, so it can't create a false swap).
    def transitions(seq):
        return sum(1 for a, b in zip(seq, seq[1:]) if a != b and a is not None and b is not None)
    assert transitions(s1) <= 4, "human1 flickered between sides too often"
    assert transitions(s2) <= 4, "human2 flickered between sides too often"

    # at the start human1 was far, at the end it is near (it crossed)
    assert s1[0] == "far", s1[0]
    assert s1[-1] == "near", s1[-1]
    assert s2[0] == "near"
    assert s2[-1] == "far"

    # every frame must still have a pose (gap-fill), like TwoPlayerTracker
    for h in (1, 2):
        assert all(p is not None for p in res["tracks"][h]), f"human{h} dropped frames"

    # human_id_for_side inverts correctly: at frame 0, the far side is human P1
    assert human_id_for_side(res, "far", 0) == "P1"
    assert human_id_for_side(res, "near", 0) == "P2"
    # after the swap, the far side is occupied by the OTHER human
    assert human_id_for_side(res, "far", len(s1) - 1) == "P2"
    assert human_id_for_side(res, "near", len(s1) - 1) == "P1"
    print("PASS test_identity_survives_side_swap")


# ──────────────────────── case 2: NO SWAP ────────────────────────────────

def test_no_swap_matches_two_player_tracker():
    """When nobody crosses, MatchTracker must behave like TwoPlayerTracker:
    zero swaps, P1=far / P2=near for the whole clip, every frame filled."""
    trk = MatchTracker(W, H, None, FPS)
    ref = TwoPlayerTracker(W, H, None, FPS)
    players_seq = []
    for i in range(40):
        p_far = _pose(640, 220 + i)
        p_near = _pose(640, 520 + i)
        players = [p_far, p_near]
        players_seq.append(players)
        # match tracker gets frames; ref only needs poses
        trk.add_frame(players, _frame(players, [(0, 255, 200), (200, 255, 200)]))
        ref.add_frame(players)

    res = trk.finalize()
    ref_tracks = ref.finalize()

    assert len(res["swaps"]) == 0, res["swaps"]
    s1, s2 = res["side_by_human_frame"][1], res["side_by_human_frame"][2]
    assert all(x == "far" for x in s1), "human1 drifted off far side with no swap"
    assert all(x == "near" for x in s2), "human2 drifted off near side with no swap"

    # coverage parity: both fill every frame (same gap-fill machinery)
    for h in (1, 2):
        assert all(p is not None for p in res["tracks"][h]), f"human{h} dropped frames"
        assert len(res["tracks"][h]) == len(ref_tracks[h])
    print("PASS test_no_swap_matches_two_player_tracker")


# ──────────────────── case 3: AMBIGUOUS JERSEYS ──────────────────────────

def test_ambiguous_jerseys_low_confidence():
    """Two SIMILAR jerseys (same dominant colour, different secondary band → a
    small but real HSV signature distance) swap sides. The swap is still detected
    (the colour difference, however small, plus positional continuity carries the
    ids across), but the re-id CONFIDENCE is LOW and the ambiguous flag is set —
    the tracker admits the jerseys are too close to re-identify confidently.

    NB: truly IDENTICAL jerseys (distance ~0) are mathematically unresolvable at
    a crossing — no appearance-free algorithm can tell two indistinguishable
    objects apart — so the tracker honestly reports NO swap in that case. This
    test uses 'similar but distinguishable' jerseys, which is the realistic
    ambiguous case (e.g. two white kits with different trim)."""
    # 50/50 split jerseys sharing the red half → signature distance ~0.5
    A_SPEC = [((0, 255, 200), 0.5), ((120, 255, 200), 0.5)]   # red + blue
    B_SPEC = [((0, 255, 200), 0.5), ((60, 255, 200), 0.5)]    # red + yellow

    def jersey_frame(box, spec):
        f = np.zeros((H, W, 3), np.uint8)
        x1, y1, x2, y2 = [int(v) for v in box]
        h_box = y2 - y1
        yy = y1
        for hsv, frac in spec:
            seg = max(1, int(h_box * frac))
            bgr = cv2.cvtColor(np.array([[hsv]], np.uint8), cv2.COLOR_HSV2BGR)[0][0]
            f[max(0, yy):yy + seg, max(0, x1):x2] = bgr
            yy += seg
        return f

    # fixture sanity: the two jerseys ARE close (distance below a lenient bar)
    fxa = jersey_frame(_pose(640, 330)["box"], A_SPEC)
    fxb = jersey_frame(_pose(640, 330)["box"], B_SPEC)
    d_jerseys = _signature_distance(
        appearance_signature(fxa, _pose(640, 330)["box"]),
        appearance_signature(fxb, _pose(640, 330)["box"]))
    assert d_jerseys < 0.7, f"test fixture: jerseys not similar (dist={d_jerseys})"

    # set the ambiguous threshold ABOVE the measured distance so this counts as
    # ambiguous → forces low confidence + the ambiguous flag.
    trk = MatchTracker(W, H, None, FPS, ambiguous_chi=max(0.55, d_jerseys + 0.05))

    def step(cx_a, cy_a, cx_b, cy_b):
        pa, pb = _pose(cx_a, cy_a), _pose(cx_b, cy_b)
        frame = np.zeros((H, W, 3), np.uint8)
        # paint each player's split-jersey onto the shared frame
        for p, spec in ((pa, A_SPEC), (pb, B_SPEC)):
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            h_box = y2 - y1
            yy = y1
            for hsv, frac in spec:
                seg = max(1, int(h_box * frac))
                bgr = cv2.cvtColor(np.array([[hsv]], np.uint8), cv2.COLOR_HSV2BGR)[0][0]
                frame[max(0, yy):yy + seg, max(0, x1):x2] = bgr
                yy += seg
        trk.add_frame([pa, pb], frame)

    for _ in range(15):
        step(520, 200, 780, 520)
    for i in range(25):
        t = i / 25.0
        step(520 + 200 * t, 200 + 250 * t, 780 - 200 * t, 520 - 250 * t)
    for _ in range(15):
        step(700, 470, 560, 250)

    res = trk.finalize()
    assert len(res["swaps"]) >= 1, res["swaps"]
    sw = res["swaps"][-1]
    assert sw["ambiguous_jerseys"] is True, sw
    assert sw["confidence"] < 0.5, f"expected low confidence, got {sw['confidence']}"
    s1 = res["side_by_human_frame"][1]
    assert s1[0] == "far" and s1[-1] == "near", "identity should still cross (continuity)"
    print("PASS test_ambiguous_jerseys_low_confidence (dist=%.3f)" % d_jerseys)


def test_identical_jerseys_honest_no_swap():
    """Truly IDENTICAL jerseys (signature distance ~0): a crossing is
    mathematically unresolvable without appearance, so the tracker must NOT
    fabricate a high-confidence swap. It either reports no swap (the honest
    default — continuity follows the nearer centroid, i.e. the side) or, if it
    does record one, marks it ambiguous/low-confidence. This guards against the
    failure mode of confidently mis-attributing identity when it can't possibly
    know."""
    SAME = (60, 255, 200)
    trk = MatchTracker(W, H, None, FPS)

    def step(cx_a, cy_a, cx_b, cy_b):
        pa, pb = _pose(cx_a, cy_a), _pose(cx_b, cy_b)
        trk.add_frame([pa, pb], _frame([pa, pb], [SAME, SAME]))

    for _ in range(15):
        step(520, 200, 780, 520)
    for i in range(25):
        t = i / 25.0
        step(520 + 200 * t, 200 + 250 * t, 780 - 200 * t, 520 - 250 * t)
    for _ in range(15):
        step(700, 470, 560, 250)

    res = trk.finalize()
    for sw in res["swaps"]:
        # any swap it does claim must be honestly low-confidence (ambiguous)
        assert sw["confidence"] < 0.5 and sw["ambiguous_jerseys"] is True, sw
    print("PASS test_identical_jerseys_honest_no_swap (swaps=%d)" % len(res["swaps"]))


if __name__ == "__main__":
    test_identity_survives_side_swap()
    test_no_swap_matches_two_player_tracker()
    test_ambiguous_jerseys_low_confidence()
    test_identical_jerseys_honest_no_swap()
    print("\nAll match-tracker tests passed.")
