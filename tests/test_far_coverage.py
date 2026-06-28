"""Regression lock for FAR-player SELECTION accuracy (feat/farselect-integrated).

Background: the far/P1 slot used to lock onto a static spectator above the net —
far-CORRECT was 2/225 = 0.9% on the demo3 human pose GT (mean 572px off). The
integrated selector (far-select A's unified ``hcv`` score + far-select B's
skeleton-decoupling) fixed it. Measured LIVE on the corrected demo3 clip (S2), the
production pipeline now selects the far player within ~6px of the human GT on
**225/225 = 100.0%** of frames (at every threshold down to 1.5%W), while keeping the
corner-filmed felix clip at 87.9% (a pure-centrality score regressed felix to 4.5%;
a hard keypoint gate regressed demo3 to 26.7% — the hcv score wins both).

NOTE (S2): the previously-committed cache fixture reported only 84.4% because it was
built against the OLD annotated ``tennis_demo3.mp4`` before the clip was re-extracted
(corrected). The candidate boxes came from a *different image* than the current frame,
so the replay lost frames the live pipeline nails. The fixture has been REBUILT against
the corrected clip and the replay now reproduces the live 100%. See
``docs/research/s2-far-pose-sota-CR.md``. (Lesson, again: validate on a LIVE run, not a
frozen cache — here the stale cache *understated* the truth.)

This test is FAST (no GPU / no video decode): it replays the committed cache of
posed FAR candidates per GT frame (``demo3_far_select_cache.json``) through the
PRODUCTION far-selection scoring function ``vision.pose._hcv_far_score`` and the same
on-court gate, then asserts far-CORRECT stays above a locked floor. It exercises the
exact scoring the live pipeline uses (the function is shared, so the test cannot
drift from prod), isolating SELECTION — the thing the far-select change governs —
from detection/pose. Rebuild the fixture with
``tools/pose_gt/build_far_select_cache.py`` if the detector or det_conf changes.

Run:  python tests/test_far_coverage.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.pose import _hcv_far_score  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "pose_gt" / "demo3_far_select_cache.json"

# Locked floor: measured LIVE far-CORRECT is 100.0% (225/225, mean 6px) and the rebuilt
# cache replay of the SAME scoring reproduces it exactly (225/225, 0 losses). Floor set
# to 0.95 — below the achieved 100% with margin for detector/pose float-noise (the cache
# can be rebuilt on CPU or MPS) and future model drift, yet far above the prior 0.80 and
# the 0.9% baseline, so it genuinely locks the gain without being a knife-edge.
FLOOR = 0.95
MATCH_FRAC = 0.10        # a far pick is CORRECT within 10% of frame width of GT


def _select_far(frame, fw, net_y):
    """Replay the production far-selection over a cached frame's candidates and
    return the chosen far box center (cx, cy), or None. Mirrors
    ``estimate_players_pose``: on-court gate → hcv argmax (height/hmax + conf +
    centrality + soft valid)."""
    cands = frame["candidates"]
    if not cands:
        return None
    entries = []
    for c in cands:
        box = c["box"]
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        # same on-court gate as prod: below the top-stands knee + inside lateral band
        oncourt = (cy >= 0.25 * net_y and 0.12 < cx / fw < 0.88)
        valid = 1 if c["nkp"] >= 10 else 0
        entries.append((box, cx, cy, float(c["conf"]), valid, oncourt))
    pool = [e for e in entries if e[5]] or entries        # prefer on-court
    hmax = max((e[0][3] - e[0][1]) for e in pool) or 1.0
    best = max(pool, key=lambda e: _hcv_far_score(e[0], fw, e[3], e[4], hmax))
    return (best[1], best[2])


def _coverage():
    data = json.loads(CACHE.read_text())
    fw, net_y = data["width"], data["net_y"]
    checked = correct = 0
    for fr in data["frames"]:
        checked += 1
        pick = _select_far(fr, fw, net_y)
        if pick is None:
            continue
        gx, gy = fr["gt"]
        d = ((pick[0] - gx) ** 2 + (pick[1] - gy) ** 2) ** 0.5
        if d < MATCH_FRAC * fw:
            correct += 1
    return correct, checked


def test_far_select_above_floor():
    correct, checked = _coverage()
    frac = correct / max(1, checked)
    assert checked > 0, "empty far-select cache fixture"
    assert frac >= FLOOR, (
        f"far-CORRECT {correct}/{checked} = {frac:.1%} < floor {FLOOR:.0%} — "
        f"far-player selection regressed (baseline was 0.9%)")


def test_far_select_beats_individual_approaches():
    """The integrated selector must clear BOTH prior single-clip results on demo3:
    far-select A 84.9% and far-select B 81.3%. On the rebuilt (corrected-clip) cache the
    replay is 100% — comfortably above both — so this asserts we never regress below the
    stronger of the two prior approaches (A's 84.9%)."""
    correct, checked = _coverage()
    frac = correct / max(1, checked)
    assert frac >= 0.849, f"far-CORRECT {frac:.1%} fell below A's 84.9% demo3 result"


def test_hcv_scoring_is_height_dominant_and_valid_soft():
    """Guard the two load-bearing properties of the shared scoring function:
    (1) a TALLER on-court box outscores a shorter one at equal centrality/conf
        (the cross-geometry height signal), and
    (2) ``valid`` is a SOFT bonus, never a hard gate: a tall central INVALID box
        still outscores a short central VALID one (the fix for the keypoint-gate
        backfire on the tiny far player that cannot pose)."""
    fw = 1920
    # equal centre & conf, different height → taller wins
    tall = [900, 100, 1000, 320]      # h=220, central
    short = [900, 100, 1000, 200]     # h=100, central
    hmax = 220.0
    assert (_hcv_far_score(tall, fw, 0.5, 0, hmax)
            > _hcv_far_score(short, fw, 0.5, 0, hmax))
    # tall INVALID still beats short VALID (valid is soft, weight 0.3 < height span)
    assert (_hcv_far_score(tall, fw, 0.5, 0, hmax)
            > _hcv_far_score(short, fw, 0.5, 1, hmax))


if __name__ == "__main__":
    correct, checked = _coverage()
    print(f"far-CORRECT (cache replay of prod hcv selection): "
          f"{correct}/{checked} = {100 * correct / max(1, checked):.1f}%  "
          f"(floor {FLOOR:.0%}, live measured 100.0%)")
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
            print(f"  PASS {name}")
    print("PASS")
