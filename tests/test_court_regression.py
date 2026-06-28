"""
Auto court-homography regression test (fast, no GPU / no video decode / no torch).

Runs the classical auto-court detector (models/court_auto + court_lines) on a
committed white-line-MASK cache (tests/fixtures/court_lines/*_mask.png) and
asserts the gate + fit BEHAVIOUR does not regress. Mirrors
tests/test_bounce_regression.py: a committed derived-artifact fixture, asserted
against floors set below the first measured values, runnable in <2s.

The single most important assertion is FELIX → fallback: the detector must
ABSTAIN on a grazing angle (depth axis unobservable) rather than emit a
confidently-wrong homography. The R&D philosophy: never lie about the metric.

    venv/bin/python tests/test_court_regression.py
    # or: venv/bin/python -m pytest tests/test_court_regression.py -q
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from models.court_lines import (  # noqa: E402
    detect_line_candidates, estimate_vp_pencils, abstention_verdict,
)
from models.court_auto import AutoCourtDetector  # noqa: E402

CACHE = ROOT / "tests" / "fixtures" / "court_lines"


def _load(name):
    mask = cv2.imread(str(CACHE / f"{name}_mask.png"), cv2.IMREAD_GRAYSCALE)
    meta = json.loads((CACHE / f"{name}_meta.json").read_text())
    return mask, meta


def _run_from_mask(mask, frame_wh):
    """Drive the detector's stages from a precomputed mask (no video / HSV /
    torch). Returns a homography dict identical in shape to compute_homography."""
    w, h = frame_wh
    diag = float(np.hypot(w, h))
    lines = detect_line_candidates(mask)
    if len(lines) < 4:
        return AutoCourtDetector._fallback(f"only {len(lines)} lines")
    vp_info = estimate_vp_pencils(lines, (w, h))
    verdict = abstention_verdict(vp_info)
    if verdict["abstain"]:
        return AutoCourtDetector._fallback(verdict["reason"], gate=verdict["gate_features"])
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    det = AutoCourtDetector()
    best = det._fit(vp_info["principal"], lines, dt, diag, (w, h))
    if best is None:
        return AutoCourtDetector._fallback("no court hypothesis", gate=verdict["gate_features"])
    H, score, lp = best
    H_inv = np.linalg.inv(H)
    ok, aspect = det._aspect_quad_ok(H_inv, diag)
    if not ok:
        return AutoCourtDetector._fallback(f"aspect {aspect:.2f}", gate=verdict["gate_features"])
    conf = det._confidence(score, vp_info, diag)
    if conf == "fallback":
        return AutoCourtDetector._fallback(f"score {score:.2f}", gate=verdict["gate_features"])
    return {"H": H, "H_inv": H_inv, "confidence": conf, "warpback_score": float(score),
            "landmark_px": lp, "aspect": aspect, "source": "autoline"}


def _metric_self_consistency(homo):
    """Measure court width/length BETWEEN DISTINCT corner landmarks (non-circular
    only insofar as the corners come from the solved H, but the RATIO of the two
    is a real perspective-consistency check). Returns (baseline_m, sideline_m)."""
    H, lp = np.asarray(homo["H"]), homo["landmark_px"]

    def to_m(name):
        x, y = lp[name]
        p = H @ np.array([x, y, 1.0])
        return p[:2] / p[2]
    bl = float(np.linalg.norm(to_m("far_bl_left") - to_m("far_bl_right")))
    sl = float(np.linalg.norm(to_m("far_bl_left") - to_m("near_bl_left")))
    return bl, sl


def test_court_regression():
    results = {}
    for name in ("tennis", "felix", "djokovic"):
        if not (CACHE / f"{name}_mask.png").exists():
            continue
        mask, meta = _load(name)
        results[name] = _run_from_mask(mask, meta["frame_wh"])

    # ── the load-bearing assertion: NEVER lie on the grazing angle ──
    assert "felix" in results, "felix fixture missing"
    assert results["felix"]["confidence"] == "fallback", (
        f"felix must ABSTAIN on the grazing angle, got "
        f"{results['felix']['confidence']} ({results['felix'].get('abstain_reason')})")

    # ── tennis: a full-court elevated view must solve with a sane court ──
    if "tennis" in results:
        t = results["tennis"]
        assert t["confidence"] != "fallback", (
            f"tennis should solve, abstained: {t.get('abstain_reason')}")
        # projected court aspect in a sane band (deeper-than-wide ish, not a sliver)
        assert 0.18 <= t["aspect"] <= 2.5, f"tennis aspect {t['aspect']:.2f} out of band"
        bl, sl = _metric_self_consistency(t)
        # ITF truth 10.97 / 23.77 m — allow a generous floor (classical far-field
        # conditioning), set BELOW the first measured values so noise won't flake.
        assert 9.5 <= bl <= 12.5, f"tennis baseline {bl:.2f}m off (truth 10.97)"
        assert 20.0 <= sl <= 27.5, f"tennis sideline {sl:.2f}m off (truth 23.77)"

    # ── djokovic frame-0 is a non-court / occluded shot: honest abstain ──
    # (asserting the HONEST behaviour, not a fabricated success — the clean
    #  full-court frame lacks both sidelines upstream, see CR §limits.)
    if "djokovic" in results:
        assert results["djokovic"]["confidence"] == "fallback", (
            "djokovic frame-0 (non-court) should abstain, not emit a homography")

    print("\n=== court regression ===")
    for name, r in results.items():
        if r["confidence"] == "fallback":
            print(f"  {name:9} ABSTAIN ({r.get('abstain_reason','')[:60]})")
        else:
            bl, sl = _metric_self_consistency(r)
            print(f"  {name:9} {r['confidence']:5} warpback={r['warpback_score']:.2f} "
                  f"aspect={r['aspect']:.2f} baseline={bl:.2f}m sideline={sl:.2f}m")
    print("PASS")


if __name__ == "__main__":
    test_court_regression()
