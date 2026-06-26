"""
vx-flip veto unit + gain test (fast, no GPU / no video decode).

Two parts:

  1. SYNTHETIC unit tests of vision.bounce.vx_flip_veto — the pure bounce-vs-hit
     direction veto. Hand-built trajectories assert the four behaviors:
       - vx-sign PRESERVED + vy-FLIP   -> CONFIRM, tagged "metric"
       - vx-sign FLIP (a racket hit)   -> REJECT (dropped from kept list)
       - too sparse (< 2 real pts/side)-> ABSTAIN (kept, "unconfirmed")
       - interpolated gap in a side    -> ABSTAIN (kept, "unconfirmed")
     The safety invariant: a real bounce is NEVER removed; the veto can only add
     precision.

  2. END-TO-END GAIN on the committed felix WASB native-1080p cache: reproduce
     the production detect_bounces_robust candidate list, apply the veto, and
     assert the bounce F1 rises with 0 true bounce lost and the f1609 racket-hit
     FP killed. This is the numeric proof that the veto helps on real data.

Run:  venv/bin/python tests/test_vx_veto.py
or:   venv/bin/python -m pytest tests/test_vx_veto.py -q
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vision.bounce import (  # noqa: E402
    vx_flip_veto, detect_bounces_robust, smooth_ball_trajectory)

CACHE = ROOT / "tests" / "fixtures" / "cache" / "felix_wasb_centers.json"
GT = ROOT / "tests" / "fixtures" / "bounces" / "felix.bounces.json"

# felix is natively 1920x1080; the GT json carries no width/height (CHANTIER 5:
# the candidate list is resolution-dependent, so the cache is scored at native).
FW, FH = 1920, 1080

# Post-veto floor on the WASB cache. Measured here: baseline 0.889 -> 0.914 after
# veto. The floor is set ABOVE the pre-veto baseline so it locks the gain, with a
# small margin for numeric noise.
F1_FLOOR_POST_VETO = 0.90


# ───────────────────────────── synthetic helpers ─────────────────────────────

def _make_v(half, vx_pre, vx_post, vy_pre, vy_post, x0=900, y0=400,
            pad_pre=4, pad_post=4):
    """Build a synthetic trajectory with a contact at frame f=half+pad_pre.

    Both x and y are piecewise-linear with the requested pre/post slopes, so the
    least-squares veto reads exactly (vx_pre, vx_post, vy_pre, vy_post). Padding
    frames keep the window away from the array edges.
    """
    f = half + pad_pre
    n = f + half + pad_post + 1
    centers = [None] * n
    for k in range(n):
        if k <= f:
            x = x0 + vx_pre * (k - f)
            y = y0 + vy_pre * (k - f)
        else:
            x = x0 + vx_post * (k - f)
            y = y0 + vy_post * (k - f)
        centers[k] = (float(x), float(y))
    is_real = [c is not None for c in centers]
    return centers, is_real, f, n


def _veto_one(centers, is_real, f, fps=60.0, fw=FW):
    kept, state = vx_flip_veto([(f, int(centers[f][0]), int(centers[f][1]))],
                               centers, is_real, fps, fw)
    return (len(kept) == 1), state.get(f)


# ───────────────────────────── synthetic tests ──────────────────────────────

def test_vx_preserved_vy_flip_confirms_metric():
    """A real bounce: vx sign preserved (+ -> +), vy descends then ascends."""
    half = round(60.0 * 0.10)
    centers, is_real, f, _ = _make_v(half, vx_pre=10, vx_post=6,
                                     vy_pre=8, vy_post=-8)
    kept, st = _veto_one(centers, is_real, f)
    assert kept, "a vx-preserved bounce must be KEPT"
    assert st == "metric", f"vx-preserved + vy-flip must tag 'metric', got {st}"


def test_vx_flip_rejects_hit():
    """A racket hit: vx sign reverses (- -> +). Must be removed from kept list."""
    half = round(60.0 * 0.10)
    centers, is_real, f, _ = _make_v(half, vx_pre=-12, vx_post=12,
                                     vy_pre=8, vy_post=-8)
    kept, st = _veto_one(centers, is_real, f)
    assert not kept, "a vx-FLIP candidate (hit) must be REJECTED"
    assert st == "rejected", f"a vx-flip must tag 'rejected', got {st}"


def test_vx_preserved_no_vy_flip_kept_unconfirmed():
    """vx preserved but no vy flip (a glide FP): KEEP but only 'unconfirmed'
    (the veto is one-directional — it never removes on a missing vy flip)."""
    half = round(60.0 * 0.10)
    centers, is_real, f, _ = _make_v(half, vx_pre=20, vx_post=10,
                                     vy_pre=2, vy_post=2)
    kept, st = _veto_one(centers, is_real, f)
    assert kept, "no vy flip must NOT remove the candidate (one-directional)"
    assert st == "unconfirmed", f"vx-preserved without vy-flip -> 'unconfirmed', got {st}"


def test_sparse_abstains():
    """< 2 real points on a side -> ABSTAIN (kept, 'unconfirmed'). The post side
    is entirely missing (occlusion, like felix f424)."""
    half = round(60.0 * 0.10)
    centers, is_real, f, n = _make_v(half, vx_pre=-12, vx_post=12,
                                     vy_pre=8, vy_post=-8)
    # null the post side so it cannot be measured (keep f itself)
    for k in range(f + 1, n):
        centers[k] = None
        is_real[k] = False
    kept, st = _veto_one(centers, is_real, f)
    assert kept, "an occluded/sparse side must ABSTAIN (keep), never reject"
    assert st == "unconfirmed", f"sparse -> 'unconfirmed', got {st}"


def test_gap_abstains_even_on_a_flip():
    """An interpolated gap longer than round(fps*0.05) inside a side -> ABSTAIN,
    even if the surviving points would suggest a flip. The veto must never read
    direction through a filled gap (the core safety invariant)."""
    fps = 60.0
    half = round(fps * 0.10)
    centers, is_real, f, n = _make_v(half, vx_pre=-12, vx_post=12,
                                     vy_pre=8, vy_post=-8)
    # punch a 4-frame interpolated run (> round(60*0.05)=3) into the post side.
    # Keep the (x, y) values (spline would fill them) but mark them NOT real.
    gap_thr = round(fps * 0.05)
    for k in range(f + 1, f + 1 + gap_thr + 1):
        if k < n:
            is_real[k] = False
    kept, st = _veto_one(centers, is_real, f)
    assert kept, "a gap in a side must ABSTAIN (keep), never reject"
    assert st == "unconfirmed", f"gap -> 'unconfirmed', got {st}"


def test_deadband_abstains_on_near_still_horizontal():
    """A drop-shot / slow far bounce with near-zero horizontal motion: |vx| below
    the relative deadband -> ABSTAIN (we can't read a sign reliably)."""
    half = round(60.0 * 0.10)
    # vx ~ 0 both sides, but vy has a clear (large) reflection so speed_scale and
    # the deadband floor are exercised. The horizontal motion is sub-deadband.
    centers, is_real, f, _ = _make_v(half, vx_pre=0.0, vx_post=0.0,
                                     vy_pre=10, vy_post=-10)
    kept, st = _veto_one(centers, is_real, f)
    assert kept, "near-still horizontal must ABSTAIN (keep)"
    assert st == "unconfirmed", f"deadband -> 'unconfirmed', got {st}"


# ─────────────────────────── end-to-end gain test ───────────────────────────

def _match(pred, truth, tol):
    cand = sorted((abs(p - t), p, t) for p in pred for t in truth if abs(p - t) <= tol)
    up, ut, pairs = set(), set(), []
    for _, p, t in cand:
        if p in up or t in ut:
            continue
        up.add(p)
        ut.add(t)
        pairs.append((p, t))
    matched_pred = {p for p, _ in pairs}
    matched_t = {t for _, t in pairs}
    fp = [p for p in pred if p not in matched_pred]
    fn = [t for t in truth if t not in matched_t]
    return len(pairs), fp, fn


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def evaluate():
    cache = json.loads(CACHE.read_text())
    gt = json.loads(GT.read_text())
    fps = cache["fps"]
    gt_frames = [b["frame"] for b in gt["bounces"]]
    tol = round(0.15 * fps)

    # raw pre-spline tracker centers + is_real mask (CHANTIER 0)
    raw = [None if c.get("x") is None else (float(c["x"]), float(c["y"]))
           for c in cache["centers"]]
    is_real = [c is not None for c in raw]

    # reproduce the pipeline: spline -> robust detector
    all_centers = smooth_ball_trajectory(raw, max_gap=int(fps * 0.4))
    speeds = [0.0] * len(all_centers)
    bounces, _re, _nd = detect_bounces_robust(all_centers, speeds, fps, FH, FW)

    base_pred = [b[0] for b in bounces]
    btp, bfp, bfn = _match(base_pred, gt_frames, tol)
    base_f1 = _f1(btp, len(bfp), len(bfn))

    kept, state = vx_flip_veto(bounces, raw, is_real, fps, FW)
    veto_pred = [k[0] for k in kept]
    vtp, vfp, vfn = _match(veto_pred, gt_frames, tol)
    veto_f1 = _f1(vtp, len(vfp), len(vfn))

    return {
        "tol": tol, "gt": gt_frames,
        "base": {"pred": base_pred, "tp": btp, "fp": bfp, "fn": bfn, "f1": base_f1},
        "veto": {"pred": veto_pred, "tp": vtp, "fp": vfp, "fn": vfn, "f1": veto_f1},
        "rejected": [f for f, s in state.items() if s == "rejected"],
    }


def test_veto_lifts_f1_no_tp_lost():
    r = evaluate()
    base, veto = r["base"], r["veto"]
    # gain proven
    assert veto["f1"] >= F1_FLOOR_POST_VETO, (
        f"post-veto F1 {veto['f1']:.4f} < floor {F1_FLOOR_POST_VETO}")
    assert veto["f1"] > base["f1"], (
        f"veto did not lift F1 ({base['f1']:.4f} -> {veto['f1']:.4f})")
    # 0 true bounce lost (the safety invariant)
    assert veto["tp"] == base["tp"], (
        f"TP changed {base['tp']} -> {veto['tp']} (a true bounce was lost!)")
    assert len(veto["fn"]) == len(base["fn"]), "FN grew — recall regressed"
    # the f1609 racket-hit FP is killed
    assert 1609 not in veto["pred"], "f1609 (a racket hit) survived the veto"
    assert len(veto["fp"]) < len(base["fp"]), "no FP was removed"


if __name__ == "__main__":
    failures = []
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"FAIL {t.__name__}: {e}")

    r = evaluate()
    print("\n── end-to-end (felix WASB native 1080p) ──")
    print(f"tol = {r['tol']} frames")
    print(f"baseline: pred={sorted(r['base']['pred'])}")
    print(f"          TP={r['base']['tp']} FP={r['base']['fp']} "
          f"FN={r['base']['fn']} F1={r['base']['f1']:.4f}")
    print(f"veto:     pred={sorted(r['veto']['pred'])}")
    print(f"          TP={r['veto']['tp']} FP={r['veto']['fp']} "
          f"FN={r['veto']['fn']} F1={r['veto']['f1']:.4f}")
    print(f"rejected as hits: {sorted(r['rejected'])}")
    print(f"\nFLOOR (post-veto) = {F1_FLOOR_POST_VETO}")
    print("ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    sys.exit(0 if not failures else 1)
