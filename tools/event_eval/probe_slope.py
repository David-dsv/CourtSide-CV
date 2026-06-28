"""S3 probe harness — measure a ROBUST _veto_slope variant end-to-end on EVERY
guardrail at once, by monkeypatching vision.bounce._veto_slope before import-time
binding leaks. The variant is selected by name on argv[1]; argv[2] is the
canonical methodo dump.

Reports, for the chosen slope variant:
  - canonical live_pool replay: conf H->B / B->H, bounce F1, hit F1, spurious
  - felix Kalman bounce F1 (no-arg + PROD turn-pass)   [floor 0.72]
  - felix WASB bounce F1                                [floor 0.80]
  - demo3 event-confusion cache: conf + bounce/hit F1   [conf 0/0, B>=0.80 H>=0.60]
  - vx_veto unit suite pass/fail                        [F1 0.9143, rejects 1609]

This is the single thermometer used to pick the retained variant. It patches the
function object on the module so BOTH vision.bounce.vx_flip_veto AND
vision.events._direction_features (which imported the name) see the new slope —
events.py does `from vision.bounce import _veto_slope`, binding a module-level
name, so we patch the name in BOTH modules.

Run:
  /Users/vuong/Documents/CourtSide-CV/venv/bin/python tools/event_eval/probe_slope.py \
      <variant> tests/fixtures/methodo/demo3_methodo_inputs_fullvideo.json
"""
import sys
import io
import json
import contextlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import vision.bounce as B  # noqa: E402

_ORIG = B._veto_slope


# ───────────────────────── candidate robust slopes ─────────────────────────
def _lstsq(pts, axis):
    """The current (non-robust) baseline."""
    return _ORIG(pts, axis)


def _theilsen(pts, axis):
    """Median of pairwise (frame-spaced) slopes — fully outlier-robust."""
    if len(pts) < 2:
        return None
    ks = [p[0] for p in pts]
    vs = [p[axis] for p in pts]
    slopes = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dk = ks[j] - ks[i]
            if dk != 0:
                slopes.append((vs[j] - vs[i]) / dk)
    return float(np.median(slopes)) if slopes else None


def _despike_then_lstsq(pts, axis, k=1):
    """Median-window (±k) despike of the fitted axis, then plain lstsq.
    Kills a single jitter outlier without changing the slope on a clean arc."""
    if len(pts) < 3:
        return _ORIG(pts, axis)
    vals = [p[axis] for p in pts]
    cleaned = []
    for i in range(len(pts)):
        lo, hi = max(0, i - k), min(len(pts), i + k + 1)
        med = float(np.median(vals[lo:hi]))
        cleaned.append((pts[i][0], med if axis == 1 else pts[i][1],
                        med if axis == 2 else pts[i][2]))
    return _ORIG(cleaned, axis)


def _huber_irls(pts, axis, k_mad=2.5, iters=3):
    """IRLS with MAD-scaled residual rejection (matches _robust_fit's spirit)."""
    if len(pts) < 2:
        return None
    ks = np.array([p[0] for p in pts], float)
    vs = np.array([p[axis] for p in pts], float)
    mask = np.ones(len(ks), bool)
    a = None
    for _ in range(iters):
        if mask.sum() < 2:
            break
        A = np.vstack([ks[mask], np.ones(mask.sum())]).T
        (a, b), *_ = np.linalg.lstsq(A, vs[mask], rcond=None)
        resid = vs - (a * ks + b)
        med = np.median(resid[mask])
        mad = np.median(np.abs(resid[mask] - med)) or 1e-6
        new = np.abs(resid - med) <= k_mad * 1.4826 * mad
        if new.sum() < 2 or np.array_equal(new, mask):
            break
        mask = new
    if a is None:
        A = np.vstack([ks, np.ones(len(ks))]).T
        (a, _b), *_ = np.linalg.lstsq(A, vs, rcond=None)
    return float(a)


def _theilsen_guarded(pts, axis, min_pts=3):
    """Theil-Sen only when there are >=min_pts (>=3 pairwise slopes); else lstsq.
    A 2-point side has a single slope == lstsq, so no behavior change there; the
    robustness only kicks in when there is enough support to vote out an outlier."""
    if len(pts) < min_pts:
        return _ORIG(pts, axis)
    return _theilsen(pts, axis)


def _conditional_despike(pts, axis, k=1, mad_mult=3.5):
    """Despike ONLY the points whose |Δ to local median| exceeds mad_mult*MAD of
    the per-frame steps — leaves a clean arc byte-identical, edits only a true
    outlier. Then plain lstsq. The least invasive robustifier."""
    if len(pts) < 4:
        return _ORIG(pts, axis)
    vals = np.array([p[axis] for p in pts], float)
    steps = np.abs(np.diff(vals))
    mad = np.median(np.abs(steps - np.median(steps))) or 1e-6
    thr = np.median(steps) + mad_mult * 1.4826 * mad
    cleaned = list(pts)
    for i in range(len(pts)):
        lo, hi = max(0, i - k), min(len(pts), i + k + 1)
        med = float(np.median(vals[lo:hi]))
        if abs(vals[i] - med) > thr:                     # only a genuine spike
            cleaned[i] = (pts[i][0], med if axis == 1 else pts[i][1],
                          med if axis == 2 else pts[i][2])
    return _ORIG(cleaned, axis)


VARIANTS = {
    "lstsq": _lstsq,
    "theilsen": _theilsen,
    "theilsen_g3": _theilsen_guarded,
    "despike1": lambda p, a: _despike_then_lstsq(p, a, k=1),
    "despike2": lambda p, a: _despike_then_lstsq(p, a, k=2),
    "conddespike": _conditional_despike,
    "huber": _huber_irls,
}


_CURRENT_VARIANT = "lstsq"


def patch(variant):
    global _CURRENT_VARIANT
    _CURRENT_VARIANT = variant
    fn = VARIANTS[variant]
    B._veto_slope = fn
    # events.py did `from vision.bounce import _veto_slope` -> rebind there too
    import vision.events as E
    E._veto_slope = fn


# ───────────────────────── measurements ─────────────────────────
def canonical(path):
    import importlib
    import tools.event_eval.live_pool as LP
    importlib.reload(LP)  # re-bind its `from vision.bounce import` to the patched fn
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = LP.report(path)
    res, s = r["res"], r["scores"]
    return {
        "confHtoB": res["conf_HtoB"], "confBtoH": res["conf_BtoH"],
        "bF1": s["bounce"]["F1"], "hF1": s["hit"]["F1"],
        "pool": r["pool_recall"], "spurious": res["spurious"],
        "bmiss": res["FN_b"], "hmiss": res["FN_h"],
    }


def _run_test_module(modpath):
    """Import a tests/*.py module fresh and capture its asserts; return (ok, tail)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("t_" + Path(modpath).stem, modpath)
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    ok = True
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)
            # re-bind the patched slope in THIS freshly-imported module's deps
            patch(_CURRENT_VARIANT)
            tests = [getattr(mod, nm) for nm in dir(mod)
                     if nm.startswith("test_") and callable(getattr(mod, nm))]
            if not tests:
                buf.write("\n(no test_* functions found)")
                ok = False
            for t in tests:
                t()
    except SystemExit as e:
        ok = (e.code in (0, None))
    except AssertionError as e:
        ok = False
        buf.write(f"\nASSERT FAIL: {e}")
    except Exception as e:  # noqa
        ok = False
        buf.write(f"\nERROR: {type(e).__name__}: {e}")
    tail = "\n".join(l for l in buf.getvalue().splitlines()
                     if "INFO" not in l and "loguru" not in l)[-600:]
    return ok, tail


def main():
    variant = sys.argv[1]
    path = sys.argv[2]
    patch(variant)
    print(f"\n################ VARIANT = {variant} ################")
    c = canonical(path)
    print(f"[CANONICAL live_pool] conf H->B={c['confHtoB']} B->H={c['confBtoH']} | "
          f"bounce F1={c['bF1']:.3f} | hit F1={c['hF1']:.3f} | pool {c['pool']}/17")
    print(f"  spurious={c['spurious']}  bmiss={c['bmiss']} hmiss={c['hmiss']}")

    for label, modfile in (
        ("felix Kalman  [>=0.72]", "tests/test_bounce_regression.py"),
        ("felix WASB    [>=0.80]", "tests/test_bounce_wasb_regression.py"),
        ("demo3 conf    [0/0]   ", "tests/test_event_confusion_regression.py"),
        ("vx_veto unit          ", "tests/test_vx_veto.py"),
    ):
        ok, tail = _run_test_module(str(ROOT / modfile))
        last = [l for l in tail.splitlines() if l.strip()]
        summ = last[-3:] if last else []
        print(f"\n[{label}] {'PASS' if ok else '*** FAIL ***'}")
        for l in summ:
            print(f"    {l}")


if __name__ == "__main__":
    main()
