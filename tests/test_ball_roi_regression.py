"""Guardrail for the SHIPPED dynamic-ROI ball-fallback defaults (fast, no GPU/video).

The dynamic ROI is a search-region fallback: on a frame where full-frame detection
finds no in-gate ball but the Kalman has a prediction, it crops around the prediction,
upscales, re-detects, and feeds the (tightly-gated) hit back into the Kalman. The
DANGER it guards against: an off-target ROI hit poisons the Kalman state and degrades
CORRECT (the accuracy metric) even while it lifts coverage — so this test asserts the
SHIPPED params give a CLEAN win on the committed demo3 ROI replay cache:

  - coverage and CORRECT both >= the no-ROI baseline (never a net regression), AND
  - the shipped params match the validated clean-win config (so a silent default
    drift toward the unguarded/narrow poisoning config trips here).

It reads the LIVE argparse defaults from run_pipeline_8s.py, so it can never drift
from prod. The replay cache mirrors the exact prod interleaved-Kalman path (built by
tools/ball_gt/roi_lab.py --build) and was confirmed against a full LIVE pipeline run.

Run:  venv/bin/python tests/test_ball_roi_regression.py
"""
import sys
import re
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "tools" / "ball_gt" / "_roi_lab_cache.json"

# load roi_lab as a module (run_kalman / score / load_gt mirror prod Pass 1b)
_spec = importlib.util.spec_from_file_location(
    "_roi_lab", str(ROOT / "tools" / "ball_gt" / "roi_lab.py"))
_rl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rl)


def shipped(arg):
    """Read a --<arg> float default straight from run_pipeline_8s.py."""
    src = (ROOT / "run_pipeline_8s.py").read_text()
    m = re.search(rf'--{arg}",\s*type=float,\s*default=([0-9.]+)', src)
    assert m, f"could not find --{arg} default in run_pipeline_8s.py"
    return float(m.group(1))


def measure():
    c = json.loads(CACHE.read_text())
    fw, fh, fps = c["fw"], c["fh"], c["fps"]
    per_frame_ff = c["per_frame_ff"]
    gtbf = _rl.load_gt()

    half_frac = shipped("ball-roi-half-frac")
    zoom = shipped("ball-roi-zoom")
    gate_frac = shipped("ball-roi-gate-frac")
    min_conf = shipped("ball-roi-min-conf")

    base = _rl.score(_rl.run_kalman(per_frame_ff, fw, fh, fps)[0], gtbf, fw, fps)

    # find the cached ROI grid closest to the shipped window/zoom (the cache is built
    # on a discrete grid; the shipped half_frac/zoom must be one of the cached keys).
    key = None
    for hf in c["half_fracs"]:
        for z in c["zooms"]:
            if abs(hf - half_frac) < 1e-3 and abs(z - zoom) < 1e-3:
                key = f"{hf:.4f}|{z:.2f}"
    assert key is not None, (
        f"shipped ROI window/zoom ({half_frac}, {zoom}) is not in the replay cache "
        f"grid {c['half_fracs']} x {c['zooms']}; rebuild tools/ball_gt/roi_lab.py "
        f"--build to include it.")
    rc = c["roi_cache"][key]

    def roi_fn(i, px, py):
        return rc.get(str(i))
    track, recovered = _rl.run_kalman(per_frame_ff, fw, fh, fps, roi_fn=roi_fn,
                                      roi_gate_frac=gate_frac, roi_min_conf=min_conf)
    roi = _rl.score(track, gtbf, fw, fps)
    return base, roi, len(recovered), (half_frac, zoom, gate_frac, min_conf)


def test_ball_roi_clean_win():
    base, roi, recovered, params = measure()
    # never a NET regression on either metric (the poisoning trap)
    assert roi["corr"] >= base["corr"], (
        f"shipped ROI params {params} REGRESS demo3 CORRECT "
        f"{roi['corr']} < baseline {base['corr']} — the ROI is poisoning the Kalman "
        f"state (off-target recovery). Tighten --ball-roi-gate-frac / raise "
        f"--ball-roi-min-conf / widen --ball-roi-half-frac. See "
        f"docs/research/ball-roi-dynamic-CR.md.")
    assert roi["cov"] >= base["cov"], (
        f"shipped ROI params {params} REGRESS demo3 coverage "
        f"{roi['cov']} < baseline {base['cov']}.")
    # and it must actually do SOMETHING (a positive net, else the feature is inert)
    assert roi["corr"] + roi["cov"] > base["corr"] + base["cov"], (
        f"shipped ROI params {params} are a no-op (cov {roi['cov']}=={base['cov']}, "
        f"CORRECT {roi['corr']}=={base['corr']}) — the ROI recovered {recovered} "
        f"frames but netted zero. Defaults may have drifted to an over-tight config.")


if __name__ == "__main__":
    base, roi, recovered, params = measure()
    print(f"shipped ROI params (half_frac,zoom,gate_frac,min_conf) = {params}")
    print(f"baseline:  cov {base['cov']}/{base['n']}  CORRECT {base['corr']}/{base['n']}")
    print(f"+ ROI   :  cov {roi['cov']}/{roi['n']}  CORRECT {roi['corr']}/{roi['n']}  "
          f"(recovered {recovered})")
    print(f"Δcov {roi['cov']-base['cov']:+d}  Δcorr {roi['corr']-base['corr']:+d}")
    try:
        test_ball_roi_clean_win()
        print("PASS")
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
