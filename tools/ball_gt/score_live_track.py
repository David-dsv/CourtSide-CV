"""Score a LIVE ball-track dump (from run_pipeline_8s.py --dump-ball-track) against
the demo3 ball GT — the honest end-to-end thermometer for the dynamic-ROI fallback.

  python run_pipeline_8s.py data/output/tennis_demo3.mp4 -s 0 -d 13 --device cpu \
      --dump-ball-track /tmp/live_roi.json
  python tools/ball_gt/score_live_track.py /tmp/live_roi.json

Metric matches tools/ball_gt/measure_ball_coverage.py: coverage (any ball tracked) +
CORRECT (within tol_frac*W of the GT point), overall + cold-start (first 4s).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "tests" / "fixtures" / "ball_gt" / "tennis_demo3.ball_gt.json"


def main():
    dump = json.loads(Path(sys.argv[1]).read_text())
    tol_frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    centers = dump["centers"]
    fw, fps = dump["width"], dump["fps"]
    start_frame = dump.get("start_frame", 0)
    tol = tol_frac * fw

    gt = json.loads(GT.read_text())
    gtbf = {f["frame"]: f["ball"] for f in gt["frames"]
            if f.get("verified") and f.get("visible", True) and f.get("ball")}

    start_n = int(4.0 * fps)
    cov = corr = n = covs = corrs = ns = 0
    for fidx, g in gtbf.items():
        # GT frames are absolute; the dump is window-relative to start_frame.
        di = fidx - start_frame
        if di < 0 or di >= len(centers):
            continue
        n += 1
        s = fidx < start_n
        ns += s
        t = centers[di]
        if t is not None:
            cov += 1
            covs += s
            if ((t[0] - g[0]) ** 2 + (t[1] - g[1]) ** 2) ** 0.5 < tol:
                corr += 1
                corrs += s
    print(f"=== LIVE ball track vs GT — {Path(sys.argv[1]).name} ===")
    print(f"coverage (any ball): {cov}/{n} = {100*cov/max(1,n):.1f}%")
    print(f"CORRECT (within {tol:.0f}px): {corr}/{n} = {100*corr/max(1,n):.1f}%")
    print(f"--- COLD START (first 4s, {ns} GT frames) ---")
    print(f"  coverage: {covs}/{ns} = {100*covs/max(1,ns):.1f}%")
    print(f"  CORRECT : {corrs}/{ns} = {100*corrs/max(1,ns):.1f}%")


if __name__ == "__main__":
    main()
