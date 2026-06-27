"""THROWAWAY: inspect the posed cache. For each demo3 GT frame, show the TRUE far
player (GT center) vs every cached candidate (cx,cy,asp,conf,valid,h), flag which
one centrality picks and which one the GT is, so we can SEE what separates the real
(invalid,central,on-court) player from the (valid,higher) spectator that beats it.

  python tools/pose_gt/_probe_cache.py [demo|felix] [n_frames]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
FELIX_GT = "/tmp/felix_far_gt.json"
CACHE = json.load(open("/tmp/strat_cache.json"))

clip = sys.argv[1] if len(sys.argv) > 1 else "demo"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

if clip == "demo":
    cache = {int(k): v for k, v in CACHE["demo"].items()}
    fw, ny = CACHE["dfw"], CACHE["dny"]
    g = json.load(open(DEMO_GT))
    gtbf = {f["frame"]: f["p1_far"]["center"] for f in g["frames"]
            if f["p1_far"].get("verified") and f["p1_far"].get("center")}
else:
    cache = {int(k): v for k, v in CACHE["felix"].items()}
    fw, ny = CACHE["ffw"], CACHE["fny"]
    gtbf = {int(k): v for k, v in json.load(open(FELIX_GT)).items()}

fh = 2 * ny


def geom(r):
    central = 1.0 - abs(r["cx"] / fw - 0.5) * 2.0
    human = 1.0 if r["asp"] <= 2.2 else max(0.0, 1.0 - (r["asp"] - 2.2) * 0.5)
    s = central + 0.5 * human + 0.2 * r["conf"]
    knee = 0.25 * ny
    if r["cy"] < knee:
        s -= 1.0 * (1.0 - r["cy"] / knee)
    return s


print(f"{clip}: fw={fw:.0f} fh={fh:.0f} ny={ny:.0f} knee(0.25ny)={0.25*ny:.0f}\n")
shown = 0
for fidx, (gx, gy) in sorted(gtbf.items()):
    rows = cache.get(fidx, [])
    if not rows:
        continue
    shown += 1
    if shown > N:
        break
    pick = max(rows, key=geom)
    print(f"f{fidx}: GT=({gx:.0f},{gy:.0f})  cands={len(rows)}")
    for r in sorted(rows, key=geom, reverse=True):
        d = ((r["cx"] - gx) ** 2 + (r["cy"] - gy) ** 2) ** 0.5
        is_gt = "  <-GT" if d < 0.10 * fw else ""
        is_pick = " [PICK]" if r is pick else ""
        print(f"    cx={r['cx']:6.0f} cy={r['cy']:5.0f} xr={r['cx']/fw:.2f} "
              f"asp={r['asp']:4.1f} conf={r['conf']:.2f} valid={r['valid']} "
              f"h={r['h']:5.0f}({r['h']/fh:.2f}H) geom={geom(r):+.2f} "
              f"d={d:5.0f}{is_gt}{is_pick}")
    print()
