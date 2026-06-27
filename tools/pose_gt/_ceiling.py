"""THROWAWAY: measure the SELECTION ceiling per clip from the posed cache: of the GT
frames that have ANY cached candidate, how many have at least one candidate within
0.10*fw of the GT center? That is the max any selector can reach (the rest are a
DETECTION gap, not a selection failure). Also reports the wc=1.2 hcv pick's score.

  python tools/pose_gt/_ceiling.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO_GT = ROOT / "tests" / "fixtures" / "pose_gt" / "tennis_demo3.pose_gt.json"
FELIX_GT = "/tmp/felix_far_gt.json"
C = json.load(open("/tmp/strat_cache.json"))


def ceiling(cache, gtbf, fw):
    n = reach = 0
    for fidx, (gx, gy) in gtbf.items():
        rows = cache.get(str(fidx), cache.get(fidx, []))
        if not rows:
            continue
        n += 1
        if any(((r["cx"] - gx) ** 2 + (r["cy"] - gy) ** 2) ** 0.5 < 0.10 * fw
               for r in rows):
            reach += 1
    return reach, n


dcache = {int(k): v for k, v in C["demo"].items()}
fcache = {int(k): v for k, v in C["felix"].items()}
dg = json.load(open(DEMO_GT))
dgtbf = {f["frame"]: f["p1_far"]["center"] for f in dg["frames"]
         if f["p1_far"].get("verified") and f["p1_far"].get("center")}
fgtbf = {int(k): v for k, v in json.load(open(FELIX_GT)).items()}

dr, dn = ceiling(dcache, dgtbf, C["dfw"])
fr, fn = ceiling(fcache, fgtbf, C["ffw"])
print(f"demo3 selection ceiling: {dr}/{dn} = {100*dr/dn:.1f}% "
      f"(the other {dn-dr} GT frames have NO candidate near the player -> detection gap)")
print(f"felix selection ceiling: {fr}/{fn} = {100*fr/fn:.1f}% "
      f"(other {fn-fr} = detection gap)")
