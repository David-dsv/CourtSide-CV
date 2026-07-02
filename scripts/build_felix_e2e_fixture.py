"""Freeze the felix END-TO-END event fixture from a live methodo-inputs dump.

The existing felix guards (test_bounce_regression / test_bounce_wasb_regression)
pin the DETECTOR layer on frozen detections. Nothing guarded the full chain the
user actually sees on the second camera angle — WASB track → candidate
generators → classify_events arbitration → final BOUNCE markers — which is
where the 2026-07-02 session measured felix at end-to-end F1 0.865 with recall
16/16 (was 0.83). This fixture freezes the EXACT live inputs of that chain
(`COURTSIDE_DUMP_METHODO_INPUTS` on `felix.mp4 -s 0 -d 31 --ball-tracker wasb`)
so tests/test_felix_e2e_regression.py can replay it in seconds, no GPU.

Only the replay-necessary keys are kept (the raw_ppf/hits/bset diagnostics are
dropped — the test re-derives every candidate set with the CURRENT code, like
the demo3 confusion test, so generator changes are exercised too).

Run:  venv/bin/python scripts/build_felix_e2e_fixture.py <methodo_dump.json>
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "methodo" / "felix_e2e_inputs.json"


def main(dump_path):
    d = json.loads(Path(dump_path).read_text())
    fixture = {
        "video": "felix.mp4 -s 0 -d 31 --ball-tracker wasb (live canonical run)",
        "fps": d["fps"], "width": d["width"], "height": d["height"],
        "raw_ball_centers": d["raw_ball_centers"],
        "ball_is_real": d["ball_is_real"],
        "all_ball_centers": d["all_ball_centers"],
        "methodo_ppf": d["methodo_ppf"],
    }
    OUT.write_text(json.dumps(fixture))
    n = len(d["raw_ball_centers"])
    print(f"wrote {OUT}: {n} frames, "
          f"ball {sum(1 for c in d['raw_ball_centers'] if c)}/{n}, "
          f"pose>=2 {sum(1 for s in d['methodo_ppf'] if len(s) >= 2)}/{n}, "
          f"{OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
