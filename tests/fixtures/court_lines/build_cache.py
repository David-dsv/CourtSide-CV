"""
Build the committed court-line cache fixtures for tests/test_court_regression.py.

For each probe clip we dump the WHITE-LINE MASK of the first frame as a small PNG
(+ a tiny JSON of dims). The regression test reconstructs the line candidates and
distance transform from the mask and exercises the FULL fit+gate path — no video
decode, no torch, no HSV color step — so it runs in well under a second.

The mask is a DERIVED artifact (binary court lines, not the source frame), so
committing it for the third-party Djokovic clip carries no footage-rights issue;
felix/tennis are our own clips. Re-run only when the mask front end changes:

    venv/bin/python tests/fixtures/court_lines/build_cache.py
"""
import sys
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.court_lines import white_line_mask  # noqa: E402

# clips live in the sibling main repo (not committed here)
SRC = Path("/Users/vuong/Documents/CourtSide-CV")
CLIPS = {
    "tennis":   (SRC / "tennis.mp4", "elevated_oblique"),
    "felix":    (SRC / "felix.mp4", "grazing"),
    "djokovic": (SRC / "Djokovic vs Ruud Practice  Court Level Tennis 4K60.mp4", "court_level"),
}
OUT = Path(__file__).resolve().parent


def main():
    for name, (path, difficulty) in CLIPS.items():
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"SKIP {name}: could not read first frame")
            continue
        h, w = frame.shape[:2]
        mask = white_line_mask(frame)
        cv2.imwrite(str(OUT / f"{name}_mask.png"), mask)
        (OUT / f"{name}_meta.json").write_text(json.dumps(
            {"clip": name, "difficulty": difficulty, "frame_wh": [w, h]}, indent=2))
        print(f"wrote {name}: {w}x{h}, mask coverage {(mask>0).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
