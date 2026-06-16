"""
Shared schema + I/O for bounce ground-truth and prediction files.

Both the annotation helper (annotate_bounces.py) and the evaluation script
(eval_bounces.py) import from here so the on-disk format is defined in exactly
one place and can never drift between writer and reader.

File format (JSON, one file per clip)
-------------------------------------
{
  "video": "data/clips/match1.mp4",   # source video path (informational)
  "fps": 30.0,                          # source frame rate
  "frame_offset": 0,                    # absolute frame of this clip's frame 0
  "annotator": "david",                 # who/what produced this file
  "notes": "",
  "bounces": [
    {"frame": 142, "x": 980, "y": 612, "depth": "deep"},
    {"frame": 287}                       # x/y/depth all optional
  ]
}

Only `frame` is required per bounce. `x`, `y`, `depth` are optional.
`frame` is ABSOLUTE in the source video (frame 0 = first frame of the file).
`frame_offset` lets a pre-cut clip's frames map back to the source: a bounce's
absolute frame is `frame` as written; `frame_offset` records where the clip
started so predictions and ground truth can be aligned even if they were
produced over different windows.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

DEPTH_LABELS = ("deep", "mid", "short")


def make_bounce(frame: int,
                x: Optional[float] = None,
                y: Optional[float] = None,
                depth: Optional[str] = None) -> Dict:
    """Build one bounce record, omitting optional fields that are None."""
    rec: Dict = {"frame": int(frame)}
    if x is not None:
        rec["x"] = int(round(x))
    if y is not None:
        rec["y"] = int(round(y))
    if depth is not None:
        if depth not in DEPTH_LABELS:
            raise ValueError(f"depth must be one of {DEPTH_LABELS}, got {depth!r}")
        rec["depth"] = depth
    return rec


def save_bounce_file(path: str,
                     video: str,
                     fps: float,
                     bounces: List[Dict],
                     frame_offset: int = 0,
                     annotator: str = "",
                     notes: str = "") -> None:
    """
    Write a bounce file. Bounces are sorted by frame and de-duplicated so the
    file stays clean even after many edits.
    """
    seen = set()
    clean = []
    for b in sorted(bounces, key=lambda r: r["frame"]):
        if b["frame"] in seen:
            continue
        seen.add(b["frame"])
        clean.append(b)

    doc = {
        "video": video,
        "fps": float(fps),
        "frame_offset": int(frame_offset),
        "annotator": annotator,
        "notes": notes,
        "bounces": clean,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def load_bounce_file(path: str) -> Dict:
    """
    Load and lightly validate a bounce file. Returns the parsed dict with
    guaranteed keys: video, fps, frame_offset, bounces (list of records).
    """
    with open(path) as f:
        doc = json.load(f)

    if "bounces" not in doc or not isinstance(doc["bounces"], list):
        raise ValueError(f"{path}: missing or invalid 'bounces' list")

    doc.setdefault("video", "")
    doc.setdefault("fps", 0.0)
    doc.setdefault("frame_offset", 0)

    for b in doc["bounces"]:
        if "frame" not in b:
            raise ValueError(f"{path}: a bounce record is missing required 'frame'")
        b["frame"] = int(b["frame"])

    return doc


def absolute_frames(doc: Dict) -> List[int]:
    """
    Return the sorted list of absolute bounce frame numbers for a loaded doc.
    `frame` is already absolute; frame_offset is informational here but kept in
    the API so callers have a single place to reason about alignment.
    """
    return sorted(b["frame"] for b in doc["bounces"])
