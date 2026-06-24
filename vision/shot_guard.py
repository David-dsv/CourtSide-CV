"""
Court-visibility gate (shot-guard).

On broadcast tennis footage the pipeline used to detect balls / shots / poses /
bounces on EVERY frame — including camera close-ups (player zoom), replays and
graphics, where the full court is NOT visible. Those detections are noise
attributed to a player who isn't really playing, which corrupts the stats.

This module answers one cheap, colour-independent question per frame:

    "Is the full court visible?"

Signal: a wide court shot shows the court LINES — many long straight segments.
A close-up shows almost none. We count long straight lines (HoughLinesP on
Canny edges); a frame is "court visible" when that count is >= a threshold.
This is colour-independent (works on green clay, blue hard, anything) and
zero-hardcoding (the minimum line length is a fraction of the frame width).

Validated empirically:
  - tennis.mp4 (broadcast, with close-ups/replays): bimodal — non-court frames
    give <40 lines, court frames give 60-220, with a clean gap at 40-60.
  - felix.mp4 and the Djokovic practice clip (static court, never a close-up):
    always >=60 lines → zero false negatives.

The raw signal is noisy at transitions (single-frame dips below threshold), so
``compute_court_visibility_mask`` applies dwell hysteresis: a frame is only
NON-court when the low state has persisted for >= ``dwell`` consecutive frames
(~0.3 s). Short dips stay "court". This is what the pipeline consumes.

Pure numpy + cv2, no video/CLI/IO — unit-testable on synthetic frames, like
``vision/bounce.py`` and ``vision/player_track.py``.
"""
from __future__ import annotations

import cv2
import numpy as np

# Default threshold on the long-line count. Calibrated on the observed bimodal
# gap (non-court <40, court 60-220) — documented in the emitted stats JSON so
# the choice is reproducible/auditable, not a hidden magic number.
DEFAULT_LINE_THRESHOLD = 40
# A court line spans a meaningful fraction of the frame width. Expressed as a
# ratio (zero-hardcoding) so it scales with resolution.
DEFAULT_MIN_LINE_FRAC = 0.15


def frame_court_visible(frame, min_line_frac=DEFAULT_MIN_LINE_FRAC,
                        line_threshold=DEFAULT_LINE_THRESHOLD,
                        canny_lo=80, canny_hi=180, hough_threshold=80,
                        max_line_gap=20):
    """Raw per-frame court-visibility signal.

    True when the frame contains >= ``line_threshold`` long straight segments
    (HoughLinesP on Canny edges) — i.e. the court lines are visible, so the
    full court is in frame. ``minLineLength`` is ``min_line_frac * frame_width``
    (resolution-independent). Returns False for a degenerate/empty frame.

    Colour-independent: a green clay court, a blue hard court and a graphic all
    differ in hue but a wide shot is the only one full of long straight lines.
    """
    if frame is None or frame.size == 0:
        return False
    w = frame.shape[1]
    if w < 4:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold,
                            minLineLength=int(w * min_line_frac),
                            maxLineGap=max_line_gap)
    n = 0 if lines is None else len(lines)
    return n >= line_threshold


def _apply_hysteresis(raw_mask, dwell):
    """Turn a noisy raw boolean mask into a cleaned one with dwell hysteresis.

    A frame is NON-court only when the raw signal has been low (False) for >=
    ``dwell`` CONSECUTIVE frames. Short low-runs (incl. 1-frame transition
    dips) are gap-filled back to court (True). This matches the spec: a court
    doesn't vanish for a single frame, so single-frame dips are noise."""
    raw = np.asarray(raw_mask, dtype=bool)
    n = len(raw)
    out = np.ones(n, dtype=bool)  # default: court visible
    i = 0
    while i < n:
        if not raw[i]:
            j = i
            while j < n and not raw[j]:
                j += 1
            if (j - i) >= dwell:
                out[i:j] = False  # persistent low → truly non-court
            # else: short dip → stays True (court)
            i = j
        else:
            i += 1
    return out.tolist()


def compute_court_visibility_mask(frames, fps, min_line_frac=DEFAULT_MIN_LINE_FRAC,
                                  line_threshold=DEFAULT_LINE_THRESHOLD,
                                  dwell_s=0.3):
    """Cleaned per-frame court-visibility mask over a frame sequence.

    Runs ``frame_court_visible`` per frame, then dwell-hysteresis (``dwell =
    round(dwell_s * fps)`` frames). Returns a list[bool] aligned with the input.
    ``dwell`` is an fps-fraction (zero-hardcoding): ~0.3 s — shorter than any
    real close-up/replay, longer than a transition flicker."""
    dwell = max(1, int(round(dwell_s * fps)))
    raw = [frame_court_visible(f, min_line_frac=min_line_frac,
                               line_threshold=line_threshold)
           for f in frames]
    return _apply_hysteresis(raw, dwell)


def non_court_spans(mask):
    """Inclusive [start, end] runs of False (non-court) in a cleaned mask.
    Returned as a list of [start, end] pairs (JSON-friendly) for the stats."""
    spans = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            spans.append([int(i), int(j - 1)])
            i = j
        else:
            i += 1
    return spans

