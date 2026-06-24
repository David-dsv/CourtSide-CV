"""Unit tests for vision.shot_guard — the court-visibility gate.

No model, no video, no GPU: frames are synthesised with numpy + cv2 so the
tests run in well under a second and are deterministic. Run directly or via
pytest.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.shot_guard import (
    _apply_hysteresis,
    compute_court_visibility_mask,
    frame_court_visible,
    non_court_spans,
)

H, W = 720, 1280
_RNG = np.random.default_rng(0)


def _court_frame(n_lines=60, seed=0):
    """A synthetic WIDE court shot: a textured (noisy) background with many
    long bright lines, the way a real court shot is full of court lines. This
    is what HoughLinesP reliably detects (>40 lines)."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 50, (H, W, 3)).astype(np.uint8)  # court/crowd texture
    for _ in range(n_lines):
        y0 = int(rng.integers(0, H))
        y1 = int(np.clip(y0 + rng.integers(-90, 90), 0, H - 1))
        cv2.line(frame, (0, y0), (W, y1), (220, 220, 220), 3)
    return frame


def _closeup_frame():
    """A synthetic CLOSE-UP / non-court frame: a single big blob on a smooth
    background, no long straight lines. A player's torso filling the frame."""
    frame = np.full((H, W, 3), 30, np.uint8)  # smooth dark bg
    cv2.ellipse(frame, (W // 2, H // 2), (W // 4, H // 3), 0, 0, 360, (160, 120, 90), -1)
    cv2.GaussianBlur(frame, (15, 15), 0, dst=frame)  # smooth out any accidental edges
    return frame


# ───────────────────────── frame_court_visible ────────────────────────────

def test_closeup_is_non_court():
    assert frame_court_visible(_closeup_frame()) is False
    print("PASS test_closeup_is_non_court")


def test_wide_shot_is_court():
    assert frame_court_visible(_court_frame(n_lines=60)) is True
    print("PASS test_wide_shot_is_court")


def test_resolution_invariance():
    """The line-length gate must scale with frame width (zero-hardcoding): the
    same scene at half resolution must still classify the same way."""
    big = _court_frame(n_lines=60, seed=3)
    small = cv2.resize(big, (W // 2, H // 2))
    assert frame_court_visible(big) is True
    assert frame_court_visible(small) is True
    assert frame_court_visible(cv2.resize(_closeup_frame(), (W // 2, H // 2))) is False
    print("PASS test_resolution_invariance")


def test_min_line_length_frac_matters():
    """Tiny short segments (<< 15% of width) must NOT count as court lines,
    even when there are many of them — confirming the gate keys on LONG lines."""
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 40, (H, W, 3)).astype(np.uint8)
    for _ in range(80):
        x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
        # short segments (~5% of width) — too short to be a court line
        x1, y1 = x0 + int(W * 0.05), y0 + int(rng.integers(-20, 20))
        cv2.line(frame, (x0, y0), (x1, y1), (220, 220, 220), 3)
    assert frame_court_visible(frame) is False, "short segments should not read as court"
    print("PASS test_min_line_length_frac_matters")


# ───────────────────────── hysteresis ─────────────────────────────────────

def test_hysteresis_cleans_single_dip():
    """A 1-frame dip in the raw signal stays court (dwell not reached)."""
    out = _apply_hysteresis([1, 1, 1, 0, 1, 1, 1], dwell=3)
    assert out == [True, True, True, True, True, True, True], out
    print("PASS test_hysteresis_cleans_single_dip")


def test_hysteresis_keeps_persistent_low():
    """A low-run >= dwell becomes truly non-court."""
    out = _apply_hysteresis([1, 1, 0, 0, 0, 0, 1, 1], dwell=3)
    assert out == [True, True, False, False, False, False, True, True], out
    print("PASS test_hysteresis_keeps_persistent_low")


def test_hysteresis_dwell_boundary():
    """Exactly `dwell` low frames → non-court; `dwell-1` → stays court."""
    assert _apply_hysteresis([0, 0, 0], dwell=3) == [False, False, False]
    assert _apply_hysteresis([0, 0], dwell=3) == [True, True]
    print("PASS test_hysteresis_dwell_boundary")


# ───────────────────────── non_court_spans ────────────────────────────────

def test_non_court_spans():
    spans = non_court_spans([1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    assert spans == [[2, 4], [6, 9]], spans
    assert non_court_spans([1, 1, 1]) == []
    assert non_court_spans([0, 0]) == [[0, 1]]
    print("PASS test_non_court_spans")


# ───────────────────── compute_court_visibility_mask ──────────────────────

def test_compute_mask_end_to_end():
    """30 court frames, 40 close-up frames, 30 court frames → the cleaned mask
    is court / non-court / court with exactly one non-court span (after dwell
    hysteresis; the 40-frame close-up well exceeds the ~0.3 s dwell @50 fps)."""
    fps = 50
    frames = ([_court_frame(seed=i) for i in range(30)]
              + [_closeup_frame() for _ in range(40)]
              + [_court_frame(seed=100 + i) for i in range(30)])
    mask = compute_court_visibility_mask(frames, fps)
    assert len(mask) == len(frames)
    # first 30 court
    assert all(mask[:30]), mask[:30]
    # the close-up span (40 frames) is non-court (dwell=15 @50fps, 40>15)
    assert all(not m for m in mask[30:70]), mask[30:70]
    # last 30 court
    assert all(mask[70:]), mask[70:]
    spans = non_court_spans(mask)
    assert spans == [[30, 69]], spans
    print("PASS test_compute_mask_end_to_end")


def test_compute_mask_short_dip_stays_court():
    """A single close-up frame sandwiched in court frames stays court (dwell)."""
    fps = 50
    frames = ([_court_frame(seed=i) for i in range(20)]
              + [_closeup_frame()]
              + [_court_frame(seed=50 + i) for i in range(20)])
    mask = compute_court_visibility_mask(frames, fps)
    assert all(mask), "single-frame dip must be cleaned back to court"
    assert non_court_spans(mask) == []
    print("PASS test_compute_mask_short_dip_stays_court")


# ────────────────── upstream-null invariant (contract) ────────────────────

def test_detect_hits_emits_nothing_on_non_court_span():
    """Documents the pipeline contract: when ball + pose are nulled on a span
    of frames (what the shot-guard does on close-ups), detect_hits must not
    emit any hit in that span. We don't need the real detector — we assert the
    trivial sufficient condition: empty ball + empty pose ⇒ no peaks at all."""
    from vision.shots import detect_hits

    n = 60
    ball = [None] * n                      # ball nulled on every frame
    players = [[] for _ in range(n)]       # pose nulled on every frame
    hits = detect_hits(ball, players, fps=50, frame_height=H, frame_width=W)
    assert hits == [], hits
    print("PASS test_detect_hits_emits_nothing_on_non_court_span")


if __name__ == "__main__":
    test_closeup_is_non_court()
    test_wide_shot_is_court()
    test_resolution_invariance()
    test_min_line_length_frac_matters()
    test_hysteresis_cleans_single_dip()
    test_hysteresis_keeps_persistent_low()
    test_hysteresis_dwell_boundary()
    test_non_court_spans()
    test_compute_mask_end_to_end()
    test_compute_mask_short_dip_stays_court()
    test_detect_hits_emits_nothing_on_non_court_span()
    print("\nAll shot-guard tests passed.")
