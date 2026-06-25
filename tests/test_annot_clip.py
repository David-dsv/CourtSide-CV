"""Unit tests for vision.annot_clip — the annotation clip-cut mapping.

No video, no model, no GPU: every function is pure list/dict arithmetic over a
court-visibility mask, so the tests are deterministic and run in milliseconds.
Run directly or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision.annot_clip import (
    annotated_to_original,
    build_clip_index,
    frames_to_annotate,
    original_to_annotated,
    write_clip_index,
)

FPS = 50.0


# ───────────────────────── all-True (no cut) ──────────────────────────────

def test_all_true_is_identity():
    """A fully court-visible window keeps every frame: kept == range(n),
    no cut spans, and the mapping is the identity (offset by start_frame)."""
    n = 12
    mask = [True] * n
    start = 0

    assert frames_to_annotate(mask) == list(range(n))

    ci = build_clip_index(mask, FPS, start_frame=start)
    assert ci["original_total"] == n
    assert ci["annotated_total"] == n
    assert ci["kept_original_frames"] == list(range(n))
    assert ci["cut_spans"] == []
    assert ci["start_frame"] == start

    # identity round-trip on every frame
    for f in range(n):
        a = original_to_annotated(ci, f)
        assert a == f, (f, a)
        assert annotated_to_original(ci, a) == f
    print("PASS test_all_true_is_identity")


def test_all_true_with_start_offset():
    """With a non-zero start_frame the kept frames are ABSOLUTE original
    indices (start_frame + i); annotated indices remain 0-based."""
    n = 5
    mask = [True] * n
    start = 100
    ci = build_clip_index(mask, FPS, start_frame=start)
    assert ci["kept_original_frames"] == [100, 101, 102, 103, 104]
    assert ci["start_frame"] == 100
    # annotated 0 → original 100; original 102 → annotated 2
    assert annotated_to_original(ci, 0) == 100
    assert original_to_annotated(ci, 102) == 2
    # an original frame BEFORE the window is not kept → None
    assert original_to_annotated(ci, 99) is None
    print("PASS test_all_true_with_start_offset")


# ───────────────────────── mask with a hole (cut) ─────────────────────────

def test_mask_with_hole():
    """A non-court hole is cut out: those frames are absent from the annotated
    video, annotated_total drops by the hole size, and the round-trip is
    consistent on kept frames / None on cut frames."""
    # frames:        0    1    2    3    4    5    6    7    8
    mask = [True, True, False, False, False, True, True, False, True]
    start = 0
    n = len(mask)
    cut = {2, 3, 4, 7}
    kept = [i for i in range(n) if i not in cut]  # [0,1,5,6,8]

    assert frames_to_annotate(mask) == kept

    ci = build_clip_index(mask, FPS, start_frame=start)
    assert ci["original_total"] == n
    assert ci["annotated_total"] == len(kept) == 5
    assert ci["kept_original_frames"] == kept
    # non_court_spans → inclusive runs of False
    assert ci["cut_spans"] == [[2, 4], [7, 7]]

    # round-trip original_to_annotated ∘ annotated_to_original on kept frames
    for f in kept:
        a = original_to_annotated(ci, f)
        assert a is not None, f
        assert annotated_to_original(ci, a) == f, (f, a)

    # the inverse direction too: every annotated index maps back to a kept frame
    for a in range(ci["annotated_total"]):
        f = annotated_to_original(ci, a)
        assert original_to_annotated(ci, f) == a, (a, f)

    # cut frames map to None
    for f in cut:
        assert original_to_annotated(ci, f) is None, f

    # cut frames are genuinely absent from the annotated stream
    assert all(f not in ci["kept_original_frames"] for f in cut)
    print("PASS test_mask_with_hole")


def test_mask_with_hole_and_start_offset():
    """The hole-cut mapping with a window offset: kept frames are absolute,
    cut_spans stay window-relative, and start_frame bridges the two."""
    mask = [True, False, False, True]
    start = 60
    ci = build_clip_index(mask, FPS, start_frame=start)
    assert ci["kept_original_frames"] == [60, 63]
    assert ci["cut_spans"] == [[1, 2]]  # window-relative
    assert ci["start_frame"] == 60
    # absolute frames 61,62 are inside the cut span → None
    assert original_to_annotated(ci, 61) is None
    assert original_to_annotated(ci, 62) is None
    # cut_spans are window-relative: span[0]+start_frame = absolute cut start
    s, e = ci["cut_spans"][0]
    assert original_to_annotated(ci, start + s) is None
    assert original_to_annotated(ci, start + e) is None
    print("PASS test_mask_with_hole_and_start_offset")


# ───────────────────────── boundary: empty mask ───────────────────────────

def test_empty_mask():
    """An empty window yields an empty, well-formed clip-index."""
    mask = []
    assert frames_to_annotate(mask) == []
    ci = build_clip_index(mask, FPS, start_frame=7)
    assert ci["original_total"] == 0
    assert ci["annotated_total"] == 0
    assert ci["kept_original_frames"] == []
    assert ci["cut_spans"] == []
    assert ci["start_frame"] == 7
    # nothing to look up; original lookups all return None
    assert original_to_annotated(ci, 0) is None
    assert original_to_annotated(ci, 7) is None
    print("PASS test_empty_mask")


# ───────────────────────── boundary: all-False ────────────────────────────

def test_all_false_mask():
    """A fully non-court window cuts EVERYTHING: nothing annotated, one big
    cut span, every original frame maps to None."""
    n = 6
    mask = [False] * n
    start = 0
    assert frames_to_annotate(mask) == []
    ci = build_clip_index(mask, FPS, start_frame=start)
    assert ci["original_total"] == n
    assert ci["annotated_total"] == 0
    assert ci["kept_original_frames"] == []
    assert ci["cut_spans"] == [[0, n - 1]]
    for f in range(n):
        assert original_to_annotated(ci, f) is None, f
    print("PASS test_all_false_mask")


# ───────────────────────── annotated_to_original bounds ───────────────────

def test_annotated_to_original_out_of_range_raises():
    """annotated_to_original raises on an out-of-range annotated index (every
    annotated frame is kept by construction, so a miss is a real bug)."""
    ci = build_clip_index([True, False, True], FPS, start_frame=0)
    assert ci["annotated_total"] == 2
    for bad in (-1, 2, 99):
        raised = False
        try:
            annotated_to_original(ci, bad)
        except IndexError:
            raised = True
        assert raised, f"expected IndexError for annotated_frame={bad}"
    print("PASS test_annotated_to_original_out_of_range_raises")


# ───────────────────────── write_clip_index (I/O) ─────────────────────────

def test_write_clip_index_sidecar(tmp_path=None):
    """write_clip_index writes <output without ext>_clipindex.json next to the
    output and returns the same dict build_clip_index would. Uses a temp dir."""
    import json
    import tempfile

    mask = [True, True, False, True]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "felix_annotated.mp4")
        path, ci = write_clip_index(out, mask, FPS, start_frame=10)
        assert path == os.path.join(d, "felix_annotated_clipindex.json")
        assert os.path.exists(path)
        with open(path) as f:
            on_disk = json.load(f)
        assert on_disk == ci
        assert ci["kept_original_frames"] == [10, 11, 13]
        assert ci["cut_spans"] == [[2, 2]]
    print("PASS test_write_clip_index_sidecar")


if __name__ == "__main__":
    test_all_true_is_identity()
    test_all_true_with_start_offset()
    test_mask_with_hole()
    test_mask_with_hole_and_start_offset()
    test_empty_mask()
    test_all_false_mask()
    test_annotated_to_original_out_of_range_raises()
    test_write_clip_index_sidecar()
    print("\nAll annot-clip tests passed.")
