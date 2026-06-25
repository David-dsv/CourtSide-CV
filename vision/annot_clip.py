"""
Annotation clip-cut: turn a court-visibility mask into a real frame *cut*.

``vision/shot_guard.py`` answers "is the full court visible?" per frame and
yields a cleaned boolean mask. The *shot-guard* integration on its own branch
KEPT every frame (timeline 1:1) and merely drew "clean" on the non-court ones.

This module implements the decision taken by the PM instead: **actually CUT the
non-court frames out of the ANNOTATED video** (replays / close-ups / graphics
disappear), so the annotation loop spends **zero CPU** on out-of-play frames.
The annotated output is therefore SHORTER than the source — the frame↔time
mapping changes — so we also emit a **clip-index** that maps every annotated
frame back to its original frame for front-end re-alignment (consumed by S5).

The original video is never rewritten: the pipeline only READS the input and
writes a *new* annotated file. This module never touches video at all — it is
pure list/dict arithmetic over the mask, so it is fully unit-testable without a
video/model/GPU (like ``vision/bounce.py``, ``vision/shot_guard.py``).

Index conventions (IMPORTANT — read before wiring):
  * The mask is computed over the *processed window* only, so mask index ``i``
    is RELATIVE to the window (``i`` in ``0..len(mask)-1``).
  * ``start_frame`` is the absolute frame index of the window's first frame
    (``int(args.start * fps)`` in the pipeline). The ORIGINAL (absolute) frame
    index of mask position ``i`` is therefore ``start_frame + i``.
  * ``kept_original_frames`` holds ABSOLUTE original indices (``start_frame``
    already added) so the front can re-align straight onto the source-video
    timeline. ``cut_spans`` stay window-RELATIVE (they mirror
    ``non_court_spans(mask)`` 1:1), and ``start_frame`` is recorded so either
    can be converted to the other.
"""
from __future__ import annotations

from vision.shot_guard import non_court_spans


def frames_to_annotate(mask):
    """Ordered list of mask indices to annotate — the court-visible frames.

    Returns ``[i for i, m in enumerate(mask) if m]`` (temporal order). These
    are *mask-relative* indices: the annotation loop iterates the processed
    window and only does pose/ball/overlay/write on the frames in this list,
    skipping every other one → no CPU spent on out-of-play frames.

    Note: indices here are window-relative (0-based within the processed
    window). Add ``start_frame`` to get the absolute original index, or use
    ``build_clip_index`` whose ``kept_original_frames`` does that for you.
    """
    return [i for i, m in enumerate(mask) if m]


def build_clip_index(mask, fps, start_frame=0):
    """Build the clip-index sidecar dict for the annotated cut.

    Parameters
    ----------
    mask : list[bool]
        Cleaned court-visibility mask over the processed window
        (``vision.shot_guard.compute_court_visibility_mask``). ``True`` =
        court visible = keep & annotate; ``False`` = cut.
    fps : float
        Frames per second of the source video (``reader.fps``), recorded so the
        front can convert annotated/original frame indices to seconds.
    start_frame : int
        Absolute index of the window's first frame (``int(args.start * fps)``).
        Added to every kept index so ``kept_original_frames`` are ABSOLUTE.

    Returns
    -------
    dict
        ``{
            "fps": float,
            "original_total": int,          # len(mask) — frames in the window
            "annotated_total": int,         # sum(mask) — frames actually kept
            "kept_original_frames": [int],  # ABSOLUTE original idx, annotated order
            "cut_spans": [[s, e], ...],     # window-RELATIVE inclusive non-court runs
            "start_frame": int,             # window offset (abs idx of mask[0])
        }``

    ``kept_original_frames[k]`` is the absolute original frame index of the
    k-th annotated frame — the re-alignment mapping the front consumes.
    """
    fps = float(fps)
    start_frame = int(start_frame)
    original_total = len(mask)
    kept_original_frames = [start_frame + i for i, m in enumerate(mask) if m]
    return {
        "fps": fps,
        "original_total": int(original_total),
        "annotated_total": int(len(kept_original_frames)),
        "kept_original_frames": kept_original_frames,
        "cut_spans": non_court_spans(mask),
        "start_frame": start_frame,
    }


def annotated_to_original(clip_index, annotated_frame):
    """Original (absolute) frame index of the ``annotated_frame``-th kept frame.

    The inverse of the cut: given a position in the SHORTER annotated video,
    return where it came from in the source. ``annotated_frame`` is 0-based
    into ``kept_original_frames``.

    Raises ``IndexError`` if ``annotated_frame`` is out of range — every
    annotated frame is by construction a kept one, so a miss is a real bug, not
    a "cut" frame (use ``original_to_annotated`` for the cut-detection
    direction, which returns ``None`` for cut frames).
    """
    kept = clip_index["kept_original_frames"]
    if annotated_frame < 0 or annotated_frame >= len(kept):
        raise IndexError(
            f"annotated_frame {annotated_frame} out of range "
            f"[0, {len(kept)})"
        )
    return int(kept[annotated_frame])


def original_to_annotated(clip_index, original_frame):
    """Annotated frame index for an ABSOLUTE original frame, or ``None``.

    The forward direction: where does source frame ``original_frame`` land in
    the annotated video? Returns the 0-based annotated index if the frame was
    kept, or ``None`` if it fell in a cut (non-court) span — so this doubles as
    the "was this frame cut?" test.

    ``original_frame`` is an ABSOLUTE index (same space as
    ``kept_original_frames`` / ``start_frame + i``). O(log n) via binary search
    on the sorted kept list.
    """
    kept = clip_index["kept_original_frames"]
    # kept_original_frames is strictly increasing → binary search.
    lo, hi = 0, len(kept)
    while lo < hi:
        mid = (lo + hi) // 2
        if kept[mid] < original_frame:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(kept) and kept[lo] == original_frame:
        return lo
    return None


# ─────────────────────── optional integration helper ──────────────────────
# S1 will call shot_guard + this module from run_pipeline_8s.py; this helper
# packages the "write the sidecar next to _stats.json" step so S1 doesn't have
# to re-derive the path/contract. It is OPTIONAL — S1 may inline it instead.

def write_clip_index(output_path, mask, fps, start_frame=0):
    """Build the clip-index and write it next to the annotated video / stats.

    Mirrors how ``run_pipeline_8s.py`` derives ``_stats.json``:
    ``<output without ext>_clipindex.json``. Returns ``(path, clip_index)``.
    Never raises on I/O — like the stats emission, a failed sidecar write must
    not break the (already produced) video output; on failure it returns
    ``(None, clip_index)`` so the caller can still log/use the index in memory.

    S1 usage (Pass 0.5 already computed ``mask``)::

        from vision.annot_clip import frames_to_annotate, write_clip_index
        keep = frames_to_annotate(mask)          # window-relative indices to annotate
        ...                                       # annotate ONLY those frames
        write_clip_index(output_path, mask, fps, start_frame)
    """
    import json
    from pathlib import Path

    clip_index = build_clip_index(mask, fps, start_frame=start_frame)
    path = str(Path(output_path).with_suffix("")) + "_clipindex.json"
    try:
        with open(path, "w") as f:
            json.dump(clip_index, f, indent=2)
    except Exception:
        return None, clip_index
    return path, clip_index
