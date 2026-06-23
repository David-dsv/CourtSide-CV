"""Per-frame court homography tracker (Pass 0) for moving cameras.

A broadcast camera pans/tilts between points; a homography solved once on frame
0 then drifts (measured 5-140px on a 13s clip). This recomputes the image↔world
homography at every frame via the keypoint detector and exposes a per-frame
lookup that Pass 2 consumes, so the court overlay/minimap/speed follow the
camera.

Design choices (measured, not assumed):
- **Recompute every frame, don't interpolate keypoints.** Under a panning
  camera the corners accelerate, so linear keypoint interpolation between two
  sampled frames would introduce drift exactly where the camera moves fastest.
  Recomputing each frame is geometrically exact; its only artifact (keypoint
  jitter) is handled by the robust solve in CourtKeypointDetector.
- **Hold on fallback.** A single 'fallback' frame (keypoint detector lost the
  court for ~0.3s) is a glitch — reuse the last good H rather than dropping to
  None and breaking every downstream consumer. Warn after `max_hold` consecutive
  holds (camera truly lost).
- **No EMA on H.** Homographies don't form a vector space. If jitter remains
  after the robust solve, EMA the *keypoints* then re-solve (Phase 3, not here).
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from loguru import logger


@dataclass
class FrameH:
    """Per-frame homography result (looked up by frame_idx in Pass 2)."""
    H: Optional[np.ndarray]          # image px → world meters (3×3), or None
    H_inv: Optional[np.ndarray]      # world meters → image px (3×3), or None
    confidence: str                  # "high" | "low" | "fallback" | "held"
    rms_px: Optional[float]
    n_used: int
    frame_idx: int


def compute_homography_series(frames, court_det, frame_diag,
                              *, max_hold: int = 15) -> List[FrameH]:
    """Compute one homography per frame, holding the last good H on fallbacks.

    Args:
        frames: iterable of BGR frames (native res), one per frame.
        court_det: CourtKeypointDetector (its compute_homography is robust to
            hallucinated keypoints).
        frame_diag: frame diagonal in px (passed through to compute_homography).
        max_hold: consecutive fallback frames before a warning (camera lost).

    Returns: list[FrameH] indexed by frame_idx. H is None only if no frame up to
        that point ever produced a valid H (rare; happens if f0 itself fails).
    """
    out: List[FrameH] = []
    last_good: Optional[dict] = None
    consec_fallback = 0

    for i, frame in enumerate(frames):
        res = court_det.compute_homography(frame, frame_diag)
        if res["confidence"] == "fallback" or res.get("H") is None:
            consec_fallback += 1
            if last_good is not None:
                out.append(FrameH(
                    H=last_good["H"], H_inv=last_good["H_inv"],
                    confidence="held", rms_px=last_good["rms_px"],
                    n_used=last_good["n_used"], frame_idx=i))
                if consec_fallback == max_hold:
                    logger.warning(
                        f"court_track: {max_hold} frames held consecutively at "
                        f"frame {i} — camera may have lost the court")
            else:
                out.append(FrameH(
                    H=None, H_inv=None, confidence="fallback",
                    rms_px=None, n_used=res.get("n_used", 0), frame_idx=i))
        else:
            consec_fallback = 0
            last_good = res
            out.append(FrameH(
                H=res["H"], H_inv=res["H_inv"], confidence=res["confidence"],
                rms_px=res["rms_px"], n_used=res["n_used"], frame_idx=i))

    n_good = sum(1 for f in out if f.H is not None)
    logger.info(f"court_track: {n_good}/{len(out)} frames with a usable H "
                f"({n_good * 100 / max(1, len(out)):.0f}%)")
    return out


def h_at(by_frame: Optional[List[FrameH]], idx: int, default: Optional[dict]) -> Optional[dict]:
    """Look up the homography dict for frame `idx`, falling back to `default`
    (the f0 / calibration H) when the per-frame series is absent or that frame
    has no usable H. Returns a dict shaped like CourtKeypointDetector's output
    so the downstream consumers (shot_quality, draw_minimap, derive_court_zones)
    are unchanged."""
    if by_frame is None or idx < 0 or idx >= len(by_frame):
        return default
    fh = by_frame[idx]
    if fh.H is None:
        return default
    return {"H": fh.H, "H_inv": fh.H_inv, "confidence": fh.confidence,
            "rms_px": fh.rms_px, "n_used": fh.n_used}
