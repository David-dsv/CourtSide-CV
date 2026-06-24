"""Fatigue analysis for the near player.

Derives a fatigue signal from the trajectory of the near player's shots over
the clip: a significant downward drift in shot speed and/or shot quality.

The module is intentionally dependency-light (numpy only) and every threshold
is expressed as a *ratio* of an in-clip quantity (mean speed, clip duration) so
that it generalizes across resolutions, fps and players — no magic numbers tied
to a specific video (see CLAUDE.md "Zero hardcoding").

Public API
----------
    analyze_fatigue(shots, frame_range, fps) -> dict

``shots`` is the list of shot dicts produced by the pipeline::

    {"frame": int, "x": int, "y": int,
     "player_side": "near"|"far", "stroke": "forehand"|"backhand",
     "quality": 0..100, "speed_kmh": float}
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# ─── Ratio thresholds (no absolute pixel / km/h values) ──────────────────────
# A shot-speed slope is "significantly negative" if it would erase this fraction
# of the clip's mean speed over one minute of play.
SPEED_SLOPE_NEG_RATIO = 0.10            # −10 % of mean speed per minute
# Quality slope: same idea, in quality points per minute, as a fraction of the
# 0..100 quality scale.
QUALITY_SLOPE_NEG_RATIO = 0.10          # −10 pts/min (= 10 % of the scale)
# Drop between the first and last thirds of the clip that counts as "fatigue".
DROP_PCT_FLAG = 0.10                    # ≥ 10 % slower in the last third
# Confidence is "high" when BOTH signals agree (slope + drop) and the drop is
# at least this large.
DROP_PCT_HIGH_CONFIDENCE = 0.18         # ≥ 18 % drop ⇒ high confidence
# Fewer than this many near shots → not enough signal to judge fatigue.
MIN_SHOTS_FOR_ANALYSIS = 6


def _linear_slope_per_min(frames: np.ndarray, values: np.ndarray, fps: float) -> float:
    """Ordinary-least-squares slope of ``values`` vs time, in *units per minute*.

    ``frames`` are absolute frame indices; time in minutes is ``frame / fps / 60``.
    Returns 0.0 when fewer than 2 distinct time points exist.
    """
    times_min = frames.astype(float) / float(fps) / 60.0
    if times_min.size < 2 or np.ptp(times_min) == 0:
        return 0.0
    # slope has units (value / minute).
    slope, _ = np.polyfit(times_min, values, 1)
    return float(slope)


def _thirds_means(frames: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Mean of ``values`` over the first third and the last third of the time span.

    Thirds are defined by the temporal extent of the clip (min..max frame), not
    by shot count, so a slow-then-fast ordering is captured even if the shots are
    unevenly distributed. Returns (0.0, 0.0) if either third is empty.
    """
    fmin, fmax = float(frames.min()), float(frames.max())
    span = fmax - fmin
    if span <= 0:
        return float(values.mean()), float(values.mean())
    t1 = fmin + span / 3.0
    t2 = fmin + 2.0 * span / 3.0
    first = values[frames < t1]
    last = values[frames >= t2]
    first_avg = float(first.mean()) if first.size else float("nan")
    last_avg = float(last.mean()) if last.size else float("nan")
    return first_avg, last_avg


def analyze_fatigue(shots: Sequence[dict], frame_range: Sequence[int], fps: float) -> dict:
    """Estimate whether the *near* player fatigues over the clip.

    Parameters
    ----------
    shots : list[dict]
        Pipeline shot dicts (see module docstring). Only ``player_side == "near"``
        shots are used — fatigue is assessed for the filmed (near) player.
    frame_range : (start_frame, end_frame)
        Absolute frame window of the clip, from the stats JSON.
    fps : float
        Clip frame rate (e.g. 60.0 for felix, 50.0 for tennis).

    Returns
    -------
    dict with keys:
        slope_kmh_per_min      : float   — OLS slope of speed vs time (km/h per min)
        slope_quality_per_min  : float   — OLS slope of quality vs time (pts per min)
        first_third_avg_speed  : float   — mean near-shot speed in the first third
        last_third_avg_speed   : float   — mean near-shot speed in the last third
        drop_pct               : float   — relative speed drop, last vs first third
                                           (positive = getting slower; 0.0 if unknown)
        fatigued               : bool    — True if a significant drop is detected
        confidence             : "high" | "low" | "none"
        sample_frame           : int | None — frame of the slowest near shot
                                              (useful to point the UI at the low point)
        n_near_shots           : int     — number of near shots analysed
    """
    near = [s for s in shots if s.get("player_side") == "near"]
    n_near = len(near)

    # Default "no signal" payload, returned verbatim when there is too little data
    # or when inputs are degenerate (e.g. fps <= 0).
    empty = {
        "slope_kmh_per_min": 0.0,
        "slope_quality_per_min": 0.0,
        "first_third_avg_speed": 0.0,
        "last_third_avg_speed": 0.0,
        "drop_pct": 0.0,
        "fatigued": False,
        "confidence": "none",
        "sample_frame": None,
        "n_near_shots": n_near,
    }

    if n_near < MIN_SHOTS_FOR_ANALYSIS or not fps or fps <= 0:
        return empty

    frames = np.array([float(s["frame"]) for s in near], dtype=float)
    speeds = np.array([float(s["speed_kmh"]) for s in near], dtype=float)
    qualities = np.array([float(s["quality"]) for s in near], dtype=float)

    mean_speed = float(speeds.mean()) if speeds.size else 0.0
    if mean_speed <= 0:
        # No usable speed scale → cannot express thresholds as ratios.
        return empty

    slope_speed = _linear_slope_per_min(frames, speeds, fps)
    slope_quality = _linear_slope_per_min(frames, qualities, fps)

    first_avg, last_avg = _thirds_means(frames, speeds)
    if first_avg and first_avg > 0 and not np.isnan(first_avg) and not np.isnan(last_avg):
        drop_pct = float((first_avg - last_avg) / first_avg)
    else:
        drop_pct = 0.0

    # Thresholds as ratios of the in-clip speed/quality scale (zero hardcoding).
    speed_slope_flag = slope_speed <= -SPEED_SLOPE_NEG_RATIO * mean_speed
    quality_slope_flag = slope_quality <= -QUALITY_SLOPE_NEG_RATIO * 100.0
    drop_flag = drop_pct >= DROP_PCT_FLAG

    # The fatigue FLAG is driven by the SPEED signal (spec): a significantly
    # negative speed slope OR a clear first-third → last-third speed drop.
    # Quality decline is reported (slope_quality_per_min) and used to *raise*
    # confidence when it agrees with the speed signal, but a quality-only decline
    # with flat speed does NOT flag fatigue on its own (avoids false positives on
    # clips where the player simply hits more defensively).
    fatigued = bool(speed_slope_flag or drop_flag)

    # Confidence:
    #   high  — speed signal fires AND quality agrees (or the speed drop is large)
    #   low   — speed signal fires but quality doesn't corroborate
    #   none  — not fatigued
    if fatigued and (quality_slope_flag or drop_pct >= DROP_PCT_HIGH_CONFIDENCE):
        confidence = "high"
    elif fatigued:
        confidence = "low"
    else:
        confidence = "none"

    sample_frame = int(frames[int(np.argmin(speeds))]) if speeds.size else None

    return {
        "slope_kmh_per_min": round(slope_speed, 3),
        "slope_quality_per_min": round(slope_quality, 3),
        "first_third_avg_speed": round(first_avg, 1) if not np.isnan(first_avg) else 0.0,
        "last_third_avg_speed": round(last_avg, 1) if not np.isnan(last_avg) else 0.0,
        "drop_pct": round(drop_pct, 3),
        "fatigued": fatigued,
        "confidence": confidence,
        "sample_frame": sample_frame,
        "n_near_shots": n_near,
    }
