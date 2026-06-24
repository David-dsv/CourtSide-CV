"""Tests for vision.fatigue.analyze_fatigue.

Pure synthetic data — no video, no GPU. Covers:
  * a clearly fatiguing sequence (steep speed + quality decline) → fatigued, high conf
  * a constant sequence → not fatigued
  * the <6 near shots guard → confidence "none"
  * only-near filtering (far shots ignored)
  * drop_pct direction (positive = slowing down)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.fatigue import analyze_fatigue, MIN_SHOTS_FOR_ANALYSIS


FPS = 60.0
START = 0
END = 60 * FPS  # 60s clip


def _shot(frame, side, speed, quality=60, stroke="forehand"):
    return {"frame": frame, "x": 100, "y": 100, "player_side": side,
            "stroke": stroke, "quality": quality, "speed_kmh": speed}


def test_fatiguing_sequence_is_flagged_high_confidence():
    # Speed drops from ~110 to ~60 km/h over the clip; quality 80 -> 40.
    n = 10
    frames = np.linspace(0, END - 1, n).astype(int)
    speeds = np.linspace(110, 60, n)
    quals = np.linspace(80, 40, n)
    shots = [_shot(int(f), "near", round(float(s), 1), int(q))
             for f, s, q in zip(frames, speeds, quals)]

    res = analyze_fatigue(shots, [START, END], FPS)

    assert res["fatigued"] is True, res
    assert res["confidence"] == "high", res
    # Last third should be materially slower than the first third.
    assert res["drop_pct"] >= 0.10, res
    assert res["last_third_avg_speed"] < res["first_third_avg_speed"], res
    # Slope must be negative (units km/h per minute).
    assert res["slope_kmh_per_min"] < 0, res
    assert res["slope_quality_per_min"] < 0, res
    # sample_frame points at the slowest shot (last one here).
    assert res["sample_frame"] == int(frames[-1]), res
    assert res["n_near_shots"] == n


def test_constant_sequence_is_not_fatigued():
    n = 10
    frames = np.linspace(0, END - 1, n).astype(int)
    shots = [_shot(int(f), "near", 90.0, quality=65) for f in frames]

    res = analyze_fatigue(shots, [START, END], FPS)

    assert res["fatigued"] is False, res
    assert res["confidence"] == "none", res
    assert abs(res["drop_pct"]) < 0.10, res
    assert abs(res["slope_kmh_per_min"]) < 1.0, res


def test_too_few_shots_yields_none_confidence():
    shots = [_shot(0, "near", 110.0), _shot(10, "near", 60.0),
             _shot(20, "near", 55.0)]
    res = analyze_fatigue(shots, [START, END], FPS)
    assert res["confidence"] == "none", res
    assert res["fatigued"] is False, res
    assert res["n_near_shots"] == 3
    assert res["n_near_shots"] < MIN_SHOTS_FOR_ANALYSIS


def test_far_shots_are_ignored():
    n = 8
    frames = np.linspace(0, END - 1, n).astype(int)
    # All shots are 'far' — near set is empty → confidence none.
    shots = [_shot(int(f), "far", 90.0) for f in frames]
    res = analyze_fatigue(shots, [START, END], FPS)
    assert res["confidence"] == "none", res
    assert res["n_near_shots"] == 0


def test_drop_pct_positive_means_slowing_down():
    n = 9
    frames = np.linspace(0, END - 1, n).astype(int)
    speeds = np.linspace(100, 70, n)  # monotonically slowing
    shots = [_shot(int(f), "near", round(float(s), 1)) for f, s in zip(frames, speeds)]
    res = analyze_fatigue(shots, [START, END], FPS)
    assert res["drop_pct"] > 0, res
    assert res["fatigued"] is True, res


if __name__ == "__main__":
    test_fatiguing_sequence_is_flagged_high_confidence()
    test_constant_sequence_is_not_fatigued()
    test_too_few_shots_yields_none_confidence()
    test_far_shots_are_ignored()
    test_drop_pct_positive_means_slowing_down()
    print("All fatigue tests passed.")
