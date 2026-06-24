"""Tests for vision.rally_outcome.classify_rally_outcome.

Pure synthetic data — covers each outcome (winner, unforced_error, forced_error)
plus the neutral default. No video, no GPU.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.rally_outcome import classify_rally_outcome, WINNER_SPEED_RATIO, WEAK_QUALITY

FPS = 60.0


def _shot(frame, side, speed=80.0, quality=60):
    return {"frame": frame, "x": 600, "y": 600,
            "player_side": side, "stroke": "forehand",
            "quality": quality, "speed_kmh": speed}


def _bounce(frame, depth="mid", x=700, y=650):
    return {"frame": frame, "x": x, "y": y, "depth": depth,
            "speed_kmh": 70.0, "quality": 50}


def _rally(start, end, shot_frames, bounce_frames):
    return {"start_frame": start, "end_frame": end,
            "shot_frames": list(shot_frames), "bounce_frames": list(bounce_frames),
            "n_shots": len(shot_frames)}


def test_winner_big_near_shot_then_deep_no_return():
    # Near hits a big shot at f=300; bounce lands deep at f=315; rally ends at
    # f=320 (well within the no-return gap). No opponent shot after f=300.
    shots = [
        _shot(100, "far", 70.0),
        _shot(200, "near", 70.0),
        _shot(300, "near", speed=110.0),  # big attacking shot
    ]
    bounces = [_bounce(160, "mid"), _bounce(260, "mid"), _bounce(315, "deep")]
    rally = _rally(90, 320, [100, 200, 300], [160, 260, 315])
    res = classify_rally_outcome(rally, shots, bounces, FPS)
    assert res["outcome"] == "winner", res
    assert res["defining_frame"] == 300, res


def test_unforced_error_weak_near_shot_short_placement():
    # Near's last shot is weak and the resulting bounce is very short ("short"
    # depth). Rally ends on near's racquet, no opponent return.
    shots = [
        _shot(100, "far", 70.0),
        _shot(200, "near", 70.0, quality=70),
        _shot(300, "near", 60.0, quality=20),  # weak
    ]
    bounces = [_bounce(160, "mid"), _bounce(260, "mid"), _bounce(315, "short")]
    rally = _rally(90, 320, [100, 200, 300], [160, 260, 315])
    res = classify_rally_outcome(rally, shots, bounces, FPS)
    assert res["outcome"] == "unforced_error", res
    assert res["defining_frame"] == 300, res


def test_forced_error_opponent_big_shot_no_return():
    # The rally ends on a far (opponent) big shot placed deep, near can't answer.
    shots = [
        _shot(100, "near", 70.0),
        _shot(200, "far", 70.0),
        _shot(300, "far", speed=120.0),  # opponent's big shot ends it
    ]
    bounces = [_bounce(160, "mid"), _bounce(260, "mid"), _bounce(315, "deep")]
    rally = _rally(90, 320, [100, 200, 300], [160, 260, 315])
    res = classify_rally_outcome(rally, shots, bounces, FPS)
    assert res["outcome"] == "forced_error", res
    assert res["defining_frame"] == 300, res


def test_neutral_when_no_decisive_signal():
    # Ordinary exchange, last shot by near at average speed, bounce is mid, rally
    # continues past the gap → nothing decisive.
    shots = [
        _shot(100, "far", 80.0),
        _shot(200, "near", 80.0, quality=70),
    ]
    bounces = [_bounce(160, "mid"), _bounce(260, "mid")]
    rally = _rally(90, 280, [100, 200], [160, 260])
    res = classify_rally_outcome(rally, shots, bounces, FPS)
    assert res["outcome"] == "neutral", res


def test_winner_requires_deep_placement():
    # Big near shot but the next bounce is SHORT (not deep) → not a winner; with
    # good quality it stays neutral (not an error either).
    shots = [
        _shot(100, "far", 70.0),
        _shot(300, "near", speed=110.0, quality=70),  # big, but well-struck
    ]
    bounces = [_bounce(160, "mid"), _bounce(315, "short")]
    rally = _rally(90, 320, [100, 300], [160, 315])
    res = classify_rally_outcome(rally, shots, bounces, FPS)
    assert res["outcome"] == "neutral", res


def test_unforced_error_needs_low_quality():
    # Last near shot lands short, but it was a HIGH-quality strike (not a fault) →
    # not an unforced error. Falls through to neutral.
    shots = [
        _shot(100, "far", 70.0),
        _shot(300, "near", speed=110.0, quality=80),  # high quality
    ]
    bounces = [_bounce(160, "mid"), _bounce(315, "short")]
    rally = _rally(90, 320, [100, 300], [160, 315])
    res = classify_rally_outcome(rally, shots, bounces, FPS)
    assert res["outcome"] == "neutral", res


if __name__ == "__main__":
    test_winner_big_near_shot_then_deep_no_return()
    test_unforced_error_weak_near_shot_short_placement()
    test_forced_error_opponent_big_shot_no_return()
    test_neutral_when_no_decisive_signal()
    test_winner_requires_deep_placement()
    test_unforced_error_needs_low_quality()
    print("All rally-outcome tests passed.")
