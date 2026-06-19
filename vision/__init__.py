"""
CourtSide-CV vision engine — pure, testable algorithms decoupled from CLI and
video I/O (the 'moteur propre' milestone in PROJET.md).

Currently exposes the ball-trajectory bounce detector. run_pipeline_8s.py imports
from here so the production path and the tests exercise the exact same code.
"""
from .bounce import detect_bounces_from_trajectory, smooth_ball_trajectory

__all__ = ["detect_bounces_from_trajectory", "smooth_ball_trajectory"]
