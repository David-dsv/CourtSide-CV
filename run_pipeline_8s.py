"""
Tennis Match Annotation Pipeline
Pose skeletons on all detected persons + Kalman-filtered ball trail.
2-player tracking with bounding boxes, ball bounce detection, court zone overlay.

Works on any tennis match video regardless of resolution, camera angle, or players.
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from collections import deque
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO

from models.yolo_detector import TennisYOLODetector
from utils.video_utils import VideoReader, VideoWriter
from utils.drawing import draw_player_bbox, draw_bounce_marker, draw_court_overlay, create_legend

# Derived analytics (Session B): fatigue trend + per-rally outcome.
# These are pure functions over the stats dicts; they add no rendering.
from vision.fatigue import analyze_fatigue
from vision.rally_outcome import classify_rally_outcome

# COCO skeleton connections (fixed, not video-dependent)
BODY_BONES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
HEAD_BONES = [(0,1),(0,2),(1,3),(2,4)]

# Default colors (BGR)
BALL_COLOR = (0, 255, 100)
TRAIL_COLOR = (50, 255, 50)

# Player colors (BGR) — amber for P1, blue for P2
PLAYER_COLORS = {
    1: (0, 191, 255),   # amber/gold
    2: (255, 160, 50),   # blue
}


class BallKalmanFilter:
    """
    Constant-velocity Kalman filter for ball tracking.
    State: [x, y, vx, vy]  — position and velocity in pixels.

    - predict(): advance state by one timestep, return predicted (x, y)
    - update(x, y): correct state with a measurement
    - gating_distance(x, y): Mahalanobis-like distance to check if
      a detection is consistent with the predicted position
    """

    def __init__(self, process_noise=100.0, measurement_noise=4.0):
        # State: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
        self.initialized = False
        self.frames_since_update = 0

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.frames_since_update = 0

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        self.frames_since_update += 1
        return float(pred[0][0]), float(pred[1][0])

    def update(self, x, y):
        if not self.initialized:
            self.init_state(x, y)
            return
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.frames_since_update = 0

    def gating_distance(self, x, y):
        """Euclidean distance between predicted position and measurement."""
        if not self.initialized:
            return 0.0
        pred = self.kf.statePost
        return float(np.hypot(x - pred[0][0], y - pred[1][0]))


def reject_teleport_frozen_blobs(centers, frame_diag, fps,
                                 jump_frac=0.15, hard_jump_frac=0.30,
                                 freeze_frac=0.012, min_run_frac=0.16,
                                 premotion_k=5, premotion_floor_frac=0.0008,
                                 chain_gap_frac=0.10, chain_min_run=3,
                                 cold_min_run_frac=0.30):
    """Delete "teleport-then-frozen" static-FP blobs from a raw ball track.

    The ball track can lock onto a static false positive (a corner blob / logo /
    court mark): the trajectory ENTERS the cluster by a large single-frame jump
    from the last valid ball position, then STAYS FROZEN there for many frames.
    Such a frozen blob, read by vision/events.py `_direction_features`, fabricates
    a `vy_flip` → a MANDATORY (un-skippable) BOUNCE anchor that floods the
    alternation DP, so the event F1 swings run-to-run (the S6 "phantom bounces
    that come and go"). Deleting the blob AT THE SOURCE fixes every downstream
    consumer at once (events, bounce, speed, minimap) and is DETERMINISTIC.

    THE TRAP (do not destroy real tracking): a REAL ball at an arc apex / serve
    toss is ALSO near-frozen (it decelerates at the top of the arc — the ball GT
    confirms ±5px frozen runs at apexes). A "frozen" reject ALONE would delete the
    real ball. So a run is rejected only on a CONJUNCTION, never on frozenness:

      FROZEN: >= min_run consecutive non-None centers within freeze_radius
              (= freeze_frac*diag) of the running centroid, AND
      one of:
        (Tier 1) HARD TELEPORT: entry jump > hard_jump_frac*diag — a
                 physically-impossible single-frame jump (no real ball covers
                 30% of the frame diagonal in one frame); OR
        (Tier 2) SOFT TELEPORT (entry jump > jump_frac*diag) AND the track was
                 HEALTHILY MOVING just before the jump (mean inter-frame step of
                 the premotion_k real centers ending at the entry >= a small
                 motion floor). A real ball re-acquired AFTER A COAST enters from
                 a near-frozen (decayed-prediction) state and is KEPT; only a
                 healthy lock that suddenly teleports into a freeze is an FP.

    A redirected real ball (occluded at contact, reappears travelling the other
    way) is also kept: it KEEPS MOVING after re-acquisition, so it never forms a
    >= min_run frozen run (the frozen requirement is the ultimate safety net).

    EPISODE CHAINING: one FP capture is often a *compound* excursion — the track
    teleports into a corner blob, freezes, then briefly latches a SECOND, shorter
    frozen FP on the way back before re-acquiring the real ball (measured: the
    f399-414 corner blob is followed by a 5-frame frozen FP @ (711,517) the base
    rule misses for run_len < min_run). Leaving that short sub-blob lets the spline
    reconnect through it and the phantom BOUNCE anchor simply MOVES to the episode
    edge instead of vanishing. So after a core blob is deleted we keep absorbing
    immediately-following SHORT frozen runs (>= chain_min_run, separated by <=
    chain_gap frames) that are still TELEPORTS from the pre-episode real anchor,
    until a run appears that is NOT a teleport from the anchor — a genuine
    ballistic re-acquisition, which ends the episode and is kept.

    COLD-START case (the dominant live FP): the track's VERY FIRST lock can be an
    FP — there is no prior real point, so the teleport-ENTRY test cannot fire
    (entry_jump is undefined). Measured live: the track latches a mid-court blob
    @ (538,473) for 35 frames at clip start while the real ball is at the top of
    frame (GT ~700px away), faking a cold-start BOUNCE and tanking cold-start
    CORRECT. The separator vs a real cold-start apex/toss (which IS frozen too) is
    the EXIT: a real ball LEAVES the frozen apex on a smooth arc (small exit jump),
    while an FP is ABANDONED by a teleport when the track finally finds the real
    ball elsewhere. So the FIRST real lock (no prior real point), frozen for a LONG
    run (>= cold_min_run, longer than the normal floor — a real toss apex is
    short), and EXITED by a teleport (> jump_frac*diag from the centroid to the
    next real center) is rejected. The exit test is causal-safe here because this
    is a buffered post-pass over the fully-built raw track. Restricted to the first
    lock so a mid-clip real ball that merely PRECEDES an FP (its own teleport-out
    is caused by the *next* segment) is never falsely deleted.

    All thresholds are fractions of the frame diagonal / fps — no clip pixels.
    Returns (cleaned_centers, deleted_blobs) where each blob is a dict for logging.
    """
    n = len(centers)
    if n == 0:
        return centers, []
    jump_t = jump_frac * frame_diag
    hard_jump_t = hard_jump_frac * frame_diag
    freeze_r = freeze_frac * frame_diag
    min_run = max(8, int(round(min_run_frac * fps)))
    cold_min_run = max(min_run, int(round(cold_min_run_frac * fps)))
    premotion_floor = premotion_floor_frac * frame_diag

    chain_gap = max(2, int(round(chain_gap_frac * fps)))

    def grow_frozen_run(s):
        """Length + centroid of the maximal frozen run starting at index s.

        Each point must stay within freeze_r of the run ANCHOR (the first point),
        NOT a running mean. A running mean LAGS a steady slow drift, so a real ball
        creeping ~5-7px/frame would keep the mean-distance under freeze_r and read
        as "frozen" indefinitely (adversarial FINDING 2). Anchoring on the first
        point bounds the whole run to a fixed freeze_r ball: a real drifting ball
        leaves it after ~freeze_r/step frames and the run ends, so only a genuinely
        pinned cluster reaches min_run.
        """
        ax, ay = float(centers[s][0]), float(centers[s][1])
        sx, sy, cnt = ax, ay, 1
        j = s + 1
        while j < n and centers[j] is not None:
            if np.hypot(centers[j][0] - ax, centers[j][1] - ay) <= freeze_r:
                sx += centers[j][0]
                sy += centers[j][1]
                cnt += 1
                j += 1
            else:
                break
        return cnt, (sx / cnt, sy / cnt)

    def adjacent_exit_jump(centroid, end_idx):
        """Jump from a frozen-run centroid to the IMMEDIATELY ADJACENT next center.

        Returns None if the run is the end of the track OR a gap (None) follows it.
        The cold-start rule needs the exit to be ADJACENT (no intervening gap): a
        static FP is ABANDONED by a clean single-frame teleport (the track jumps
        straight from the frozen FP to the real ball on the very next frame — the
        live (538,473) FP exits at f52, adjacent to f51). A real toss/apex that is
        OCCLUDED at contact has a None gap before the struck ball re-appears far
        away — the jump across that gap is ambiguous (the ball moved while
        occluded), so we ABSTAIN (return None ⇒ no cold reject) and keep the real
        ball. This closes the serve-toss false-delete found in adversarial review.
        """
        j = end_idx + 1
        if j >= n or centers[j] is None:
            return None
        return np.hypot(centers[j][0] - centroid[0], centers[j][1] - centroid[1])

    out = list(centers)
    deleted = []
    i = 0
    last_real = None
    last_real_idx = None
    seen_real = False  # has the track locked a real (kept) point yet? (cold-start)
    cold_done = False  # the one-shot cold-start slot has been used
    while i < n:
        c = centers[i]
        if c is None:
            i += 1
            continue
        run_len, centroid = grow_frozen_run(i)
        entry_jump = (np.hypot(c[0] - last_real[0], c[1] - last_real[1])
                      if last_real is not None else None)
        # premotion = mean inter-frame step of the real centers ending at the
        # entry point (a coast re-acquisition has a near-zero decayed step).
        premotion = None
        if last_real_idx is not None:
            seq, k = [], last_real_idx
            while k >= 0 and centers[k] is not None and len(seq) < premotion_k + 1:
                seq.append(centers[k])
                k -= 1
            seq = seq[::-1]
            if len(seq) >= 2:
                steps = [np.hypot(seq[m + 1][0] - seq[m][0], seq[m + 1][1] - seq[m][1])
                         for m in range(len(seq) - 1)]
                premotion = float(np.mean(steps))

        is_frozen = run_len >= min_run
        is_teleport = entry_jump is not None and entry_jump > jump_t
        is_hard_teleport = entry_jump is not None and entry_jump > hard_jump_t
        is_healthy_before = premotion is not None and premotion >= premotion_floor
        reject = is_frozen and (is_hard_teleport or (is_teleport and is_healthy_before))

        # COLD-START: the FIRST real lock has no teleport-entry to test. Reject it
        # iff it is a LONG freeze (>= cold_min_run) ABANDONED by an ADJACENT
        # teleport (a real cold-start apex/toss leaves on a smooth arc → small exit
        # jump; an occluded real apex has a None gap before re-acquisition → abstain).
        # The cold rule may fire only ONCE — for the genuine first lock (last_real
        # is None) AND before any earlier cold reject (cold_done). After a cold
        # reject last_real is still None (the FP is not "real"), so without
        # cold_done a SECOND frozen run — possibly a REAL occluded apex — could be
        # re-cold-rejected (adversarial FINDING 1). cold_done closes that.
        exit_jump = None
        is_cold_start = False
        if (not reject and last_real is None and not cold_done
                and run_len >= cold_min_run):
            exit_jump = adjacent_exit_jump(centroid, i + run_len - 1)
            if exit_jump is not None and exit_jump > jump_t:
                reject = True
                is_cold_start = True

        if reject:
            for f in range(i, i + run_len):
                out[f] = None
            deleted.append({
                "start": i, "end": i + run_len - 1, "len": run_len,
                "centroid": (round(centroid[0], 1), round(centroid[1], 1)),
                "entry_jump_frac": (None if entry_jump is None
                                    else round(float(entry_jump) / frame_diag, 3)),
                "exit_jump_frac": (None if exit_jump is None
                                   else round(float(exit_jump) / frame_diag, 3)),
                "premotion_px": None if premotion is None else round(premotion, 2),
                "tier": ("cold" if is_cold_start
                         else (1 if is_hard_teleport else 2)),
            })
            # the deleted blob is not "real" — keep last_real at the pre-blob ball
            if is_cold_start:
                cold_done = True  # one-shot: never cold-reject again this track
            anchor = last_real
            i = i + run_len
            # EPISODE CHAINING: absorb trailing SHORT frozen sub-blobs that are
            # still teleports from the pre-episode anchor (the same FP capture
            # bouncing through more than one frozen spot before re-acquiring).
            while i < n and anchor is not None:
                g = 0
                while i < n and centers[i] is None and g < chain_gap:
                    i += 1
                    g += 1
                if i >= n or centers[i] is None:
                    break
                c2_len, c2_centroid = grow_frozen_run(i)
                ej2 = np.hypot(centers[i][0] - anchor[0], centers[i][1] - anchor[1])
                if c2_len >= chain_min_run and ej2 > jump_t:
                    for f in range(i, i + c2_len):
                        out[f] = None
                    deleted.append({
                        "start": i, "end": i + c2_len - 1, "len": c2_len,
                        "centroid": (round(c2_centroid[0], 1), round(c2_centroid[1], 1)),
                        "entry_jump_frac": round(float(ej2) / frame_diag, 3),
                        "premotion_px": None, "tier": "chain",
                    })
                    i = i + c2_len
                else:
                    break  # ballistic re-acquisition — episode over
        else:
            last_real = c
            last_real_idx = i
            seen_real = True
            i += 1
    return out, deleted


# Player identity is now handled by vision.player_track.TwoPlayerTracker (Pass 1.5):
# persistent P1(far)/P2(near) lock + gap-fill + smoothing. The old 2-slot IoU
# PlayerTracker was replaced by it.

# Bounce detection + trajectory smoothing now live in the vision/ engine
# module (PROJET.md 'moteur propre'), imported here so the pipeline and the
# test suite exercise the identical algorithm.
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory  # noqa: E402
from vision.pose import estimate_players_pose  # noqa: E402
from vision.minimap import draw_minimap  # noqa: E402
import vision.fx as fx  # noqa: E402  — cinematic visual effects (glow, glass HUD, shockwave)
from vision.court_track import h_at  # noqa: E402  — per-frame homography lookup (track-camera)
from vision.shot_guard import (  # noqa: E402  — court-visibility gate (cut non-play frames)
    compute_court_visibility_mask_streaming, non_court_spans)
from vision.annot_clip import (  # noqa: E402  — CUT bookkeeping: keep-list + clip-index sidecar
    frames_to_annotate, write_clip_index)


def _court_frames_between(mask, a, b):
    """Count court-visible (True) frames strictly between window-relative indices a and b.

    Used by rally segmentation so a multi-second non-court span (a replay between
    two real shots) doesn't falsely split one rally: the gap is measured in COURT
    frames only, treating the cut span as if it weren't there. When shot-guard is
    off, callers pass the plain frame delta instead."""
    if b <= a + 1:
        return 0
    return sum(1 for k in range(a + 1, b) if 0 <= k < len(mask) and mask[k])


def _kept_time_intervals(mask, fps, start_frame):
    """Contiguous runs of court-visible frames → [(start_sec, end_sec), ...] in the
    ORIGINAL timeline. These are the audio segments to keep so the soundtrack stays
    in sync with a cut (shorter) annotated video. mask is window-relative; the
    absolute original time of frame i is (start_frame + i) / fps."""
    intervals = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            # frames [i, j-1] kept → seconds [(start_frame+i)/fps, (start_frame+j)/fps)
            intervals.append(((start_frame + i) / fps, (start_frame + j) / fps))
            i = j
        else:
            i += 1
    return intervals


def _remux_audio(video_path, annotated_path, mask, fps, start_frame):
    """Re-attach the source audio to the (silent) annotated video, CUT to the same
    kept spans as the video so it stays perfectly in sync. Best-effort: a missing
    ffmpeg / no-audio source / any failure leaves the silent video untouched (never
    breaks the already-produced output). Returns True iff audio was muxed in.

    The annotated video drops non-court frames, so a plain `-c:a copy` of the full
    original audio would drift. We build an audio filtergraph that keeps only the
    kept time intervals (atrim per run + concat) and mux that onto the video."""
    import shutil
    import subprocess
    import tempfile
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found → annotated video has NO audio.")
        return False
    # Does the source even have an audio stream?
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=30)
        if "audio" not in (probe.stdout or ""):
            logger.info("Source has no audio stream → annotated video stays silent.")
            return False
    except Exception as exc:
        logger.warning(f"ffprobe failed ({exc}) → skipping audio remux.")
        return False

    intervals = _kept_time_intervals(mask, fps, start_frame)
    if not intervals:
        return False
    # Build the audio filtergraph: trim each kept span, then concat them in order.
    # Audio comes from input 1 (the ORIGINAL video); input 0 is the silent
    # annotated video. atrim each kept span off [1:a], then concat in order.
    parts = []
    for k, (s, e) in enumerate(intervals):
        parts.append(f"[1:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{k}]")
    concat_in = "".join(f"[a{k}]" for k in range(len(intervals)))
    filtergraph = ";".join(parts) + f";{concat_in}concat=n={len(intervals)}:v=0:a=1[aout]"

    out_fd, tmp_out = tempfile.mkstemp(suffix=".mp4",
                                       dir=str(Path(annotated_path).parent))
    import os as _os
    _os.close(out_fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", annotated_path,    # 0:v — the silent annotated video (kept frames)
        "-i", video_path,        # 1:a — the original audio (full timeline)
        "-filter_complex", filtergraph,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", tmp_out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        _os.replace(tmp_out, annotated_path)  # atomic swap onto the final path
        logger.info(f"Audio re-attached (cut to {len(intervals)} kept spans) → "
                    f"{annotated_path}")
        return True
    except Exception as exc:
        stderr = getattr(exc, "stderr", "") or ""
        logger.warning(f"Audio remux failed ({exc}) {stderr[:300]} → video stays silent.")
        try:
            _os.path.exists(tmp_out) and _os.remove(tmp_out)
        except Exception:
            pass
        return False


def _merge_solo_rallies(rallies):
    """Re-attribute any 1-shot rally ("solo") into its nearest neighbour.

    Mirrors mergeSoloRallies in web/src/lib/analysis/highlights.ts exactly, so
    the pipeline JSON and the front-end-derived rallies stay in sync. A solo is
    absorbed into whichever neighbour (prev/next) is closer by gap; ties go to
    the previous (chronological). Returns a new list.
    """
    if len(rallies) <= 1:
        return rallies
    out = [
        {
            "start_frame": r["start_frame"],
            "end_frame": r["end_frame"],
            "shot_frames": list(r["shot_frames"]),
            "bounce_frames": list(r["bounce_frames"]),
            "n_shots": r["n_shots"],
        }
        for r in rallies
    ]
    i = 0
    while i < len(out):
        r = out[i]
        if r["n_shots"] != 1:
            i += 1
            continue
        prev = out[i - 1] if i - 1 >= 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        gap_prev = (r["start_frame"] - prev["end_frame"]) if prev else float("inf")
        gap_next = (nxt["start_frame"] - r["end_frame"]) if nxt else float("inf")
        if gap_prev <= gap_next and prev:
            prev["shot_frames"].extend(r["shot_frames"])
            prev["bounce_frames"].extend(r["bounce_frames"])
            prev["end_frame"] = max(prev["end_frame"], r["end_frame"])
            prev["n_shots"] += r["n_shots"]
            out.pop(i)
        elif nxt:
            nxt["shot_frames"] = list(r["shot_frames"]) + nxt["shot_frames"]
            nxt["bounce_frames"] = list(r["bounce_frames"]) + nxt["bounce_frames"]
            nxt["start_frame"] = min(nxt["start_frame"], r["start_frame"])
            nxt["n_shots"] += r["n_shots"]
            out.pop(i)
        else:
            i += 1
    return out


def segment_rallies_relative(shot_by_frame, bounce_by_frame, fps, court_mask=None):
    """Segment shots into rallies in RELATIVE (clip-local) frame coordinates.

    This mirrors the post-Pass-2 absolute-frame segmentation (gap > 2.5s between
    consecutive shots → new rally, then ``_merge_solo_rallies``) EXACTLY, but
    keeps frame numbers relative so the result indexes ``shot_by_frame`` /
    ``bounce_by_frame`` directly. It is run BEFORE the render loop so the live
    per-rally HUD can know, at every frame, which rally is in progress.

    Returns a list of rally dicts with relative ``start_frame``/``end_frame``/
    ``shot_frames``/``bounce_frames``/``n_shots`` (same keys ``_merge_solo_rallies``
    expects), 1-based by list position.
    """
    if not shot_by_frame:
        return []
    rally_gap_frames = 2.5 * fps
    sorted_sf = sorted(shot_by_frame.keys())
    grouped = []
    cur = [sorted_sf[0]]
    for sf in sorted_sf[1:]:
        # shot-guard: measure the gap in COURT frames only, so a replay/close-up
        # between two real shots doesn't falsely split one rally in two.
        gap = (_court_frames_between(court_mask, cur[-1], sf)
               if court_mask is not None else sf - cur[-1])
        if gap > rally_gap_frames:
            grouped.append(cur)
            cur = []
        cur.append(sf)
    grouped.append(cur)
    sorted_bf = sorted(bounce_by_frame.keys())
    rallies = []
    for g in grouped:
        start = g[0]
        end = max(g[-1], max([bf for bf in sorted_bf if start <= bf <= g[-1] + rally_gap_frames],
                             default=g[-1]))
        bframes = [bf for bf in sorted_bf if start <= bf <= end]
        rallies.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "shot_frames": [int(sf) for sf in g],
            "bounce_frames": [int(bf) for bf in bframes],
            "n_shots": len(g),
        })
    # Same solo-merge as the absolute path so the HUD's rally count and the JSON
    # output agree (no off-by-one between what the video shows and what's exported).
    return _merge_solo_rallies(rallies)


def build_rally_at_frame(rallies, max_frames):
    """Map every frame index → 0-based rally index (or None between/around rallies).

    A frame belongs to a rally if it lies within [start_frame, end_frame]. The
    inter-rally gap frames map to None (no active point → the HUD freezes the last
    rally's tag rather than inventing an "ÉCHANGE 0"). Built from RELATIVE rally
    frames (see ``segment_rallies_relative``).
    """
    rally_at = [None] * max_frames
    for ri, r in enumerate(rallies):
        lo = max(0, int(r["start_frame"]))
        hi = min(max_frames - 1, int(r["end_frame"]))
        for f in range(lo, hi + 1):
            rally_at[f] = ri
    return rally_at


def _quality_color(q):
    """Quality → BGR (green ≥70, amber 40-69, red <40). Shared by labels + HUD."""
    return ((90, 220, 90) if q >= 70 else
            (90, 220, 220) if q >= 40 else (90, 90, 230))


def draw_event_legend(out, fx):
    """Persistent top-right key teaching SHAPE = event (color = attribute).

    Drawn under FX too (always on), so the viewer learns the vocabulary:
      ● flat ellipse + diamond stake = REBOND (ball on the ground)
      ◗ upright diamond + downward caret = FRAPPE (player strike, in the air)
    Glyphs are drawn in neutral grey with the SAME cv2 calls as the live events,
    so the legend matches the overlay exactly. Color is explicitly called out as
    the depth/stroke channel. Top-right is the only free corner (bottom-left =
    HUD card, bottom-right = radar, top-left = scoreboard-safe).
    """
    H, W = out.shape[:2]
    U = max(6, int(H * 0.018))
    pad = int(H * 0.014)
    pw, ph = int(W * 0.16), int(H * 0.095)
    x = W - pw - pad
    y = pad
    fx.glass_panel(out, x, y, pw, ph, tint=(30, 24, 20), alpha=0.5,
                   border=(170, 175, 195), radius=12)
    gx = x + int(2.0 * U)
    txs = x + int(4.2 * U)
    neut = (220, 220, 220)
    # row 1 — REBOND: flat squashed ellipse + diamond stake (ground silhouette)
    r1 = y + int(ph * 0.30)
    cv2.ellipse(out, (gx, r1), (int(1.3 * U), int(1.3 * U * 0.40)), 0, 0, 360,
                neut, 2, cv2.LINE_AA)
    cv2.drawMarker(out, (gx, r1), neut, cv2.MARKER_DIAMOND, max(7, int(0.6 * U)),
                   2, cv2.LINE_AA)
    fx.draw_text(out, "REBOND (sol)", (txs, r1), size=15, color=(225, 225, 230),
                 font="bold", anchor="lt")
    # row 2 — FRAPPE: upright diamond + downward caret (air silhouette)
    r2 = y + int(ph * 0.60)
    cv2.drawMarker(out, (gx, r2), neut, cv2.MARKER_DIAMOND, int(1.8 * U), 2, cv2.LINE_AA)
    cv2.line(out, (gx - int(0.7 * U), r2 - int(1.3 * U)), (gx, r2 - int(0.6 * U)),
             neut, 2, cv2.LINE_AA)
    cv2.line(out, (gx + int(0.7 * U), r2 - int(1.3 * U)), (gx, r2 - int(0.6 * U)),
             neut, 2, cv2.LINE_AA)
    fx.draw_text(out, "FRAPPE (balle)", (txs, r2), size=15, color=(225, 225, 230),
                 font="bold", anchor="lt")
    # color key
    fx.draw_text(out, "couleur = profondeur / coup",
                 (x + int(1.4 * U), y + int(ph * 0.86)), size=12,
                 color=(150, 155, 170), font="regular", anchor="lt")
    return out


def estimate_px_per_meter(court_zones, frame_height):
    """
    Estimate the scale factor (pixels per meter) from court zones or frame geometry.
    A tennis court is 23.77m long.
    """
    # Try to use court zone detection
    for name, info in court_zones.items():
        if name == "court":
            box = info["box"]
            court_h_px = box[3] - box[1]
            if court_h_px > 0:
                px_per_m = court_h_px / 23.77
                logger.info(f"Scale from court zone: {px_per_m:.2f} px/m "
                            f"(court height={court_h_px:.0f}px)")
                return px_per_m

    # Fallback: in broadcast tennis, the court occupies ~58% of frame height
    court_h_px = frame_height * 0.58
    px_per_m = court_h_px / 23.77
    logger.info(f"Scale from fallback (58% frame height): {px_per_m:.2f} px/m")
    return px_per_m


def classify_bounce_depth(bounce_events, court_zones, frame_height=None):
    """
    Classify each bounce as deep, short, or mid based on court zones.
    - deep: in dead zones or top/bottom 20% of court
    - short: in service boxes
    - mid: elsewhere in court

    Falls back to geometric approximation when court_zones is empty.

    Args:
        bounce_events: list of (frame_idx, x, y)
        court_zones: dict from detect_court_zones()
        frame_height: frame height in pixels (for geometric fallback)

    Returns:
        list of (frame_idx, x, y, depth_label)
    """
    def _point_in_box(px, py, box):
        return box[0] <= px <= box[2] and box[1] <= py <= box[3]

    # ─── Geometric fallback when no court zones detected ───
    if not court_zones and frame_height is not None:
        # Broadcast tennis approximations:
        # Net ≈ 47% of frame height, far baseline ≈ 18%, near baseline ≈ 82%
        net_y = frame_height * 0.47
        far_baseline = frame_height * 0.18
        near_baseline = frame_height * 0.82
        classified = []
        for frame_idx, x, y in bounce_events:
            # Determine which half the bounce is in
            if y < net_y:
                # Far side: depth relative to far baseline
                half_span = net_y - far_baseline
                dist_from_baseline = y - far_baseline
            else:
                # Near side: depth relative to near baseline
                half_span = near_baseline - net_y
                dist_from_baseline = near_baseline - y

            ratio = max(0.0, min(1.0, dist_from_baseline / half_span)) if half_span > 0 else 0.5
            # ratio close to 0 = near baseline = deep, close to 1 = near net = short
            if ratio <= 0.35:
                depth = "deep"
            elif ratio >= 0.70:
                depth = "short"
            else:
                depth = "mid"
            classified.append((frame_idx, x, y, depth))
        return classified

    # ─── Court-zone-based classification ───
    dead_zones = []
    service_boxes = []
    court_box = None
    for name, info in court_zones.items():
        box = info["box"]
        if "dead-zone" in name:
            dead_zones.append(box)
        elif "service-box" in name:
            service_boxes.append(box)
        elif name == "court":
            court_box = box

    classified = []
    for frame_idx, x, y in bounce_events:
        # Check dead zones first
        in_dead = any(_point_in_box(x, y, dz) for dz in dead_zones)
        if in_dead:
            classified.append((frame_idx, x, y, "deep"))
            continue

        # Check service boxes
        in_service = any(_point_in_box(x, y, sb) for sb in service_boxes)
        if in_service:
            classified.append((frame_idx, x, y, "short"))
            continue

        # If inside court but not in dead/service, check top/bottom 20%
        if court_box is not None and _point_in_box(x, y, court_box):
            court_h = court_box[3] - court_box[1]
            top_thresh = court_box[1] + court_h * 0.2
            bot_thresh = court_box[3] - court_h * 0.2
            if y <= top_thresh or y >= bot_thresh:
                classified.append((frame_idx, x, y, "deep"))
                continue

        classified.append((frame_idx, x, y, "mid"))

    return classified


def draw_skeleton(frame, keypoints, kp_conf, color, conf_thresh=0.25):
    for (i, j) in BODY_BONES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if kp_conf[i] < conf_thresh or kp_conf[j] < conf_thresh:
            continue
        cv2.line(frame, (int(keypoints[i][0]), int(keypoints[i][1])),
                 (int(keypoints[j][0]), int(keypoints[j][1])), color, 3, cv2.LINE_AA)
    for (i, j) in HEAD_BONES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if kp_conf[i] < conf_thresh or kp_conf[j] < conf_thresh:
            continue
        cv2.line(frame, (int(keypoints[i][0]), int(keypoints[i][1])),
                 (int(keypoints[j][0]), int(keypoints[j][1])), color, 2, cv2.LINE_AA)
    for i, (kp, sc) in enumerate(zip(keypoints, kp_conf)):
        if sc < conf_thresh:
            continue
        pt = (int(kp[0]), int(kp[1]))
        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, color, 2, cv2.LINE_AA)
    return frame


def draw_ball_trail(frame, trail):
    n = len(trail)
    if n < 2:
        return frame
    for i in range(1, n):
        alpha = i / n
        thickness = max(2, int(5 * alpha))
        c = tuple(int(v * (0.3 + 0.7 * alpha)) for v in TRAIL_COLOR)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness, cv2.LINE_AA)
    return frame


def draw_ball_marker(frame, cx, cy):
    cv2.circle(frame, (cx, cy), 7, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Tennis Match Annotation Pipeline")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path (default: data/output/<input>_annotated.mp4)")
    parser.add_argument("-s", "--start", type=float, default=0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("-e", "--end", type=float, default=None,
                        help="End time in seconds (default: end of video)")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Duration in seconds from start (alternative to --end)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, mps (default: cpu)")
    parser.add_argument("--trail-length", type=int, default=12, help="Ball trail length in frames")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold (persons)")
    parser.add_argument("--ball-conf", type=float, default=None,
                        help="BALL-only confidence floor, decoupled from --conf (persons stay "
                             "0.2). DEFAULT None = density-adaptive (--ball-auto): clean "
                             "broadcast gets 0.05, parasite-heavy oblique gets 0.16. Pass an "
                             "explicit value to OVERRIDE auto (pins both paths to it). The "
                             "tennis ball is tiny/faint and sat below the person conf (the OLD "
                             "code ran it at --conf=0.2). Kalman gating, not confidence, keeps "
                             "the real ball. See docs/research/s1-ball-sota-CR.md.")
    parser.add_argument("--ball-imgsz", type=int, default=None,
                        help="BALL-only inference resolution, decoupled from the detector imgsz "
                             "(1280). DEFAULT None = density-adaptive (--ball-auto): clean "
                             "broadcast gets 1920 (tiny/far ball win: demo3 cov 86.2%%->95.5%%, "
                             "CORRECT 79.1%%->92.3%%, cold-start 74.5%%->90.8%%), parasite-heavy "
                             "oblique stays 1280 (1920 floods far-field parasites the Kalman "
                             "locks onto, collapsing bounce F1 — felix). Pass an explicit value "
                             "to OVERRIDE auto. See docs/research/s1-ball-sota-CR.md.")
    parser.add_argument("--no-ball-auto", dest="ball_auto", action="store_false",
                        help="Disable density-adaptive ball resolution/conf. Falls back to the "
                             "legacy fixed defaults (imgsz 1280, conf 0.16) unless --ball-imgsz / "
                             "--ball-conf are given. Auto is ON by default.")
    parser.set_defaults(ball_auto=True)
    parser.add_argument("--ball-density-thr", type=float, default=3.5,
                        help="Mean ball-candidate density (per probed frame, probe conf 0.05) "
                             "below which a clip is treated as CLEAN broadcast (-> 1920/0.05) and "
                             "above which as PARASITE-heavy (-> 1280/0.16). Measured separation: "
                             "demo3 2.0, tennis 2.5, felix 6.9 — thr robust across 3.0-5.0. "
                             "Derived from detected features, no per-video constant.")
    # ─── Dynamic ROI ball fallback (search-region tracking) ───
    parser.add_argument("--no-ball-roi", dest="ball_roi", action="store_false",
                        help="Disable the dynamic-ROI ball fallback. When full-frame "
                             "detection misses the ball on a frame where the Kalman has a "
                             "prediction, the pipeline crops a window around the prediction, "
                             "upscales it and re-detects (search-region tracking — the ball is "
                             "bigger/sharper there and off-court parasites are excluded). The "
                             "recovered candidate is gated through the SAME Kalman gate as a "
                             "full-frame one. FALLBACK ONLY (fires on gap frames) → marginal "
                             "cost. ON by default; gated to the CLEAN (1920/0.05) density path "
                             "unless --ball-roi-all (parasite clips like felix don't fire it, "
                             "to avoid injecting far-field FPs). See "
                             "docs/research/ball-roi-dynamic-CR.md.")
    parser.set_defaults(ball_roi=True)
    parser.add_argument("--ball-roi-all", action="store_true",
                        help="Fire the dynamic-ROI ball fallback on ALL clips, including the "
                             "parasite-heavy (1280/0.16) density path. Default keeps the ROI on "
                             "the clean path only (felix-safe).")
    parser.add_argument("--ball-roi-half-frac", type=float, default=0.1667,
                        help="Dynamic-ROI half-window as a FRACTION of frame width (crop side = "
                             "2*half). Default 0.1667 (~320px half on 1920W) — wide enough to "
                             "absorb prediction error when the ball reverses (bounce/hit); a "
                             "tighter crop misses the ball at the turn. A ratio of frame width, "
                             "not a per-video pixel constant. (Measured: the WIDE window is the "
                             "CLEAN-win choice — 0 wrong/0 poison — vs a narrow window's noisier "
                             "+coverage. See docs/research/ball-roi-dynamic-CR.md.)")
    parser.add_argument("--ball-roi-zoom", type=float, default=2.0,
                        help="Dynamic-ROI upscale factor before re-detection. Default 2.0 "
                             "(moderate: enough to enlarge the tiny ball, not so much it blurs "
                             "past recognition). Measured 1.5/2.0/2.5 on the ball GT.")
    parser.add_argument("--ball-roi-gate-frac", type=float, default=0.25,
                        help="ROI candidates are gated to this FRACTION of the (grown) full "
                             "Kalman gate — TIGHTER than full-frame. Default 0.25. WHY: an "
                             "off-target ROI hit inside the wide grown gate feeds kf.correct() "
                             "and POISONS the Kalman state, corrupting neighbour frames (measured "
                             "-5 CORRECT unguarded). A fallback recovery must be CLOSE to the "
                             "prediction. Stable across 0.15-0.6 on the GT.")
    parser.add_argument("--ball-roi-min-conf", type=float, default=0.10,
                        help="ROI candidate must clear this conf (the 2x crop upscaling inflates "
                             "faint parasites). Default 0.10. With the tight gate this gives a "
                             "CLEAN win: demo3 coverage 95.5%%->96.8%%, CORRECT 92.3%%->93.2%%, "
                             "0 wrong/0 poisoned recoveries.")
    parser.add_argument("--pose-conf", type=float, default=0.10, help="Pose estimation confidence")
    parser.add_argument("--pose-model", default="yolov8m-pose.pt",
                        help="Pose model (yolov8m-pose.pt or yolov8x-pose.pt for better small player detection)")
    parser.add_argument("--pose-backend", choices=["yolo", "rtmw"], default="yolo",
                        help="Pose backend. 'yolo' = YOLOv8-pose (default, fast, 17 kpts). "
                             "'rtmw' = RTMW whole-body via rtmlib (133 kpts incl. hands/feet, "
                             "better on the tiny far player; CPU-only, ~3x slower). "
                             "Needs: pip install rtmlib onnxruntime.")
    # New feature flags
    parser.add_argument("--no-court", action="store_true", help="Disable court zone overlay")
    parser.add_argument("--no-bounces", action="store_true", help="Disable bounce detection markers")
    parser.add_argument("--no-player-boxes", action="store_true",
                        help="(default) Disable player bounding boxes. Boxes are OFF by default — "
                             "skeletons read cleaner; pass --player-boxes to re-enable.")
    parser.add_argument("--player-boxes", action="store_true",
                        help="Re-enable player bounding boxes (off by default).")
    parser.add_argument("--court-labels", action="store_true", help="Show court zone labels")
    parser.add_argument("--dump-bounces", metavar="PATH", default=None,
                        help="Also write detected bounces to PATH as a predictions JSON "
                             "(eval_bounces.py format) for scoring against ground truth. "
                             "Off by default; does not affect the annotated video.")
    parser.add_argument("--dump-ball-track", metavar="PATH", default=None,
                        help="Also write the LIVE tracked ball centers (raw, pre-spline) per "
                             "frame to PATH as JSON, for scoring against the ball GT "
                             "(tools/ball_gt/score_live_track.py). The honest end-to-end "
                             "thermometer for the ROI fallback. Off by default.")
    parser.add_argument("--demo-override", metavar="PATH", default=None,
                        help="Manual ground-truth sidecar (JSON from tools/annotate_demo.py) "
                             "that OVERRIDES this clip's bounces with bounces_true (replacing "
                             "the detector's) and adds racket->ball strikes (forehand/backhand) "
                             "as glow markers. Bounded to this clip only — no global threshold "
                             "changes (respects zero-hardcoding). demo_frame is 0-based within "
                             "the -s/-d window, matching the annotator.")
    parser.add_argument("--homography", action="store_true",
                        help="Compute a court homography (px↔meters) from court keypoints "
                             "for perspective-correct speed/depth. Falls back to the scalar "
                             "scale when the court can't be reliably localized (oblique/partial).")
    parser.add_argument("--track-camera", action="store_true",
                        help="Recompute the court homography at every frame so the overlay/"
                             "minimap/speed follow a panning/tilting camera (adds a Pass 0, "
                             "~14ms/frame). Implies --homography. Off by default (the f0 "
                             "homography is fine for a fixed camera).")
    parser.add_argument("--court-model", default="training/court/tennis_court_det.pt",
                        help="Path to the court-keypoint model weights (for --homography).")
    parser.add_argument("--auto-court", action="store_true",
                        help="Automatic CLASSICAL court homography (no GPU, no learned "
                             "model, license-clean): extract the white court lines → "
                             "vanishing-point pencils → Farin/gchlebus order-preserving "
                             "warp-back fit (models/court_auto.py). The deployable "
                             "replacement for the no-LICENSE keypoint model on oblique/"
                             "amateur/court-level angles. Runs AFTER --homography and "
                             "BEFORE the calibration sidecar in the cascade; ABSTAINS "
                             "honestly (→ sidecar/scalar) on grazing angles where the "
                             "depth axis is unobservable (e.g. felix). Off by default.")
    parser.add_argument("--ball-tracker", choices=["kalman", "wasb"], default="kalman",
                        help="Ball tracker. 'kalman' = YOLO26 + Kalman (default). 'wasb' = "
                             "WASB heatmap-temporal tracker (denser/cleaner trajectory, higher "
                             "bounce recall on broadcast; needs --wasb-weights).")
    parser.add_argument("--wasb-weights", default="training/wasb/wasb_tennis.pth.tar",
                        help="Path to WASB tennis weights (for --ball-tracker wasb).")
    parser.add_argument("--event-methodo", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Arbitrate bounces vs hits with the unified event classifier "
                             "(vision.events.classify_events) instead of running the two "
                             "detectors independently. The classifier merges the bounce "
                             "candidates, the trajectory turning points (incl. far-court apex "
                             "via detect_sharp_turns) and the wrist-speed hits, then labels "
                             "each event {BOUNCE|HIT} under a SCORE-FREE firewall + rally "
                             "alternation — confusion_H->B/B->H drop to 0 and far-court bounce "
                             "recall rises. ON by default (the firewall + vx-gate now reproduce "
                             "the 0/0 confusion on the LIVE prod track once far-pose is dense). "
                             "Pass --no-event-methodo for the strict legacy behavior "
                             "(vx_flip_veto + detect_hits, weakly arbitrated by a temporal "
                             "guard). Needs poses (Pass 1.5); on a pose-poor clip the classifier "
                             "degrades gracefully (felix: 0 events, parity with legacy).")
    parser.add_argument("--no-minimap", action="store_true",
                        help="Disable the 2D top-down court minimap (ball trajectory + bounces).")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Ignore any saved court calibration sidecar (<video>.court.json / "
                             "calibrations/<name>.court.json) used for the metric homography.")
    parser.add_argument("--no-fx", action="store_true",
                        help="Disable cinematic FX (glow trail, glass HUD, shockwaves). FX on by default.")
    parser.add_argument("--no-shot-guard", action="store_true",
                        help="Disable the court-visibility gate (shot-guard). By default, frames where the "
                             "full court is NOT visible (camera close-ups, replays, graphics) are detected and "
                             "CUT from the annotated output — no pose/ball/overlay/write spent on them (0 CPU) "
                             "and the video is shorter. A <output>_clipindex.json sidecar maps annotated→original "
                             "frames so the front re-aligns the timeline. The ORIGINAL video is never modified.")
    parser.add_argument("--match-mode", action="store_true",
                        help="Stable P1/P2 HUMAN identity across side-changes (match play). Without it, "
                             "P1=far / P2=near (net-side identity) — the training default, unchanged. With it, "
                             "players keep their id when they swap ends between games (HSV jersey re-id + "
                             "continuity), and each shot/rally carries a stable human_id on top of player_side.")
    args = parser.parse_args()
    # Player boxes default OFF (user feedback: boxes distract / a stale box looks
    # fixed). --player-boxes opts back in; --no-player-boxes is the explicit default.
    if not args.player_boxes:
        args.no_player_boxes = True
    return args


def main():
    args = parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(video_path).stem
        output_path = f"data/output/{stem}_annotated.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    trail_length = args.trail_length
    device = args.device

    # ─── Init models ───
    logger.info("Initializing models...")
    # Ball detection config resolution order (set the detector's effective
    # ball_conf/ball_imgsz AFTER a density probe below if --ball-auto is on and the
    # user did not pin them). Seed the detector with felix-safe legacy defaults so
    # it is valid even before the probe / when auto is off.
    _ball_conf_seed = args.ball_conf if args.ball_conf is not None else 0.16
    _ball_imgsz_seed = args.ball_imgsz if args.ball_imgsz is not None else 1280
    detector = TennisYOLODetector(
        conf_threshold=args.conf, iou_threshold=0.45, device=device, imgsz=1280,
        ball_conf=_ball_conf_seed, ball_imgsz=_ball_imgsz_seed,
    )
    logger.info(f"Ball detection (pre-probe): conf={detector.ball_conf}, "
                f"imgsz={detector.ball_imgsz} (persons: conf={args.conf}, imgsz=1280)")
    if args.pose_backend == "rtmw":
        from vision.pose_rtmw import RTMWPose
        logger.info("Pose backend: RTMW whole-body (rtmlib, CPU) — 133 kpts, slower")
        pose_model = RTMWPose(mode="balanced")
    else:
        pose_model = YOLO(args.pose_model)

    # ─── Read video info ───
    reader = VideoReader(video_path)
    fps = reader.fps
    frame_width = reader.width
    frame_height = reader.height
    total_frames = reader.frame_count

    # Compute frame range from start/end/duration
    start_frame = int(args.start * fps)
    if args.duration:
        end_frame = min(start_frame + int(args.duration * fps), total_frames)
    elif args.end:
        end_frame = min(int(args.end * fps), total_frames)
    else:
        end_frame = total_frames
    start_frame = max(0, min(start_frame, total_frames - 1))
    max_frames = end_frame - start_frame

    # Seek to start
    if start_frame > 0:
        reader.seek(start_frame)
        logger.info(f"Seeking to frame {start_frame} ({args.start:.1f}s)")

    # Pose image size: use large resolution for good detection of small/far players
    # For broadcast tennis, the far player is tiny — we need high resolution
    pose_imgsz = max(960, min(frame_width, 1920))
    # Round to nearest 32 (YOLO requirement)
    pose_imgsz = (pose_imgsz // 32) * 32

    # Ball jump threshold: relative to frame diagonal
    frame_diag = np.hypot(frame_width, frame_height)
    max_ball_jump = frame_diag * 0.15  # 15% of diagonal

    logger.info(f"Video: {frame_width}x{frame_height}, {fps:.1f}fps, {total_frames} frames")
    logger.info(f"Processing: frames {start_frame}-{end_frame} ({max_frames} frames, {max_frames/fps:.1f}s)")
    logger.info(f"Pose imgsz: {pose_imgsz}, ball jump threshold: {max_ball_jump:.0f}px")

    # ─── Read first frame (used for court homography + zones) ───
    first_frame = None
    if (not args.no_court or not args.no_bounces) or args.homography or args.auto_court:
        first_frame = next(reader.iter_frames())
        reader.release()
        reader = VideoReader(video_path)
        if start_frame > 0:
            reader.seek(start_frame)

    # ─── Court homography (px ↔ meters) via keypoints, for perspective-correct
    # speed & depth. Only used when it solves with high/low confidence; otherwise
    # the pipeline falls back to the scalar px-per-meter scale. Computed BEFORE
    # court zones because the zones are now DERIVED from it (the HF zone detector
    # suffers a domain shift on broadcast framings and returns {}). ───
    court_homography = None
    if args.homography and first_frame is not None:
        try:
            from models.court_detector import CourtKeypointDetector
            court_det = CourtKeypointDetector(args.court_model, device=device)
            court_homography = court_det.compute_homography(first_frame, frame_diag)
            logger.info(f"Court homography: confidence={court_homography['confidence']}, "
                        f"{court_homography['n_used']}/14 keypoints"
                        + (f", rms={court_homography['rms_px']:.1f}px"
                           if court_homography['rms_px'] else ""))
            if court_homography['confidence'] == 'fallback':
                logger.warning("Court homography unreliable on this framing "
                               "(oblique/partial court) → using scalar scale fallback")
                court_homography = None
        except Exception as e:
            logger.warning(f"Court homography failed ({e}) → scalar scale fallback")
            court_homography = None

    # ─── Automatic classical court homography (--auto-court, default OFF) ───
    # The no-GPU, license-clean replacement for the keypoint model on oblique /
    # amateur / court-level framings (where the keypoint model domain-shifts to
    # <4 points). models/court_auto.py extracts the white court lines, finds the
    # two vanishing-point pencils, and fits an image↔meters homography by the
    # Farin/gchlebus order-preserving warp-back self-score — with NO learned model.
    # It ABSTAINS honestly (confidence='fallback' → None here) on grazing angles
    # where the depth axis is unobservable (felix), deferring to the sidecar/scalar
    # below. Runs AFTER the keypoint solve (so a broadcast clip keeps using the
    # keypoint H it already nails) and BEFORE the sidecar (so a calibrated clip
    # still prefers its exact manual points). Same dict contract → drop-in.
    if court_homography is None and args.auto_court and first_frame is not None:
        try:
            from models.court_auto import AutoCourtDetector
            auto = AutoCourtDetector().compute_homography(first_frame, frame_diag)
            if auto is not None and auto.get("confidence") != "fallback":
                court_homography = auto
                logger.info(f"Auto court homography (classical, no-GPU): "
                            f"confidence={auto['confidence']}, n_used={auto['n_used']}, "
                            f"warp-back={auto.get('warpback_score', float('nan')):.2f} "
                            f"→ metric homography enabled")
            else:
                reason = (auto or {}).get("abstain_reason", "no court fit")
                logger.info(f"Auto court homography abstained ({reason}) "
                            "→ falling through to calibration sidecar / scalar")
        except Exception as e:
            logger.warning(f"Auto court homography failed ({e}) → sidecar/scalar fallback")

    # ─── Calibration homography (semi-auto 4-corner) ───
    # On oblique/amateur angles the keypoint model can't localize the court, but a
    # static camera only needs the court annotated ONCE. If a calibration sidecar
    # exists for this clip (<video>.court.json, or calibrations/<name>.court.json),
    # use it — it gives an exact metric homography where the keypoint model fails.
    # This is what makes the minimap metric on felix-style angles.
    if court_homography is None and not args.no_calibration:
        try:
            from models.court_calibrator import load_calibration
            cal = load_calibration(video_path)
            if cal is not None:
                court_homography = cal
                logger.info(f"Court calibration loaded ({cal['source']}): "
                            f"{cal['n_used']} points, rms={cal['rms_px']:.1f}px, "
                            f"confidence={cal['confidence']} → metric homography enabled")
                if cal['confidence'] == 'low':
                    logger.warning("Calibration is LOW confidence (grazing/oblique angle "
                                   "— far-field depth is approximate). Add more well-spread "
                                   "court points via tools/calibrate_court.py to improve.")
        except Exception as e:
            logger.warning(f"Court calibration load failed ({e})")

    # ─── Pass 0 (optional): per-frame homography for a moving camera. ───
    # A broadcast camera pans/tilts, so the f0 homography drifts (measured up to
    # ~140px on a 13s clip). When --track-camera is set, recompute the homography
    # at every frame (robust to hallucinated keypoints + hold on fallback) so the
    # court overlay / minimap / speed follow the camera. Pass 2 looks up the
    # per-frame H via _h_at. Off by default (fixed camera → f0 H is exact + cheap).
    court_homography_by_frame = None
    if args.track_camera:
        if not args.homography:
            args.homography = True  # --track-camera implies --homography
        if court_homography is None and first_frame is not None:
            # no f0 solve above (e.g. homography just forced) → solve f0 first
            try:
                from models.court_detector import CourtKeypointDetector
                court_det = CourtKeypointDetector(args.court_model, device=device)
                court_homography = court_det.compute_homography(first_frame, frame_diag)
                if court_homography['confidence'] == 'fallback':
                    court_homography = None
            except Exception as e:
                logger.warning(f"Court homography (track-camera f0) failed: {e}")
        try:
            from vision.court_track import compute_homography_series
            from models.court_detector import CourtKeypointDetector
            court_det = CourtKeypointDetector(args.court_model, device=device)
            rdr0 = VideoReader(video_path)
            if start_frame > 0:
                rdr0.seek(start_frame)
            pass0_frames = []
            for j, fr in enumerate(rdr0.iter_frames()):
                if j >= max_frames:
                    break
                pass0_frames.append(fr)
            rdr0.release()
            court_homography_by_frame = compute_homography_series(
                pass0_frames, court_det, frame_diag)
            logger.info(f"Pass 0: per-frame homography ready "
                        f"({sum(1 for f in court_homography_by_frame if f.H is not None)}/"
                        f"{len(court_homography_by_frame)} frames tracked)")
        except Exception as e:
            logger.warning(f"Pass 0 (track-camera) failed: {e} — falling back to f0 homography")
            court_homography_by_frame = None

    # ─── Court zones: DERIVE from the homography (exact perspective geometry)
    # when one is available; fall back to the HF YOLO zone detector only when
    # there is no homography at all. Derived zones are drop-in compatible
    # (same {name: {box, ...}} shape) and carry a `polygon` for a perspective
    # overlay. See vision/court_zones.py. ───
    court_zones = {}
    if not args.no_court or not args.no_bounces:
        from vision.court_zones import get_court_zones
        yolo_det = detector if not args.no_court else None
        court_zones = get_court_zones(first_frame, court_homography, yolo_det,
                                      conf_threshold=0.3)
        if court_zones:
            src = next(iter(court_zones.values())).get("source", "yolo")
            logger.info(f"Court zones ({src}): {len(court_zones)} — "
                        f"{list(court_zones.keys())}")
        else:
            logger.warning("No court zones (no homography and YOLO returned nothing)")

    # ─── Pass 0.5: shot-guard — court-visibility mask + CUT bookkeeping ───
    # By default, frames where the full court is not visible (close-ups, replays,
    # graphics) are CUT from the annotated output: every per-frame pass skips them
    # (no detect/pose/overlay/write → 0 CPU) and the video comes out shorter. A
    # clip-index sidecar maps annotated→original frames for the front. The ORIGINAL
    # video is only read, never rewritten. --no-shot-guard disables (mask all-True
    # → identity clip-index, nothing cut). See vision/shot_guard.py + annot_clip.py.
    shot_guard_enabled = not args.no_shot_guard
    # --event-methodo: arbitrate bounce-vs-hit with the unified classifier
    # (vision.events) AFTER Pass 1.5 (it needs poses + both candidate sets at once),
    # instead of running the two detectors independently. OFF (default) keeps the
    # legacy path bit-for-bit; ON defers the bounce_by_frame build past Pass 1.5.
    event_methodo = bool(args.event_methodo)
    court_visible_mask = [True] * max_frames        # window-relative
    non_court_spans_rel = []
    keep_set = set(range(max_frames))               # window-relative indices to annotate
    if shot_guard_enabled and max_frames > 0:
        sg_reader = VideoReader(video_path)
        if start_frame > 0:
            sg_reader.seek(start_frame)
        def _sg_frames():
            for j, fr in enumerate(sg_reader.iter_frames()):
                if j >= max_frames:
                    break
                yield fr
        court_visible_mask = compute_court_visibility_mask_streaming(_sg_frames(), fps)
        sg_reader.release()
        non_court_spans_rel = non_court_spans(court_visible_mask)
        keep_set = set(frames_to_annotate(court_visible_mask))
        n_keep = len(keep_set)
        logger.info(f"=== Pass 0.5 (shot-guard CUT): annotating {n_keep}/{max_frames} frames "
                    f"({100 * n_keep / max(1, max_frames):.1f}%), "
                    f"{max_frames - n_keep} cut, {len(non_court_spans_rel)} non-court spans ===")

    # ─── Pass 1: Detect ball positions + bounces ───
    logger.info("=== PASS 1: Detect ball ===")

    # ─── Density-adaptive ball resolution/conf (--ball-auto, default ON) ───
    # The tiny/far ball wins big from a high-res low-conf pass (1920/0.05) on CLEAN
    # broadcast, but the SAME config floods a parasite-heavy oblique/amateur clip
    # with far-field false candidates the Kalman locks onto (measured: demo3 ~2
    # cand/frame -> 1920/0.05 lifts coverage 86%->96% & CORRECT 79%->92%; felix ~7
    # cand/frame -> 1920 collapses its bounce track). We probe the mean candidate
    # density on a cheap subsample at a low floor and pick the path accordingly —
    # a threshold on a DETECTED feature, not a per-video constant (demo3 2.0 /
    # tennis 2.5 / felix 6.9, robust 3.0-5.0). Skipped if auto is off, the user
    # pinned --ball-imgsz / --ball-conf (explicit always wins), or the WASB tracker
    # is selected (its heatmap path ignores YOLO conf/imgsz, so probing is wasted).
    ball_clean = None  # set by the density probe below; gates the ROI fallback too
    if (args.ball_auto and args.ball_tracker != "wasb"
            and (args.ball_imgsz is None or args.ball_conf is None)):
        probe_reader = VideoReader(video_path)
        if start_frame > 0:
            probe_reader.seek(start_frame)
        probe_step = 10  # keep only every 10th frame (probe is subsample-stable)
        probe_frames = []
        for i, f in enumerate(probe_reader.iter_frames()):
            if i >= max_frames:
                break
            if i % probe_step == 0:
                probe_frames.append(f)
        probe_reader.release()
        # frames are pre-subsampled, so probe at subsample=1
        density = detector.probe_ball_density(probe_frames, probe_conf=0.05,
                                              probe_imgsz=1280, subsample=1)
        clean = density <= args.ball_density_thr
        ball_clean = clean
        # Explicit user values always win; auto only fills the unset ones.
        auto_imgsz = 1920 if clean else 1280
        auto_conf = 0.05 if clean else 0.16
        if args.ball_imgsz is None:
            detector.ball_imgsz = auto_imgsz
        if args.ball_conf is None:
            detector.ball_conf = auto_conf
        logger.info(
            f"Ball-auto: density={density:.2f} (thr={args.ball_density_thr}) → "
            f"{'CLEAN broadcast' if clean else 'PARASITE-heavy'} → "
            f"imgsz={detector.ball_imgsz}, conf={detector.ball_conf}")
        # Re-seek the main reader to the range start for the detection pass.
        reader.release()
        reader = VideoReader(video_path)
        if start_frame > 0:
            reader.seek(start_frame)

    logger.info(f"Ball detection: conf={detector.ball_conf}, imgsz={detector.ball_imgsz} "
                f"(persons: conf={args.conf}, imgsz=1280)")
    # The ball model already ran at detector.ball_conf (the real, low floor); this
    # secondary gate stays equal to it so it never re-filters at a higher value
    # (a stale 0.15 here would silently undo the low-conf densification). We keep
    # the real ball by MOTION (Kalman gating) + static-FP, not by confidence.
    ball_conf_thresh = detector.ball_conf

    # ─── Dynamic-ROI ball fallback gating (search-region tracking) ───
    # Fires only when full-frame detection misses on a frame the Kalman can predict
    # (a gap). Gated to the CLEAN broadcast density path by default: a parasite-heavy
    # oblique clip (felix) would get extra far-field candidates from the crop that
    # could leak through the gate, so we don't fire it there unless --ball-roi-all.
    # When --ball-auto ran, ball_clean is the measured density verdict; when auto is
    # off / pinned, treat a high-res ball pass (>=1920) as the clean-broadcast intent.
    if ball_clean is None:
        _clean_path = detector.ball_imgsz >= 1920
    else:
        _clean_path = ball_clean
    ball_roi_active = (args.ball_roi and not (args.ball_tracker == "wasb")
                       and (args.ball_roi_all or _clean_path))
    if ball_roi_active:
        ball_roi_half = max(20, int(args.ball_roi_half_frac * frame_width))
        logger.info(
            f"Ball ROI fallback: ON (half={ball_roi_half}px="
            f"{args.ball_roi_half_frac:.3f}W, zoom={args.ball_roi_zoom}, "
            f"gate×{args.ball_roi_gate_frac}, min_conf={args.ball_roi_min_conf}, "
            f"path={'clean' if _clean_path else 'parasite (--ball-roi-all)'})")

    # ─── WASB heatmap tracker path (--ball-tracker wasb) ───
    # A heatmap-temporal tracker (WASB) produces a far denser, cleaner trajectory
    # than YOLO+Kalman (no static-FP/reinit jitter), which is the real lever for
    # bounce recall. We buffer frames, run the tracker, and feed all_ball_centers
    # straight into the (robust) bounce detector, skipping static-FP + Kalman.
    wasb_path = (args.ball_tracker == "wasb")
    if wasb_path:
        try:
            from models.wasb_ball_tracker import HeatmapBallTracker
            wasb_frames = []
            for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 1a: read"):
                if len(wasb_frames) >= max_frames:
                    break
                wasb_frames.append(frame)
            reader.release()
            tracker = HeatmapBallTracker(args.wasb_weights, device=device)
            logger.info("Tracking ball with WASB heatmap model...")
            raw_ball_centers = tracker.track_ball(wasb_frames)
            # shot-guard CUT: WASB can hallucinate a heatmap blob on a close-up;
            # null its output on non-court frames so no ball is fabricated there.
            if shot_guard_enabled:
                raw_ball_centers = [
                    None if (i not in keep_set) else c
                    for i, c in enumerate(raw_ball_centers)]
            ball_speeds_px = [0.0] * len(raw_ball_centers)  # filled below
            raw_detections = None
        except Exception as e:
            logger.warning(f"WASB tracker failed ({e}) → falling back to YOLO+Kalman")
            wasb_path = False
            reader = VideoReader(video_path)
            if start_frame > 0:
                reader.seek(start_frame)

    # Step 1a: Collect ALL ball candidates per frame (YOLO+Kalman path)
    if not wasb_path:
        raw_detections = []  # list of list of (cx, cy, score)
        # When the dynamic-ROI fallback is active we keep the decoded frames so
        # Pass 1b can re-crop them around the Kalman prediction on a gap frame.
        # (The WASB path already buffers all frames; this mirrors that pattern.)
        roi_frames = [] if ball_roi_active else None
        frame_count = 0
        for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 1a: detect"):
            if frame_count >= max_frames:
                break
            if roi_frames is not None:
                roi_frames.append(frame)
            # shot-guard CUT: no detection on a non-court frame (it won't be written
            # nor counted). Keep the per-frame array aligned with an empty slot.
            if shot_guard_enabled and frame_count not in keep_set:
                raw_detections.append([])
                frame_count += 1
                continue
            det = detector.detect(frame, classes=[32])
            ball_mask = det["class_ids"] == 32
            candidates = []
            if ball_mask.any():
                boxes = det["boxes"][ball_mask]
                scores = det["scores"][ball_mask]
                for i in range(len(boxes)):
                    if scores[i] >= ball_conf_thresh:
                        bx = boxes[i]
                        cx, cy = int((bx[0]+bx[2])/2), int((bx[1]+bx[3])/2)
                        candidates.append((cx, cy, float(scores[i])))
            raw_detections.append(candidates)
            frame_count += 1
        reader.release()

    # The static-FP filter + Kalman tracking only run on the YOLO path; WASB
    # already produced raw_ball_centers directly.
    if not wasb_path:
        # Step 1b: Identify static false positives (logos, court marks)
        # A point that appears at ~same location in >8% of frames is static
        from collections import Counter
        static_radius = 20  # pixels — detections within this radius are "same spot"
        all_points = []
        for candidates in raw_detections:
            for cx, cy, _ in candidates:
                # Quantize to grid
                all_points.append((cx // static_radius, cy // static_radius))

        point_counts = Counter(all_points)
        static_threshold = max(10, int(max_frames * 0.08))  # 8% of frames
        static_zones = set()
        for (gx, gy), count in point_counts.items():
            if count >= static_threshold:
                static_zones.add((gx, gy))

        if static_zones:
            logger.info(f"Filtered {len(static_zones)} static false positive zones")

        def is_static(cx, cy):
            gx, gy = cx // static_radius, cy // static_radius
            # Check the cell and neighbors
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (gx + dx, gy + dy) in static_zones:
                        return True
            return False

        # Step 1c: Kalman filter on cleaned detections
        raw_ball_centers = []
        ball_speeds_px = []  # speed in pixels/frame from Kalman state

        ball_kf = BallKalmanFilter(
            process_noise=200.0,
            measurement_noise=8.0
        )
        # Gate grows with frames_since_update: ball moves further when we miss frames.
        # Tuned against the felix.mp4 bounce ground truth: the looser reinit + wider
        # gate + shorter coast recover the trajectory through direction reversals
        # (raising GT-window coverage to 16/16 and bounce recall from 0.25 to ~0.75).
        kf_gate_base = 150.0              # base gate in pixels for 1-frame gap
        kf_gate_growth = 80.0            # extra pixels per missed frame
        kf_gate_max = frame_diag * 0.35   # never exceed 35% of diagonal
        kf_max_coast = int(fps * 0.12)    # coast ~0.12s then attempt reinit
        last_known_pos = None              # last accepted ball position for reinit check

        for frame_idx in tqdm(range(len(raw_detections)), desc="Pass 1b: track"):
            candidates = raw_detections[frame_idx]

            # Filter out static false positives
            moving = [(cx, cy, sc) for cx, cy, sc in candidates if not is_static(cx, cy)]

            ball_center = None
            predicted = ball_kf.predict()

            # Adaptive gate: wider when we've been missing frames
            current_gate = min(kf_gate_max,
                               kf_gate_base + kf_gate_growth * ball_kf.frames_since_update)

            if moving:
                if not ball_kf.initialized:
                    # Pick highest confidence moving detection to init
                    best = max(moving, key=lambda c: c[2])
                    ball_kf.init_state(best[0], best[1])
                    ball_center = (best[0], best[1])
                else:
                    # Pick the moving detection closest to Kalman prediction
                    best_det = None
                    best_dist = float('inf')
                    for cx, cy, sc in moving:
                        dist = ball_kf.gating_distance(cx, cy)
                        if dist < current_gate and dist < best_dist:
                            best_dist = dist
                            best_det = (cx, cy)

                    # ─── Dynamic-ROI fallback (search-region tracking) ───
                    # Full-frame gave no in-gate candidate on a frame the Kalman can
                    # predict → crop a window around the prediction, upscale, re-detect.
                    # The ball is larger/sharper there and off-court parasites are
                    # excluded. Recovered candidates are gated through the SAME Kalman
                    # gate (TIGHTER, see below), so a spurious crop hit far from the
                    # prediction is rejected exactly like a full-frame parasite. Fires
                    # ONLY on this gap (~32% of frames on a clean clip → ~+15-25% of the
                    # ball pass). See docs/research/ball-roi-dynamic-CR.md.
                    if (best_det is None and ball_roi_active
                            and roi_frames is not None and predicted is not None):
                        # ROI candidate must clear a HIGHER conf bar (crop upscaling
                        # inflates faint parasites) and a TIGHTER gate (0.25× the grown
                        # full gate) — an off-target ROI hit poisons the Kalman state.
                        roi_conf = max(ball_conf_thresh, args.ball_roi_min_conf)
                        roi_gate = current_gate * args.ball_roi_gate_frac
                        rcands = detector.detect_ball_roi(
                            roi_frames[frame_idx], predicted[0], predicted[1],
                            half=ball_roi_half, zoom=args.ball_roi_zoom,
                            conf=roi_conf)
                        for r in rcands:
                            rcx, rcy = r["cx"], r["cy"]
                            if is_static(rcx, rcy):
                                continue
                            dist = ball_kf.gating_distance(rcx, rcy)
                            if dist < roi_gate and dist < best_dist:
                                best_dist = dist
                                best_det = (rcx, rcy)

                    if best_det is not None:
                        # Good match — update Kalman
                        ball_kf.update(best_det[0], best_det[1])
                        state = ball_kf.kf.statePost
                        ball_center = (int(state[0][0]), int(state[1][0]))
                    elif ball_kf.frames_since_update <= kf_max_coast and predicted:
                        # No match but still coasting — use prediction
                        ball_center = (int(predicted[0]), int(predicted[1]))
                    elif last_known_pos is not None:
                        # Coast expired — only reinit if a detection is near last known pos
                        nearest = None
                        nearest_dist = float('inf')
                        reinit_gate = frame_diag * 0.50
                        for cx, cy, sc in moving:
                            d = np.hypot(cx - last_known_pos[0], cy - last_known_pos[1])
                            if d < reinit_gate and d < nearest_dist:
                                nearest_dist = d
                                nearest = (cx, cy)
                        if nearest is not None:
                            ball_kf.init_state(nearest[0], nearest[1])
                            ball_center = nearest
                        # else: no valid reinit — skip frame

            if ball_center is None and ball_kf.initialized:
                if ball_kf.frames_since_update <= kf_max_coast and predicted:
                    ball_center = (int(predicted[0]), int(predicted[1]))

            if ball_center is not None:
                last_known_pos = ball_center

            # Extract speed from Kalman state (pixels/frame)
            if ball_kf.initialized:
                vx = float(ball_kf.kf.statePost[2][0])
                vy = float(ball_kf.kf.statePost[3][0])
                speed_px = np.hypot(vx, vy)
            else:
                speed_px = 0.0
            ball_speeds_px.append(speed_px)

            raw_ball_centers.append(ball_center)

        # Free the ROI frame buffer immediately — it is only needed during Pass 1b
        # (the ROI re-crops the decoded frame at the gap). Holding every decoded BGR
        # frame would otherwise scale memory with clip length (~6 MB/frame at 1080p);
        # releasing here caps the peak to the detection pass, not the whole run.
        roi_frames = None

        # ─── S6: reject "teleport-then-frozen" static-FP blobs (determinism) ───
        # A transient corner/field FP the track can lock onto: the trajectory
        # teleports into a frozen cluster (>0.15·diag jump → ≥0.16s frozen). Read
        # by vision/events.py it fabricates a vy_flip → a MANDATORY BOUNCE anchor
        # that floods the alternation DP and makes the event F1 swing run-to-run
        # ("phantom bounces that come and go"). We delete it from the RAW track
        # here, BEFORE the spline / is_real mask / dump / events consume it, so the
        # fix is deterministic and reaches every downstream consumer. The reject is
        # a strict conjunction (teleport-entry + frozen + healthy-moving-before),
        # never "frozen alone" — a real apex/toss is also near-frozen but enters
        # from a coast (low pre-jump motion) and is kept (see the function doc +
        # tests/test_reinit_frozen_fp.py). GATED to the CLEAN-broadcast density
        # path (like the ROI fallback): on a parasite-heavy oblique clip (felix)
        # the far-court ball genuinely hovers near the baseline and a big apparent
        # jump is real perspective motion, so the reject must not run there (it
        # would delete real far-apex bounces — measured: 11 felix deletions incl.
        # GT bounce f53). See docs/research/s6-balltrack-staticfp-CR.md.
        if _clean_path:
            raw_ball_centers, _s6_deleted = reject_teleport_frozen_blobs(
                raw_ball_centers, frame_diag, fps)
            if _s6_deleted:
                logger.info(
                    f"S6 static-FP reject: removed {len(_s6_deleted)} "
                    f"teleport-then-frozen blob(s) "
                    + ", ".join(f"f{b['start']}-{b['end']}@{b['centroid']}"
                                f"(in {b['entry_jump_frac']} out {b.get('exit_jump_frac')}"
                                f" ·diag, {b['tier']})"
                                for b in _s6_deleted))

            # ─── S6: reject OFF-FRAME coast predictions (determinism) ───
            # When the ball leaves the frame (or the Kalman prediction diverges) the
            # tracker coasts the prediction OFF the frame edge (measured: f150-153
            # coast to y=-62..-359, hundreds of px above the top edge, then
            # re-acquires the real ball). A real ball is ALWAYS within [0,W)×[0,H) —
            # GT confirms every verified ball point is in-frame — so an off-frame
            # center can never be correct and only feeds a fake direction reversal →
            # a phantom BOUNCE (B@152 on the canonical clip). Null them: 0 GT points
            # lost by construction. Clean-path gated (same block as the blob reject)
            # so the felix bounce track stays byte-identical.
            _s6_off = 0
            for _f in range(len(raw_ball_centers)):
                _c = raw_ball_centers[_f]
                if _c is not None and (_c[0] < 0 or _c[1] < 0
                                       or _c[0] >= frame_width or _c[1] >= frame_height):
                    raw_ball_centers[_f] = None
                    _s6_off += 1
            if _s6_off:
                logger.info(f"S6 off-frame reject: nulled {_s6_off} off-frame coast point(s)")

    # ─── CHANTIER 0: pre-spline real/interpolated mask (for the vx-flip veto) ───
    # The vx-flip bounce-vs-hit veto MUST read direction on the RAW tracker
    # centers (pre-spline) with a real/interpolated mask — never on the spline,
    # which rounds the reflection and reads through filled gaps. raw_ball_centers
    # is complete here and is NOT overwritten by the spline below (a fresh list is
    # returned), so it stays available at the bounce call site. See
    # docs/research/bounce-vs-shot-direction.md.
    ball_is_real = [c is not None for c in raw_ball_centers]

    # ─── Optional: dump the LIVE tracked ball centers for GT scoring ───
    # The honest end-to-end thermometer: the ACTUAL prod track (raw, pre-spline —
    # what tools/ball_gt/measure_ball_coverage.py scores), so a lab measurement can
    # be confirmed against a real pipeline run (not a frozen cache).
    if getattr(args, "dump_ball_track", None):
        import json as _json
        with open(args.dump_ball_track, "w") as _f:
            _json.dump({
                "start_frame": int(start_frame),
                "fps": float(fps),
                "width": int(frame_width), "height": int(frame_height),
                "centers": [None if c is None else [float(c[0]), float(c[1])]
                            for c in raw_ball_centers],
            }, _f)
        logger.info(f"Dumped {sum(1 for c in raw_ball_centers if c is not None)}/"
                    f"{len(raw_ball_centers)} live ball centers -> {args.dump_ball_track}")

    # ─── Post-process ball: cubic spline interpolation ───
    interp_gap = int(fps * 0.4)  # interpolate gaps up to 0.4s
    all_ball_centers = smooth_ball_trajectory(raw_ball_centers, max_gap=interp_gap)
    filled = sum(1 for a, b in zip(raw_ball_centers, all_ball_centers) if a is None and b is not None)
    detected = sum(1 for c in raw_ball_centers if c is not None)
    logger.info(f"Ball tracking: {detected}/{len(raw_ball_centers)} detected, "
                f"{filled} interpolated, "
                f"{sum(1 for c in all_ball_centers if c is not None)}/{len(all_ball_centers)} total")

    # ─── Convert speeds to km/h ───
    # Primary path: a court homography maps ball positions to real court METERS,
    # which is perspective-correct (a single px-per-meter scalar is geometrically
    # wrong everywhere except one depth). Fallback: the scalar scale.
    px_per_m = estimate_px_per_meter(court_zones, frame_height)
    ball_speeds_kmh = []
    speed_source = "scalar"
    if court_homography is not None and court_homography.get("H") is not None:
        from models.court_detector import img_to_world
        H = court_homography["H"]
        speed_source = f"homography({court_homography['confidence']})"
        # speed at frame i from the metric displacement between consecutive centers
        prev_world = None
        world_pts = []
        for c in all_ball_centers:
            world_pts.append(img_to_world(H, c[0], c[1]) if c is not None else None)
        for i in range(len(all_ball_centers)):
            if (i > 0 and world_pts[i] is not None and world_pts[i - 1] is not None):
                dx = world_pts[i][0] - world_pts[i - 1][0]
                dy = world_pts[i][1] - world_pts[i - 1][1]
                dist_m = np.hypot(dx, dy)
                ball_speeds_kmh.append(dist_m * fps * 3.6)
            else:
                ball_speeds_kmh.append(0.0)
        # reject physically-impossible spikes from tracking jitter (a tennis ball
        # tops out ~260 km/h on a pro serve); clamp before smoothing so one bad
        # metric jump doesn't poison the HUD/stats.
        MAX_PLAUSIBLE_KMH = 280.0
        ball_speeds_kmh = [s if s <= MAX_PLAUSIBLE_KMH else 0.0 for s in ball_speeds_kmh]
        # light smoothing to tame per-frame jitter
        if len(ball_speeds_kmh) >= 5:
            k = 5
            sm = np.convolve(ball_speeds_kmh, np.ones(k) / k, mode="same")
            ball_speeds_kmh = list(sm)
    else:
        # Scalar px→m fallback. Compute per-frame speed from the TRAJECTORY pixel
        # displacement (works for both WASB and Kalman trackers; ball_speeds_px is
        # only populated on the Kalman path). Perspective makes this approximate,
        # but it's the honest fallback when no homography is available.
        raw_kmh = [0.0]
        for i in range(1, len(all_ball_centers)):
            a, b = all_ball_centers[i - 1], all_ball_centers[i]
            if a is not None and b is not None:
                dpx = np.hypot(b[0] - a[0], b[1] - a[1])
                raw_kmh.append((dpx / px_per_m) * fps * 3.6)
            else:
                raw_kmh.append(0.0)
        MAX_PLAUSIBLE_KMH = 280.0
        raw_kmh = [s if s <= MAX_PLAUSIBLE_KMH else 0.0 for s in raw_kmh]
        if len(raw_kmh) >= 5:
            raw_kmh = list(np.convolve(raw_kmh, np.ones(5) / 5, mode="same"))
        ball_speeds_kmh = raw_kmh
    logger.info(f"Speed source: {speed_source}")
    valid_speeds = [s for s in ball_speeds_kmh if s > 5.0]
    if valid_speeds:
        logger.info(f"Ball speed stats: avg={np.mean(valid_speeds):.0f} km/h, "
                    f"max={np.max(valid_speeds):.0f} km/h, "
                    f"min={np.min(valid_speeds):.0f} km/h")

    # ─── Detect bounces from smoothed trajectory ───
    # On the WASB (dense heatmap) trajectory, use the robust detector: it despikes
    # outliers and uses IRLS fits + an energy hit-vs-bounce test, which gives the
    # best recall on a dense trajectory (felix: F1 0.865 vs 0.79). On the sparser
    # YOLO+Kalman trajectory the curve-fit detector is the tuned default.
    bounce_events = []
    direction_state_by_frame = {}  # frame -> {"metric","unconfirmed"} (CHANTIER 2)
    turn_bounce_frames = []        # LEGACY: turn-angle-sourced (lower-conf) bounces
    # --event-methodo: the unified classifier (vision.events) needs the RAW,
    # un-vetoed bounce candidates PLUS poses + hits, which only exist after Pass
    # 1.5. So we SKIP the Pass-1 bounce detection/veto entirely here and generate
    # the candidates + arbitrate after Pass 1.5 (the classifier owns the firewall,
    # so no veto is applied upstream). This also keeps all_ball_centers as the
    # clean pre-robust spline track for the methodo's smoothed_centers (the WASB
    # branch below re-smooths it in place, which the methodo must not see).
    if not args.no_bounces and not event_methodo:
        if wasb_path:
            from vision.bounce import detect_bounces_robust, vx_flip_veto
            bounce_events, all_ball_centers, _ = detect_bounces_robust(
                all_ball_centers, ball_speeds_px, fps, frame_height, frame_width)
            # CHANTIER 1: one-directional vx-flip veto (bounce-vs-hit by direction).
            # Reads the RAW pre-spline tracker centers + is_real mask (NOT the
            # detector's re_smoothed array, which the line above just assigned to
            # all_ball_centers). One-directional: only removes a candidate whose
            # horizontal velocity sign confidently flips (a racket hit); abstains
            # on every doubt → can only add precision, never drop a real bounce.
            bounce_events, direction_state_by_frame = vx_flip_veto(
                bounce_events, raw_ball_centers, ball_is_real, fps, frame_width)
        else:
            # CHANTIER 3: the Kalman / curve-fit path exposes no raw-centers+mask
            # contract (detect_bounces_from_trajectory returns only the chosen
            # frames), so the vx-flip veto is NOT applied here. Applying it on a
            # re-derived spline would read direction through filled gaps where
            # vy-flip recall is only 62-73% → risk of dropping below the 0.72
            # bounce floor. Documented debt: a future CHANTIER 0 for this path
            # would thread its pre-spline trajectory + mask before vetoing.
            #
            # LEGACY SEPARATION (feat/legacy-bounce-shot-sep): we DO thread the raw
            # pre-spline centers + is_real mask here so the curve-fit detector also
            # admits far-court APEX bounces from the velocity-vector turn angle
            # (no y-extremum → the strict y-max test alone found only ~1/9 on the
            # demo3 broadcast clip). Each turn candidate is validated by an on-court
            # band + vy-reflection + vx-not-flipped check (a vx flip is a hit), so
            # recall rises without flooding FPs and without confusing hits as
            # bounces. demo3 legacy: bounce F1 0.20→0.80; the 3 confusion_B→H vanish
            # (real bounces are now claimed as bounces, not hits). On felix the turn
            # branch (which this prod path DOES enter, unlike the test's no-arg call)
            # adds one mid-court turn FP → F1 0.800→0.774, still well above the 0.72
            # floor: the turn pass is GAP-FILLER-only (never displaces a refined y-V
            # bounce in NMS) and felix's clean y-peaks carry recall, so the cost is a
            # single FP — a fair cross-clip trade for demo3's 1/9→6/9. Both the
            # no-arg (test) and with-arg (prod) felix paths are floored in
            # test_bounce_regression.
            bounce_events, turn_bounce_frames = detect_bounces_from_trajectory(
                all_ball_centers, ball_speeds_px, fps, frame_height, frame_width,
                raw_centers=raw_ball_centers, is_real=ball_is_real,
                return_turn_frames=True)

    # ─── Optional: export bounce predictions for offline evaluation ───
    # Additive, gated, read-only w.r.t. the pipeline: reads the already-computed
    # bounce_events and writes the eval_bounces.py schema. frame_idx is 0-based
    # within the processed window, so absolute frame = start_frame + frame_idx
    # (matches the annotator's convention -> aligns with ground truth).
    # In the --event-methodo path bounce_events is still empty here (detection
    # deferred to the post-Pass-1.5 arbitration), which dumps the arbitrated
    # bounces itself — so skip this empty Pass-1 dump in that path.
    if args.dump_bounces and not event_methodo:
        pred_doc = {
            "video": video_path,
            "fps": fps,
            "frame_offset": start_frame,
            "annotator": "run_pipeline_8s.py:detect_bounces_from_trajectory",
            "notes": "auto-exported bounce predictions",
            "bounces": [
                {"frame": int(start_frame + idx), "x": int(x), "y": int(y)}
                for (idx, x, y) in bounce_events
            ],
        }
        Path(args.dump_bounces).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_bounces, "w") as f:
            json.dump(pred_doc, f, indent=2)
            f.write("\n")
        logger.info(f"Dumped {len(bounce_events)} bounce predictions -> {args.dump_bounces}")

    # ─── Demo override: replace detected bounces with the manual ground truth ───
    # A curated clip (e.g. the landing-page hero) can override the detector's
    # bounces with hand-annotated ones (the detector misses ~6/9 far-side
    # bounces on this broadcast clip) and add racket->ball strikes. demo_frame
    # in the sidecar is 0-based within the -s/-d window, i.e. it maps directly
    # to the segment frame index used below. Bounded to this clip only — no
    # global thresholds move (respects zero-hardcoding).
    override_strikes = {}  # segment_frame -> {"xy": (x,y), "type": "forehand"|"backhand"}
    if args.demo_override:
        with open(args.demo_override) as _f:
            ov = json.load(_f)
        gt_bounces = ov.get("bounces_true", [])
        if gt_bounces:
            # bounces_false let us suppress specific detector hits by proximity.
            suppress_xy = [tuple(b["xy"]) for b in ov.get("bounces_false", [])]
            override_events = [
                (int(b["demo_frame"]), float(b["xy"][0]), float(b["xy"][1]))
                for b in gt_bounces
            ]
            logger.info(f"Demo override: replacing {len(bounce_events)} detected "
                        f"bounces with {len(override_events)} manual GT bounces.")
            bounce_events = override_events
        for s in ov.get("strikes", []):
            override_strikes[int(s["demo_frame"])] = {
                "xy": (float(s["xy"][0]), float(s["xy"][1])),
                "type": s.get("type", "forehand"),
            }
        if override_strikes:
            logger.info(f"Demo override: {len(override_strikes)} racket->ball strikes "
                        f"({sum(1 for v in override_strikes.values() if v['type']=='forehand')} "
                        f"forehand, {sum(1 for v in override_strikes.values() if v['type']=='backhand')} backhand).")

    # ─── Compute quality-score constants (needed by bounces AND shots) ───
    # speed_window: average speed over ±0.3s around the bounce
    speed_window = int(fps * 0.3)
    # Reference speed: 90th percentile of all valid speeds
    valid_speeds_arr = np.array([s for s in ball_speeds_kmh if s > 5.0])
    speed_ref = float(np.percentile(valid_speeds_arr, 90)) if len(valid_speeds_arr) > 0 else 100.0
    speed_ref = max(speed_ref, 30.0)  # floor to avoid division issues

    depth_score_map = {"deep": 1.0, "mid": 0.5, "short": 0.2}

    def _build_bounce_by_frame(events):
        """Classify depth + score quality for a list of (frame, x, y) bounces →
        {frame: (x, y, depth, avg_speed, quality)}. Shared by the legacy path and
        the --event-methodo arbitration so the bounce record shape (and the exact
        depth/quality math) is identical regardless of how the events were chosen.
        """
        classified = classify_bounce_depth(events, court_zones, frame_height) if events else []
        if classified:
            dc = {}
            for _, _, _, depth in classified:
                dc[depth] = dc.get(depth, 0) + 1
            logger.info(f"Bounce detection: {len(classified)} bounces — {dc}")
        else:
            logger.info("Bounce detection: 0 bounces detected")
        out = {}
        for fidx, bx, by, depth in classified:
            lo = max(0, fidx - speed_window)
            hi = min(len(ball_speeds_kmh), fidx + speed_window + 1)
            window_speeds = [ball_speeds_kmh[i] for i in range(lo, hi) if ball_speeds_kmh[i] > 5.0]
            avg_speed = float(np.mean(window_speeds)) if window_speeds else 0.0
            speed_norm = min(1.0, avg_speed / speed_ref)
            depth_norm = depth_score_map.get(depth, 0.5)
            quality = max(0, min(100, int((0.4 * speed_norm + 0.6 * depth_norm) * 100)))
            out[fidx] = (bx, by, depth, avg_speed, quality)
            logger.debug(f"Bounce frame {fidx}: depth={depth}, speed={avg_speed:.0f}km/h, Q={quality}")
        return out

    # Legacy path builds bounce_by_frame now. The --event-methodo path leaves it
    # empty here (bounce_events is still empty — Pass 1 detection was skipped) and
    # rebuilds it after Pass 1.5 from the arbitrated events, via the same helper.
    bounce_by_frame = _build_bounce_by_frame(bounce_events)

    # ─── Pass 1.5: Pose on every frame → locked P1/P2 skeleton tracks ───
    # The per-frame pose estimator returns "the 2 players this frame" with no
    # temporal identity. We run it on EVERY frame once, then feed the results to
    # TwoPlayerTracker which assigns persistent P1(far)/P2(near) identities,
    # associates frame-to-frame, GAP-FILLS missed detections, and smooths — so
    # every frame has a stable, jitter-free skeleton for both players. This single
    # pose pass is reused by the shot pre-pass and Pass 2 (no redundant inference).
    logger.info("=== PASS 1.5: Pose + lock P1/P2 ===")
    from vision.player_track import TwoPlayerTracker
    match_mode = bool(args.match_mode)
    if match_mode:
        from vision.match_tracker import MatchTracker, human_id_for_side
    players_per_frame = [None] * max_frames
    rdr = VideoReader(video_path)
    if start_frame > 0:
        rdr.seek(start_frame)
    net_y_seed = None
    if court_zones:
        from vision.pose import _net_y
        net_y_seed = _net_y(court_zones, frame_height)
    if match_mode:
        # In match mode P1/P2 are STABLE HUMANS (not net sides): a side-change
        # between games no longer inverts the identity, so per-human stats stay
        # correct across the whole clip. See vision/match_tracker.py.
        ptracker = MatchTracker(frame_width, frame_height, net_y_seed, fps)
    else:
        ptracker = TwoPlayerTracker(frame_width, frame_height, net_y_seed, fps)
    for i, frame in enumerate(tqdm(rdr.iter_frames(), total=max_frames,
                                   desc="Pass 1.5: pose")):
        if i >= max_frames:
            break
        # shot-guard CUT: no pose on a non-court frame (no skeleton will be drawn
        # nor written). Empty list keeps the tracker + per-frame array aligned.
        if shot_guard_enabled and i not in keep_set:
            players = []
        else:
            players = estimate_players_pose(
                detector.person_model, pose_model, frame, device,
                ball_xy=all_ball_centers[i] if i < len(all_ball_centers) else None,
                court_zones=court_zones if court_zones else None)
        players_per_frame[i] = players
        if match_mode:
            ptracker.add_frame(players, frame)   # frame needed for HSV re-id
        else:
            ptracker.add_frame(players)
    rdr.release()
    # locked, gap-filled, smoothed tracks: {1: [pose|None]*N, 2: [pose|None]*N}.
    # In match mode keys are stable HUMAN ids; in training mode they are sides.
    match_result = None
    if match_mode:
        match_result = ptracker.finalize()
        player_tracks = match_result["tracks"]
        n_swaps = len(match_result["swaps"])
        c1 = (sum(1 for p in player_tracks[1] if p is not None) / len(player_tracks[1])
              if player_tracks[1] else 0.0)
        c2 = (sum(1 for p in player_tracks[2] if p is not None) / len(player_tracks[2])
              if player_tracks[2] else 0.0)
        logger.info(f"Player tracks locked (MATCH MODE): P1 {c1*100:.1f}%, "
                    f"P2 {c2*100:.1f}%, side-swaps detected: {n_swaps}")
    else:
        player_tracks = ptracker.finalize()
        cov = ptracker.coverage()
        logger.info(f"Player tracks locked: P1 coverage={cov[1]*100:.1f}%, "
                    f"P2 coverage={cov[2]*100:.1f}% (gap-filled)")

    # ─── Shot pre-pass: detect hits, attribute to player, classify FH/BH, score ───
    # A hit is a ball vertical-direction reversal near a player. Reuses the Pass 1.5
    # poses (no extra inference), attributes + classifies, then links each hit to
    # the next bounce for a multi-factor quality score (speed + depth + placement).
    shot_by_frame = {}      # frame -> dict(side, fhb, quality, speed)
    if not args.no_bounces and all_ball_centers:
        from vision.shots import detect_hits, classify_forehand_backhand, shot_quality
        bset = list(bounce_by_frame.keys())   # [] when --event-methodo (deferred)
        # LEGACY SEPARATION: feed hit detection the LOCKED, gap-filled P1/P2 tracks
        # (reshaped per-frame to [near, far]) rather than the raw per-frame poses.
        # Same reason the methodo path uses methodo_ppf: the ball↔racket-wrist
        # proximity gate (which prunes phantom wrist-speed swings) needs the FAR
        # player's pose to exist, and the tracker's gap-fill is what bridges the
        # tiny far player's missed frames (now that far-select reaches ~85%). Built
        # ONCE here and reused by both paths below.
        locked_ppf = []
        for _i in range(max_frames):
            _near = player_tracks[2][_i] if _i < len(player_tracks[2]) else None
            _far = player_tracks[1][_i] if _i < len(player_tracks[1]) else None
            _slot = []
            if _near is not None:
                _slot.append(_near)
            if _far is not None:
                _slot.append(_far)
            locked_ppf.append(_slot)
        # LEGACY confusion_H→B guard (runs BEFORE hit detection, now that poses
        # exist): drop any turn-angle-recovered bounce (no y-extremum → lower
        # confidence) whose landing falls INSIDE a player's box. A real FLOOR bounce
        # lands on the COURT, never inside a player's silhouette; a turn candidate
        # that landed inside a player is a near-court CONTACT mislabeled as a bounce.
        # Doing this BEFORE detect_hits is the point: the bogus bounce no longer
        # suppresses the real hit at that frame (bounce_guard), so the contact is
        # emitted as a HIT and confusion_H→B stays 0. Pure bounce-side filter — only
        # ever removes a turn (low-conf) bounce, never a y-extremum or a real
        # bounce-near-a-player (those land OUTSIDE the box). Verified NO-OP on the
        # demo3 cache (no turn-bounce there, incl. real b308, lands inside a box); it
        # fires on the LIVE track where the apex recovery put a bounce on a contact.
        if not event_methodo and turn_bounce_frames and bounce_events:
            from vision.bounce import drop_turn_bounces_inside_players
            _before = len(bounce_events)
            bounce_events, _dropped = drop_turn_bounces_inside_players(
                bounce_events, turn_bounce_frames, locked_ppf)
            if len(bounce_events) != _before:
                bounce_by_frame = _build_bounce_by_frame(bounce_events)
                direction_state_by_frame = {
                    f: s for f, s in direction_state_by_frame.items()
                    if f in {int(b[0]) for b in bounce_events}}
                bset = list(bounce_by_frame.keys())
        # The wrist-proximity gate is the LEGACY lever (it prunes the over-fired
        # phantom swings: demo3 hit FP 7→2). It is OFF in the --event-methodo path
        # because the classifier's score-free firewall owns hit arbitration there —
        # gating upstream would starve it (proven: it re-introduces confusion_B→H on
        # the methodo regression). So only the legacy path opts in, and the methodo
        # path keeps using the RAW per-frame poses + an un-gated candidate set
        # (unchanged from before this branch).
        if event_methodo:
            # HIT candidates from the LOCKED, gap-filled P1/P2 tracks — the same
            # source classify_events uses for proximity (see methodo_ppf below).
            # The RAW per-frame poses were kept here for compatibility with the
            # pre-farselect world, but the far player's raw wrists are too sparse/
            # noisy to time far-court swings: on the demo3 GT the raw-pose swing
            # for the far hit @394 lands at dmin 0.71 (discredited) while the
            # locked-track swing lands @396 at contact-grade dmin (< near_wrist).
            # Locked tracks are denser AND better localized (far-select 0.9%→100%),
            # so the candidate generator and the arbitration finally read the same
            # players.
            methodo_ppf = []
            for i in range(max_frames):
                near = player_tracks[2][i] if i < len(player_tracks[2]) else None
                far = player_tracks[1][i] if i < len(player_tracks[1]) else None
                slot = []
                if near is not None:
                    slot.append(near)
                if far is not None:
                    slot.append(far)
                methodo_ppf.append(slot)
            # far_aware: the frame-fraction serve gate (wy < 0.30*fh) is
            # structurally TRUE for every far-court contact — with it, no far
            # swing can ever produce a candidate and the far GT hits @141/@525
            # are undetectable. far_aware makes the serve gate PLAYER-relative
            # and the NMS per-side (a near peak no longer suppresses a far
            # contact 0.3s later). Its historical over-fire is now owned by the
            # events evidence rules (contact-corroborated swings + firewall):
            # measured on the live canonical clip, far_aware=True + the
            # side-scaled amplitude floor recovers ALL 8 GT hits + all 9 GT
            # bounces (F1 0.941/0.947, confusion 0/0) and even the un-annotated
            # SERVE (@44) — the two extra markers at clip start are the far
            # player's real pre-serve ritual (ball-dribble + serve contact),
            # verified frame-by-frame.
            hits = detect_hits(all_ball_centers, methodo_ppf, fps,
                               frame_height, frame_width, bounce_frames=bset,
                               far_aware=True)
        else:
            hits = detect_hits(all_ball_centers, locked_ppf, fps,
                               frame_height, frame_width, bounce_frames=bset,
                               wrist_prox_max=1.2)

        if event_methodo and not args.demo_override:
            # ── UNIFIED ARBITRATION (vision.events.classify_events) ──
            # NOW both candidate sets + poses exist, so we can arbitrate. We
            # reproduce the proven demo3 recipe EXACTLY (tools/event_eval/
            # run_demo3.py:methodo_events): robust bounce candidates + ALL
            # trajectory turning points + far-court apex sharp-turns as the recall
            # pool, the wrist-speed hits as the H votes, then classify_events
            # labels each event under the score-free firewall (confusion_H->B and
            # B->H == 0 by a logical gate) and the rally-alternation DP.
            #
            # Direction is read on the RAW pre-spline track + is_real mask (the
            # safety invariant); smoothed_centers is the clean spline track (NOT
            # the robust detector's re-smooth — we keep all_ball_centers intact in
            # this path). bounce candidates come from detect_bounces_robust on the
            # spline track (tracker-independent, matching the cache recipe).
            from vision.bounce import detect_bounces_robust
            from vision.events import (
                classify_events, detect_turning_points, detect_sharp_turns,
                detect_near_court_knees)
            b_cands, _re, _nd = detect_bounces_robust(
                all_ball_centers, ball_speeds_px, fps, frame_height, frame_width)
            turns = set(detect_turning_points(all_ball_centers, fps, frame_height))
            turns |= set(detect_sharp_turns(
                raw_ball_centers, ball_is_real, fps, frame_width, frame_height))
            # near-court grazing-bounce knees (S7): vy descends then settles without
            # flipping — no y-extremum and no vy-flip, so the detectors above miss it
            # (e.g. b174/b308). Gated to y>net_y; feeds turning_frames, the firewall
            # still arbitrates BOUNCE vs HIT. See docs/research/s7-near-court-bounce-CR.md.
            turns |= set(detect_near_court_knees(
                raw_ball_centers, ball_is_real, fps, frame_width, frame_height))
            # PROXIMITY SOURCE = the LOCKED, gap-filled P1/P2 tracks (player_tracks),
            # reshaped per-frame to [near, far] — NOT the raw per-frame poses. This
            # matches the benchmark cache EXACTLY (scripts/build_demo3_event_cache
            # .py serializes ptracker.finalize() tracks as players_per_frame): the
            # firewall's "ball-at-a-wrist" hit_like test needs the FAR player's
            # pose to exist, and gap-fill is what bridges the tiny far player's
            # missed frames (raw far coverage is far lower). Feeding raw poses here
            # starves the firewall and lets far-court hits leak to BOUNCE.
            # (methodo_ppf itself is built above, where detect_hits consumes it.)
            # DIAGNOSTIC (inert unless COURTSIDE_DUMP_METHODO_INPUTS is set): dump
            # the EXACT live classify_events inputs so the prod-vs-cache event
            # divergence can be replayed offline candidate-by-candidate. No behavior
            # change — pure side-effect serialization of what is fed below.
            import os as _os_diag
            _dump_path = _os_diag.environ.get("COURTSIDE_DUMP_METHODO_INPUTS")
            if _dump_path:
                def _ser_ppf(ppf):
                    out = []
                    for slot in ppf:
                        s = []
                        for p in slot:
                            if p is None:
                                continue
                            box = p.get("box")
                            kx = p.get("kps_xy")
                            kc = p.get("kps_conf")
                            s.append({
                                "box": (np.asarray(box).tolist() if box is not None else None),
                                "kps_xy": (np.asarray(kx).tolist() if kx is not None else None),
                                "kps_conf": (np.asarray(kc).tolist() if kc is not None else None),
                            })
                        out.append(s)
                    return out
                _dump = {
                    "fps": fps, "width": frame_width, "height": frame_height,
                    "start_frame": int(start_frame),
                    "raw_ball_centers": [list(c) if c is not None else None
                                         for c in raw_ball_centers],
                    "ball_is_real": [bool(x) for x in ball_is_real],
                    "all_ball_centers": [list(c) if c is not None else None
                                         for c in all_ball_centers],
                    "b_cands": [[int(f), int(x), int(y)] for (f, x, y) in b_cands],
                    "turns": sorted(int(t) for t in turns),
                    "hits": [{"frame": int(h["frame"]), "x": int(h["x"]),
                              "y": int(h["y"]), "player_side": h.get("player_side")}
                             for h in hits],
                    "methodo_ppf": _ser_ppf(methodo_ppf),
                    # replay completeness: detect_hits' exact inputs (the legacy
                    # bounce set feeding its bounce_guard + the RAW per-frame
                    # poses it consumes) so offline replays can recompute the
                    # hit candidates prod-faithfully after a shots.py change.
                    "hits_bset": sorted(int(b) for b in (bset or [])),
                    "raw_ppf": _ser_ppf(players_per_frame),
                }
                Path(_dump_path).parent.mkdir(parents=True, exist_ok=True)
                with open(_dump_path, "w") as _mf:
                    json.dump(_dump, _mf)
                logger.info(f"[diag] dumped live methodo inputs -> {_dump_path}")

            events, evt_report = classify_events(
                b_cands, hits, raw_ball_centers, ball_is_real, methodo_ppf,
                fps, frame_width, frame_height,
                turning_frames=sorted(turns), smoothed_centers=all_ball_centers,
                wasb_centers=None, wasb_is_real=None)

            # split labeled events; rebuild bounce structures FIRST (shots link to
            # the NEXT bounce), then map HITs back onto the shot_by_frame contract.
            bounce_events = [(e["frame"], e["x"], e["y"])
                             for e in events if e["label"] == "BOUNCE"]
            bounce_by_frame = _build_bounce_by_frame(bounce_events)
            # --dump-bounces: the Pass-1 dump ran before arbitration (empty in this
            # path), so emit the ARBITRATED bounces here instead (same schema).
            if args.dump_bounces:
                pred_doc = {
                    "video": video_path, "fps": fps, "frame_offset": start_frame,
                    "annotator": "run_pipeline_8s.py:classify_events",
                    "notes": "auto-exported bounce predictions (--event-methodo)",
                    "bounces": [{"frame": int(start_frame + idx), "x": int(x), "y": int(y)}
                                for (idx, x, y) in bounce_events]}
                Path(args.dump_bounces).parent.mkdir(parents=True, exist_ok=True)
                with open(args.dump_bounces, "w") as _df:
                    json.dump(pred_doc, _df, indent=2)
                    _df.write("\n")
                logger.info(f"Dumped {len(bounce_events)} bounce predictions "
                            f"(methodo) -> {args.dump_bounces}")
            # direction_state per bounce frame from the classifier's own vx read:
            # vx preserved (flip False) on a BOUNCE == the "metric" co-signal the
            # legacy veto reported; None/unreadable stays "unconfirmed".
            direction_state_by_frame = {}
            for e in events:
                if e["label"] == "BOUNCE":
                    vxf = e["features"].get("vx_flip")
                    direction_state_by_frame[e["frame"]] = (
                        "metric" if vxf is False else "unconfirmed")

            # HIT events → shot_by_frame. Match each HIT back to the originating
            # detect_hits record (same frame) to recover player_side + FH/BH +
            # human_id; a parity-fill HIT (no candidate) falls back to a geometric
            # side from the ball's vertical position (the classifier already
            # firewalled it as a real hit). Quality links to the next bounce,
            # mirroring the legacy path EXACTLY (same shot_quality call).
            hit_by_frame = {h["frame"]: h for h in hits}
            net_y = frame_height * 0.47
            sorted_bounces = sorted(bounce_by_frame.keys())
            # FH/BH composite for the methodo path. The arbitration above used the RAW
            # per-frame poses for its HIT votes (unchanged — the firewall owns that),
            # but the raw FAR pose is sparse and mis-attributes far contacts to the
            # near side with a mid-flight ball position → the composite abstained on
            # far shots (the cache-vs-LIVE gap: 8/8 cache, 1/5 live). FIX: source the
            # FH/BH contact from a SECOND detect_hits run on the DENSE locked tracks
            # (well-attributed near+far now that the serve gate is player-relative),
            # and match each arbitrated HIT event to the nearest locked contact. The
            # contact ball_dx + swing window + handedness vote all read locked_ppf —
            # the same dense poses the cache is built from, so the cache 8/8
            # reproduces live. This does NOT touch the arbitration (events) — it only
            # picks which posed contact the FH/BH label is read from.
            from vision.shots import (
                detect_hits as _detect_hits_fhb, estimate_handedness_prior,
                classify_forehand_backhand)
            locked_contacts = _detect_hits_fhb(
                all_ball_centers, locked_ppf, fps, frame_height, frame_width,
                bounce_frames=bset, wrist_prox_max=1.2, far_aware=True)
            fhb_handed = estimate_handedness_prior(
                locked_contacts, locked_ppf, fps, frame_height)
            fhb_win = int(round(fps * 0.20))  # match a HIT event to a locked contact

            def _nearest_locked_contact(ef):
                best, bestd = None, fhb_win + 1
                for lc in locked_contacts:
                    d = abs(lc["frame"] - ef)
                    if d < bestd:
                        bestd, best = d, lc
                return best

            for e in events:
                if e["label"] != "HIT":
                    continue
                ef = e["frame"]
                hm = hit_by_frame.get(ef)
                lc = _nearest_locked_contact(ef)
                if lc is not None:
                    # well-attributed locked contact → classify FH/BH on it
                    side = lc["player_side"]
                    fhb = classify_forehand_backhand(
                        lc, locked_ppf, players_per_frame_window=locked_ppf,
                        handedness=fhb_handed, fps=fps, frame_height=frame_height)
                    sx, sy = (hm["x"], hm["y"]) if hm is not None else (lc["x"], lc["y"])
                elif hm is not None:
                    # raw-pose contact only (no locked match) → classify on raw poses
                    side = hm["player_side"]
                    fhb = classify_forehand_backhand(
                        hm, players_per_frame, players_per_frame_window=locked_ppf,
                        handedness=fhb_handed, fps=fps, frame_height=frame_height)
                    sx, sy = hm["x"], hm["y"]
                else:
                    # generic/parity HIT: derive side from ball y vs net; FH/BH
                    # unknown without a posed contact player.
                    side = "near" if e["y"] >= net_y else "far"
                    fhb = "unknown"
                    sx, sy = e["x"], e["y"]
                nb = next((b for b in sorted_bounces if b > ef), None)
                if nb is not None:
                    bx, by, depth, b_speed, _ = bounce_by_frame[nb]
                    _h = h_at(court_homography_by_frame, nb, court_homography)
                    q = shot_quality(b_speed, depth, (bx, by), court_zones,
                                     frame_width, frame_height,
                                     homography=_h, speed_ref=speed_ref)
                else:
                    b_speed, q = 0.0, 0
                shot_by_frame[ef] = {
                    "side": side, "fhb": fhb, "quality": q,
                    "speed": b_speed, "x": sx, "y": sy,
                    "human_id": (human_id_for_side(match_result, side, ef)
                                 if match_mode else None)}
            logger.info(
                f"Event methodo: {len(bounce_by_frame)} bounce / {len(shot_by_frame)} "
                f"hit events arbitrated (WASB consistency {evt_report['consistency']:.0%}); "
                f"firewall guarantees confusion_H->B / B->H == 0")
        else:
            sorted_bounces = sorted(bounce_by_frame.keys())
            # FH/BH composite: vote the per-side handedness prior ONCE over ALL hits
            # (more shots → better vote margin), then classify each with the swing
            # override + abstention. Reads the SAME locked poses detect_hits indexed
            # (h["player_idx"] indexes locked_ppf).
            from vision.shots import classify_shots
            fhb_labels, fhb_handed = classify_shots(
                hits, locked_ppf, fps, frame_height,
                players_per_frame_window=locked_ppf)
            for h, fhb in zip(hits, fhb_labels):
                # link to the next bounce after this hit for depth/placement/speed
                nb = next((b for b in sorted_bounces if b > h["frame"]), None)
                if nb is not None:
                    bx, by, depth, b_speed, _ = bounce_by_frame[nb]
                    _h = h_at(court_homography_by_frame, nb, court_homography)
                    q = shot_quality(b_speed, depth, (bx, by), court_zones,
                                     frame_width, frame_height,
                                     homography=_h, speed_ref=speed_ref)
                else:
                    b_speed, q = 0.0, 0
                shot_by_frame[h["frame"]] = {
                    "side": h["player_side"], "fhb": fhb, "quality": q,
                    "speed": b_speed, "x": h["x"], "y": h["y"],
                    "human_id": (human_id_for_side(match_result, h["player_side"], h["frame"])
                                 if match_mode else None)}
        logger.info(f"Shots: {len(shot_by_frame)} detected "
                    f"({sum(1 for s in shot_by_frame.values() if s['fhb']=='forehand')} FH, "
                    f"{sum(1 for s in shot_by_frame.values() if s['fhb']=='backhand')} BH)")

    # ─── Demo override: inject manual racket->ball strikes ───
    # Overrides the geometric shot detector with hand-annotated contacts. Each
    # strike replaces any auto-detected hit at the same frame (no duplicates) and
    # is linked to the next bounce for a quality score, mirroring the auto path.
    if override_strikes:
        sorted_bounces = sorted(bounce_by_frame.keys())
        n_replaced = 0
        for sfi, s in sorted(override_strikes.items()):
            if sfi in shot_by_frame:
                n_replaced += 1
            nb = next((b for b in sorted_bounces if b > sfi), None)
            if nb is not None:
                bx, by, depth, b_speed, _ = bounce_by_frame[nb]
                try:
                    from vision.shots import shot_quality
                    _h = h_at(court_homography_by_frame, nb, court_homography)
                    q = shot_quality(b_speed, depth, (bx, by), court_zones,
                                     frame_width, frame_height,
                                     homography=_h, speed_ref=speed_ref)
                except Exception:
                    q = 0
            else:
                b_speed, q = 0.0, 0
            shot_by_frame[sfi] = {
                "side": None, "fhb": s["type"], "quality": q,
                "speed": b_speed, "x": s["xy"][0], "y": s["xy"][1]}
        logger.info(f"Demo override strikes: injected {len(override_strikes)} "
                    f"({n_replaced} replaced auto hits).")

    # ─── Pass 2: Annotate ───
    #
    # OVERLAY OVERHAUL (S1 — feat/video-overlay). Three coupled fixes in the
    # annotation loop, each with a defensive fallback so this branch compiles and
    # runs standalone AND opportunistically uses the other sessions' work:
    #
    #   #1  shot ≠ bounce — orthogonal visual languages (shape+motion+label-
    #       quadrant; color stays the depth/stroke attribute). Bounce = flat
    #       ground-mark + outward shockwave; shot = upright diamond + caret that
    #       pops inward. Labels sit in opposite quadrants (bounce below-left /
    #       shot above-right), measured via fx.text_size so they never overlap. A
    #       persistent top-right legend (draw_event_legend) teaches shape=event.
    #
    #   #3  HUD card is PER-RALLY (resets each échange) with an "ÉCHANGE N" header
    #       + colored "Issue" outcome row (shown only once the point is over).
    #
    #   #4  the radar receives ONLY the latest bounce, not every bounce so far.
    #
    # CONTRACTS expected from sibling sessions (used defensively — getattr/try /
    # signature-introspection, with a working fallback if absent):
    #
    #   S3 (vision/minimap.py):
    #     draw_minimap(..., bounces_img=<list of 0 or 1 bounce>, single_bounce=True)
    #       → draws AT MOST one clean bounce marker, NO cumulative placement heat.
    #       Passed only when draw_minimap's signature accepts `single_bounce`
    #       (introspected once). Without it, the legacy call still gets a 1-element
    #       list, which degenerates to a single dot (heat needs ≥2 points) — the
    #       acceptable transitional behavior.
    #
    #   S4 (vision/fx.py):
    #     fx.rally_card(frame, x, y, *, rally_index:int, lines:list[(label,value)],
    #                   outcome:str|None=None, width:int) → glassmorphism card with
    #       an "ÉCHANGE N" header + a colored outcome row. Used when present (the
    #       `lines` we pass EXCLUDE the Issue row, since rally_card renders outcome
    #       itself). Fallback: fx.stat_card(title="ÉCHANGE N", ...) + a semantic-
    #       colored draw_text overdraw of the Issue value (stat_card draws values
    #       white). NOTE: the gauge-stack height is computed for the stat_card
    #       fallback layout; if S4's rally_card differs in row count, re-tune
    #       card_h at integration.
    #
    logger.info("=== PASS 2: Annotate ===")
    reader = VideoReader(video_path)
    if start_frame > 0:
        reader.seek(start_frame)
    writer = VideoWriter(output_path, fps=fps)
    ball_trail = deque(maxlen=trail_length)

    # Bounce display duration: the ground-mark + shockwave lingers this long after
    # the bounce. 0.5s was too long — at 50fps the SHORT/DEEP ring stayed planted
    # mid-court for 25 frames while the ball had already travelled on, reading as a
    # phantom second bounce (user feedback). 0.3s (~15f @50fps) is still clearly
    # visible at contact (Hawk-Eye markers are brief) but clears before the ball
    # moves far. Display-only — touches no event/stat.
    bounce_display_frames = int(fps * 0.3)

    # Build legend items
    legend_items = {"Ball": BALL_COLOR}
    if not args.no_player_boxes:
        legend_items["Player 1"] = PLAYER_COLORS[1]
        legend_items["Player 2"] = PLAYER_COLORS[2]
    if not args.no_bounces:
        legend_items["Bounce (deep)"] = (0, 0, 255)
        legend_items["Bounce (short)"] = (0, 230, 255)
        legend_items["Quality 70+"] = (0, 200, 0)
        legend_items["Quality <40"] = (0, 0, 220)

    # ── Per-rally HUD scaffolding (#3) ──────────────────────────────────────
    # The live stat card must reset every NEW rally (échange) and tag the current
    # rally number, instead of cumulative counters since the clip start. Segment
    # the rallies NOW (in relative frame coords, mirroring the post-loop absolute
    # segmentation + solo-merge) so we know, at every frame_idx, which rally is in
    # progress and can recompute that rally's own FH/BH/shot/bounce counts + peak.
    sorted_shot_frames = sorted(shot_by_frame.keys())
    sorted_bounce_frames = sorted(bounce_by_frame.keys())
    rallies_rel = segment_rallies_relative(shot_by_frame, bounce_by_frame, fps,
                                           court_mask=court_visible_mask if shot_guard_enabled else None)
    rally_at_frame = build_rally_at_frame(rallies_rel, max_frames)

    # Outcome per rally, computed ONCE up front (the geometric classifier expects
    # ABSOLUTE frames, so feed it absolute-frame rally dicts + shot/bounce lists,
    # exactly as the post-loop JSON path does). Only shown live once the rally is
    # OVER (we don't reveal a winner/error mid-point — premature + flickery).
    rally_outcome_rel = [None] * len(rallies_rel)
    if rallies_rel:
        abs_shots = [{"frame": int(start_frame + f), "x": s["x"], "y": s["y"],
                      "player_side": s["side"], "stroke": s["fhb"],
                      "quality": s["quality"], "speed_kmh": round(s["speed"], 1)}
                     for f, s in sorted(shot_by_frame.items())]
        abs_bounces = [{"frame": int(start_frame + f), "x": int(v[0]), "y": int(v[1]),
                        "depth": v[2], "speed_kmh": round(v[3], 1), "quality": v[4]}
                       for f, v in sorted(bounce_by_frame.items())]
        for ri, r in enumerate(rallies_rel):
            abs_rally = {
                "start_frame": int(start_frame + r["start_frame"]),
                "end_frame": int(start_frame + r["end_frame"]),
                "shot_frames": [int(start_frame + sf) for sf in r["shot_frames"]],
                "bounce_frames": [int(start_frame + bf) for bf in r["bounce_frames"]],
                "n_shots": r["n_shots"],
            }
            try:
                rally_outcome_rel[ri] = classify_rally_outcome(
                    abs_rally, abs_shots, abs_bounces, fps)
            except Exception:
                rally_outcome_rel[ri] = None

    peak_speed = 0.0          # current-rally peak (resets on rally change)
    hud_rally_idx = None      # which rally the card is currently showing

    # DISPLAY speed series — rolling MEDIAN of the per-frame km/h (window
    # ~0.10 s, the same time constant as the direction half-window). The raw
    # series carries 1-2-frame spikes at spline junctions / re-detections; the
    # summary and bounce speeds average them out, but the HUD gauge and the
    # per-rally "Balle max" read the series directly, so one spiked frame
    # printed absurd peaks (224 km/h on a ~100 km/h rally). A short median
    # kills isolated spikes and preserves real plateaus; analytics/stats keep
    # using the raw series (this is presentation-only).
    _disp_win = max(3, int(round(fps * 0.10)) | 1)
    _half_w = _disp_win // 2
    display_speeds_kmh = [
        float(np.median([s for s in ball_speeds_kmh[max(0, i - _half_w):i + _half_w + 1]]))
        if len(ball_speeds_kmh) else 0.0
        for i, s in enumerate(ball_speeds_kmh)
    ]

    # Event-overlay geometry (constant per clip — resolution-derived, no hardcoding).
    # U: base shape unit. GAP: label clearance + inter-row pitch around an impact.
    EV_U = max(6, int(frame_height * 0.018))
    EV_GAP = max(6, int(frame_height * 0.012))

    # Does the (possibly newer, S3) draw_minimap accept the single_bounce flag?
    # Introspect ONCE so the per-frame call doesn't pay for it, and so this branch
    # runs cleanly against both the legacy and the S3 signatures.
    import inspect as _inspect
    try:
        _minimap_accepts_single_bounce = (
            "single_bounce" in _inspect.signature(draw_minimap).parameters
            or any(p.kind == _inspect.Parameter.VAR_KEYWORD
                   for p in _inspect.signature(draw_minimap).parameters.values()))
    except (ValueError, TypeError):
        _minimap_accepts_single_bounce = False

    frame_idx = 0
    for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 2"):
        if frame_idx >= max_frames:
            break
        # shot-guard CUT: a non-court frame (close-up / replay / graphic) is dropped
        # from the annotated output entirely — no overlay, no write, no CPU. We still
        # advance frame_idx so every per-frame lookup (bounce/rally/homography, all
        # keyed on the ORIGINAL window-relative index) stays correct; only the
        # WRITTEN video compresses, and the clip-index sidecar records the mapping.
        if shot_guard_enabled and frame_idx not in keep_set:
            frame_idx += 1
            continue
        out = frame.copy()

        # 1. Court zone overlay (background layer) — redrive from the per-frame
        # homography when tracking a moving camera, so the zones follow the court.
        if not args.no_court and court_zones:
            cz = court_zones
            if court_homography_by_frame is not None:
                _h = h_at(court_homography_by_frame, frame_idx, court_homography)
                if _h is not None and _h.get("H_inv") is not None:
                    from vision.court_zones import derive_court_zones_from_homography
                    cz = derive_court_zones_from_homography(_h) or court_zones
            out = draw_court_overlay(out, cz, alpha=0.15,
                                     draw_labels=args.court_labels)

        # 2. Pose — LOCKED P1/P2 skeleton tracks from Pass 1.5 (persistent identity,
        # gap-filled so every frame has both skeletons, distinct colors per player).
        # No per-frame pose here: we draw the pre-tracked, smoothed poses.
        person_boxes = []
        for pid in (1, 2):
            pose = player_tracks[pid][frame_idx] if frame_idx < len(player_tracks[pid]) else None
            if pose is None or pose.get("kps_xy") is None:
                continue
            out = draw_skeleton(out, pose["kps_xy"], pose["kps_conf"], PLAYER_COLORS[pid])
            person_boxes.append((pid, pose["box"]))

        # 3. Player bounding boxes (off by default; same locked P1/P2 identity)
        if not args.no_player_boxes:
            for pid, box in person_boxes:
                out = draw_player_bbox(out, box, pid)

        # 4. Ball — cinematic comet trail + glowing pulsing marker (FX), or the
        # plain trail/marker when --no-fx.
        bc = all_ball_centers[frame_idx]
        if bc is not None:
            ball_trail.append(bc)
        if not args.no_fx:
            if len(ball_trail) >= 2:
                out = fx.comet_trail(out, list(ball_trail), base_color=BALL_COLOR,
                                     max_thickness=7, glow=True)
            if bc is not None:
                # pulse phase advances with frame_idx so the ball "breathes"
                out = fx.glow_marker(out, (int(bc[0]), int(bc[1])), color=BALL_COLOR,
                                     radius=8, pulse=frame_idx * 0.5)
        else:
            if bc is not None:
                out = draw_ball_marker(out, bc[0], bc[1])
            draw_ball_trail(out, list(ball_trail))

        # 4b. Current ball speed (HUD drawn later, after overlays, as a gauge/card)
        current_speed = (display_speeds_kmh[frame_idx]
                         if frame_idx < len(display_speeds_kmh) else 0.0)

        # 5. BOUNCE markers — "ground-mark + shockwave". The shockwave ring blooms
        # OUTWARD (energy) while a FLAT, planted squashed-ellipse + diamond stake
        # reads as lying ON THE COURT. This flat-on-the-floor silhouette is what
        # makes a bounce unmistakable from a shot (which is an UPRIGHT diamond +
        # caret, in the air). Color still carries DEPTH (deep red / mid cyan /
        # short green). Label goes BELOW-LEFT, right-anchored. (#1)
        DEPTH_COL = {"deep": (60, 60, 255), "mid": (0, 200, 255), "short": (120, 230, 0)}
        U, GAP, H = EV_U, EV_GAP, frame_height
        if not args.no_bounces:
            for bfi in range(max(0, frame_idx - bounce_display_frames), frame_idx + 1):
                if bfi in bounce_by_frame:
                    bx, by, depth, b_speed, b_quality = bounce_by_frame[bfi]
                    bx, by = int(bx), int(by)
                    age = frame_idx - bfi
                    p = age / float(bounce_display_frames)
                    fade = 1.0 - p
                    dcol = DEPTH_COL.get(depth, (255, 255, 255))
                    mw = int(2.4 * U)
                    mh = max(2, int(mw * 0.40))
                    if not args.no_fx:
                        # A) expanding bloomed depth-ring — energy (existing primitive)
                        out = fx.shockwave(out, (bx, by), age, bounce_display_frames,
                                           color=dcol, max_radius=int(H * 0.075))
                        # B) FLAT planted ground-mark — never expands (Hawk-Eye signature)
                        ov = out.copy()
                        cv2.ellipse(ov, (bx, by), (mw, mh), 0, 0, 360, dcol, -1, cv2.LINE_AA)
                        a = 0.35 * fade + 0.15
                        cv2.addWeighted(ov, a, out, 1 - a, 0, dst=out)
                        cv2.ellipse(out, (bx, by), (mw, mh), 0, 0, 360,
                                    tuple(int(min(255, c * 1.25)) for c in dcol),
                                    2, cv2.LINE_AA)
                        cv2.drawMarker(out, (bx, by), dcol, cv2.MARKER_DIAMOND,
                                       max(7, int(0.6 * U)), 2, cv2.LINE_AA)
                    else:
                        cv2.ellipse(out, (bx, by), (mw, mh), 0, 0, 360, dcol, 2, cv2.LINE_AA)
                        cv2.drawMarker(out, (bx, by), dcol, cv2.MARKER_DIAMOND,
                                       max(7, int(0.6 * U)), 2, cv2.LINE_AA)
                    # styled label — BELOW-LEFT, right-anchored, grows DOWN (only
                    # while fresh). Opposite quadrant from the shot label, so a
                    # coincident shot+bounce never overlap their text.
                    if age <= bounce_display_frames // 2:
                        rows_b = [(depth.upper(), 18, dcol, "bold"),
                                  (f"{int(b_speed)} km/h", 15, (210, 210, 210), "regular"),
                                  (f"Q {b_quality}", 15, _quality_color(b_quality), "bold")]
                        hs = [fx.text_size(t, sz, f)[1] for (t, sz, _c, f) in rows_b]
                        total_b = sum(hs) + GAP * (len(rows_b) - 1)
                        flip_b = by > H * 0.85          # baseline-bottom → flip ABOVE
                        ly = (by - GAP - total_b) if flip_b else (by + GAP)
                        lx = bx - GAP
                        for (t, sz, c, f), hh in zip(rows_b, hs):
                            out = fx.draw_text(out, t, (lx, ly), size=sz, color=c,
                                               font=f, anchor="rt")
                            ly += hh + GAP

        # 5b. SHOT markers — "contact diamond + downward caret". An UPRIGHT diamond
        # outline + a downward chevron above the ball reads as a strike IN THE AIR;
        # it pops large at contact then settles (inward) — the opposite motion to
        # the bounce's outward bloom. Color carries STROKE (FH cyan / BH orange).
        # Label goes ABOVE-RIGHT, left-anchored (opposite quadrant from bounce). (#1)
        if shot_by_frame:
            for sfi in range(max(0, frame_idx - bounce_display_frames), frame_idx + 1):
                if sfi in shot_by_frame:
                    s = shot_by_frame[sfi]
                    age = frame_idx - sfi
                    if age > bounce_display_frames:
                        continue
                    sx, sy = int(s["x"]), int(s["y"])
                    fhb = s["fhb"]
                    tag = {"forehand": "FOREHAND", "backhand": "BACKHAND"}.get(fhb, "SHOT")
                    fcol = (0, 200, 255) if fhb == "forehand" else (255, 150, 0)
                    p = age / float(bounce_display_frames)
                    fade = 1.0 - p
                    # pop large at contact (first 25% of life), then shrink to rest
                    pop = 1.0 + 0.45 * (1 - min(1.0, age / max(1, bounce_display_frames * 0.25)))
                    m = int(max(1.4 * U, H * 0.026) * pop)
                    th = max(2, int(3 * fade + 1))
                    if not args.no_fx:
                        # tiny white-hot contact spark (NOT a big dot — that's what
                        # used to be confusable with a bounce)
                        out = fx.glow_marker(out, (sx, sy), color=fcol, radius=max(3, m // 5))
                        cv2.drawMarker(out, (sx, sy), fcol, cv2.MARKER_DIAMOND,
                                       m * 2, th, cv2.LINE_AA)
                        apex = (sx, sy - int(m * 1.4))
                        cv2.line(out, (apex[0] - m, apex[1] - m), apex, fcol, th, cv2.LINE_AA)
                        cv2.line(out, (apex[0] + m, apex[1] - m), apex, fcol, th, cv2.LINE_AA)
                    else:
                        cv2.drawMarker(out, (sx, sy), fcol, cv2.MARKER_DIAMOND,
                                       m * 2, 2, cv2.LINE_AA)
                        apex = (sx, sy - int(m * 1.4))
                        cv2.line(out, (apex[0] - m, apex[1] - m), apex, fcol, 2, cv2.LINE_AA)
                        cv2.line(out, (apex[0] + m, apex[1] - m), apex, fcol, 2, cv2.LINE_AA)
                    # label — ABOVE-RIGHT, left-anchored, grows UP (only while fresh)
                    if age <= bounce_display_frames // 2:
                        rows_s = [(tag, 17, fcol, "bold")]
                        if s["quality"] > 0:
                            rows_s.append((f"Q {s['quality']}", 15,
                                           _quality_color(s["quality"]), "bold"))
                        hs = [fx.text_size(t, sz, f)[1] for (t, sz, _c, f) in rows_s]
                        total_s = sum(hs) + GAP * (len(rows_s) - 1)
                        flip_s = sy < H * 0.10          # top-of-frame → flip BELOW
                        ly = (sy + GAP) if flip_s else (sy - GAP - total_s)
                        rx = sx + GAP
                        for (t, sz, c, f), hh in zip(rows_s, hs):
                            out = fx.draw_text(out, t, (rx, ly), size=sz, color=c,
                                               font=f, anchor="lt")
                            ly += hh + GAP

        # 5c. 2D top-down court minimap (ball trajectory + bounce locations)
        if not args.no_minimap:
            # a longer dedicated trajectory window for the map (the on-court trail
            # is intentionally short). Show ~2s of recent ball positions.
            mm_window = int(fps * 2.0)
            lo = max(0, frame_idx - mm_window)
            mm_trail = [all_ball_centers[j] for j in range(lo, frame_idx + 1)
                        if all_ball_centers[j] is not None]
            # #4 — the radar shows ONLY the most-recent bounce, not every bounce
            # since the clip start. Each new bounce replaces the previous marker
            # (no cumulative heat). Pick the single latest bounce frame <= now.
            _latest_bf = max((bf for bf in bounce_by_frame if bf <= frame_idx),
                             default=None)
            latest_bounce_img = ([(bounce_by_frame[_latest_bf][0],
                                    bounce_by_frame[_latest_bf][1],
                                    bounce_by_frame[_latest_bf][2])]
                                 if _latest_bf is not None else [])
            # GROUND-PATH for the radar arc: only metrically-trustworthy ground
            # contacts — bounces (true ground impacts) + hit/contact points (≈ground
            # at the player) — in temporal (frame) order, growing over the clip. The
            # in-flight ball is no longer projected naively onto the top-down map.
            gp_events = []
            for bf, v in bounce_by_frame.items():
                if bf <= frame_idx:
                    gp_events.append((bf, v[0], v[1], "bounce"))
            for sf, s in shot_by_frame.items():
                if sf <= frame_idx:
                    gp_events.append((sf, s["x"], s["y"], "hit"))
            gp_events.sort(key=lambda e: e[0])
            ground_path = [(e[1], e[2], e[3]) for e in gp_events]
            # live player feet (bottom-center of each locked P1/P2 box) → radar dots
            players_img = [((b[0] + b[2]) / 2.0, b[3], pid) for pid, b in person_boxes]
            # S3 contract: draw_minimap(..., single_bounce=True) draws at most one
            # clean bounce marker with NO cumulative heatmap. Pass it only when the
            # current draw_minimap accepts it (introspected once), so this branch
            # also runs against the legacy signature — there, the 1-element bounce
            # list already degenerates to a single dot (heat needs ≥2 points).
            mm_kwargs = dict(
                homography=h_at(court_homography_by_frame, frame_idx, court_homography),
                players_img=players_img,
                pulse=frame_idx * 0.5,
                ground_path=ground_path,
                now_frame=frame_idx,
            )
            if _minimap_accepts_single_bounce:
                mm_kwargs["single_bounce"] = True
            out = draw_minimap(out, mm_trail, latest_bounce_img,
                               frame_width, frame_height, **mm_kwargs)

        # 6. HUD — speed gauge + live match-stats card, anchored BOTTOM-LEFT so it
        # never collides with a broadcast scoreboard (top corners). On amateur phone
        # footage the top corners are free too; bottom-left is the safe universal slot.
        if not args.no_fx:
            # Persistent event-key (shape = event, color = attribute), top-right.
            out = draw_event_legend(out, fx)

            margin = int(frame_height * 0.022)
            gauge_r = int(frame_height * 0.052)
            gauge_box = 2 * gauge_r + 2 * int(gauge_r * 0.42)
            card_w = int(frame_width * 0.16)   # wider so "Faute provoquée" fits

            # ── Which rally are we in? ──────────────────────────────────────
            # rally_at_frame is None in the inter-rally gap; freeze the last
            # rally shown so the card doesn't blank between points. Reset the
            # per-rally peak when the active rally changes (new ÉCHANGE).
            cur_idx = rally_at_frame[frame_idx] if frame_idx < len(rally_at_frame) else None
            if cur_idx is None:
                cur_idx = hud_rally_idx  # freeze last point's tag in the gap
            if cur_idx != hud_rally_idx:
                hud_rally_idx = cur_idx
                peak_speed = 0.0        # RESET: new échange → fresh peak

            rally_live = False          # is the current rally still in progress?
            if hud_rally_idx is not None and hud_rally_idx < len(rallies_rel):
                r = rallies_rel[hud_rally_idx]
                r_start = r["start_frame"]
                # current-rally counters: shots/bounces from this rally's start up
                # to (and including) the current frame — NOT cumulative since clip
                # start. Reset to 0 the instant a new rally begins.
                n_fh = sum(1 for f in r["shot_frames"]
                           if r_start <= f <= frame_idx and shot_by_frame[f]["fhb"] == "forehand")
                n_bh = sum(1 for f in r["shot_frames"]
                           if r_start <= f <= frame_idx and shot_by_frame[f]["fhb"] == "backhand")
                n_strikes = sum(1 for f in r["shot_frames"] if r_start <= f <= frame_idx)
                n_bounce = sum(1 for f in r["bounce_frames"] if r_start <= f <= frame_idx)
                # "Balle max" = the fastest EVENT speed of this rally so far
                # (shot/bounce speeds are already windowed averages — the same
                # numbers the _stats.json and the web read). The raw per-frame
                # series max was spiky even under a short display median (a
                # 3-frame homography-amplified jitter printed 194-224 km/h on a
                # ~105 km/h rally); event speeds are robust by construction and
                # keep the card consistent with every other surface.
                ev_speeds = (
                    [shot_by_frame[f]["speed"] for f in r["shot_frames"]
                     if r_start <= f <= frame_idx and f in shot_by_frame]
                    + [bounce_by_frame[f][3] for f in r["bounce_frames"]
                       if r_start <= f <= frame_idx and f in bounce_by_frame])
                peak_speed = max([s for s in ev_speeds if s], default=0.0)
                rally_no = hud_rally_idx + 1
                # outcome only once the point is OVER (don't reveal mid-rally)
                rally_live = frame_idx < r["end_frame"]
                oc = rally_outcome_rel[hud_rally_idx] if not rally_live else None
                outcome_label = oc.get("outcome") if oc else None
            else:
                # before the first rally starts: empty card with a neutral tag
                n_fh = n_bh = n_strikes = n_bounce = 0
                rally_no = 1
                outcome_label = None
                rally_live = True

            # Issue (outcome) row: French label + semantic color. Shown as
            # "En cours" while the point is live; honest "—" if finished without a
            # decisive geometric signal (we never invent a winner).
            _ISSUE = {
                "winner":         ("Coup gagnant",    (90, 220, 90)),
                "forced_error":   ("Faute provoquée", (90, 220, 220)),  # low-conf → amber
                "unforced_error": ("Faute directe",   (90, 90, 230)),
            }
            if rally_live:
                issue_text, issue_color = "En cours", (190, 195, 205)
            elif outcome_label in _ISSUE:
                issue_text, issue_color = _ISSUE[outcome_label]
            else:
                issue_text, issue_color = "—", (190, 195, 205)

            rally_label = f"ÉCHANGE {rally_no}"
            card_lines = [
                ("Frappes",   str(n_strikes)),
                ("CD / RV",   f"{n_fh} / {n_bh}"),
                ("Rebonds",   str(n_bounce)),
                ("Balle max", f"{int(peak_speed)} km/h"),
                ("Issue",     issue_text),
            ]

            # Card height must match the renderer's actual layout so the gauge
            # stacks above it without overlap. fx.stat_card uses:
            #   title_sz = max(14, fh*0.022); row_sz = max(12, fh*0.019)
            #   pad = max(12, width*0.07); row_h = int(row_sz*1.55)
            #   header_h = title_sz + int(row_sz*0.9)  (title always present here)
            #   h = pad + header_h + row_h*n_lines + pad
            _title_sz = max(14, int(frame_height * 0.022))
            _row_sz = max(12, int(frame_height * 0.019))
            _pad = max(12, int(card_w * 0.07))
            _row_h = int(_row_sz * 1.55)
            _header_h = _title_sz + int(_row_sz * 0.9)
            card_h = _pad + _header_h + _row_h * len(card_lines) + _pad
            # stack: gauge on top, stats card under it, bottom-left aligned
            card_y = frame_height - margin - card_h
            gauge_y = card_y - gauge_box - int(frame_height * 0.012)
            fx.speed_gauge(out, margin, gauge_y, current_speed if bc is not None else 0.0,
                           max_kmh=180, label="BALL", radius=gauge_r)

            # Prefer S4's per-rally card variant (header "ÉCHANGE N" + colored
            # outcome row) when fx provides it; otherwise fall back to stat_card
            # with the rally label as the title + a semantic-colored Issue overdraw.
            _rally_card = getattr(fx, "rally_card", None)
            _used_variant = False
            if callable(_rally_card):
                try:
                    _rally_card(out, margin, card_y, rally_index=rally_no,
                                lines=card_lines[:-1], outcome=outcome_label,
                                width=card_w)
                    _used_variant = True
                except TypeError:
                    _used_variant = False  # signature mismatch → fall through
            if not _used_variant:
                fx.stat_card(out, margin, card_y, card_lines, title=rally_label,
                             width=card_w)
                # stat_card draws every value white → overdraw the Issue value in
                # its semantic color (last row, right-anchored at the value column).
                _issue_baseline = (card_y + _pad + _header_h
                                   + _row_h * (len(card_lines) - 1))
                fx.draw_text(out, issue_text,
                             (margin + card_w - _pad, _issue_baseline),
                             size=_row_sz, color=issue_color, font="bold", anchor="rt")
        else:
            out = create_legend(out, legend_items, position="top-right")

        writer.write_frame(out)
        frame_idx += 1

    reader.release()
    writer.release()

    # ─── Match Summary ───
    if bounce_by_frame:
        logger.info("=== MATCH SUMMARY ===")
        all_depths = [v[2] for v in bounce_by_frame.values()]
        all_bspeeds = [v[3] for v in bounce_by_frame.values()]
        all_qualities = [v[4] for v in bounce_by_frame.values()]

        n = len(all_depths)
        deep_n = sum(1 for d in all_depths if d == "deep")
        mid_n = sum(1 for d in all_depths if d == "mid")
        short_n = sum(1 for d in all_depths if d == "short")
        logger.info(f"Total shots: {n}")
        logger.info(f"Depth: deep={deep_n} ({100*deep_n/n:.0f}%), "
                    f"mid={mid_n} ({100*mid_n/n:.0f}%), "
                    f"short={short_n} ({100*short_n/n:.0f}%)")

        speeds_arr = np.array(all_bspeeds)
        logger.info(f"Speed: avg={np.mean(speeds_arr):.0f} km/h, "
                    f"max={np.max(speeds_arr):.0f} km/h, "
                    f"min={np.min(speeds_arr):.0f} km/h")

        quals_arr = np.array(all_qualities)
        high_q = sum(1 for q in all_qualities if q >= 70)
        logger.info(f"Quality: avg={np.mean(quals_arr):.0f}, "
                    f"max={np.max(quals_arr):.0f}, "
                    f"min={np.min(quals_arr):.0f}, "
                    f"high (>=70): {high_q}/{n} ({100*high_q/n:.0f}%)")

        # ─── Rally segmentation ───
        # Group shots into rallies: a gap > 2.5s between consecutive shots
        # breaks a rally (threshold is a fraction of fps — zero hardcoding).
        rallies = []
        if shot_by_frame:
            rally_gap_frames = 2.5 * fps
            sorted_sf = sorted(shot_by_frame.keys())
            grouped = []
            cur = [sorted_sf[0]]
            for sf in sorted_sf[1:]:
                # shot-guard: court-frames-only gap (mirror segment_rallies_relative)
                # so the JSON rallies match the live HUD numbering exactly.
                gap = (_court_frames_between(court_visible_mask, cur[-1], sf)
                       if shot_guard_enabled else sf - cur[-1])
                if gap > rally_gap_frames:
                    grouped.append(cur)
                    cur = []
                cur.append(sf)
            grouped.append(cur)
            sorted_bf = sorted(bounce_by_frame.keys())
            for g in grouped:
                start = g[0]
                end = max(g[-1], max([bf for bf in sorted_bf if start <= bf <= g[-1] + rally_gap_frames], default=g[-1]))
                bframes = [bf for bf in sorted_bf if start <= bf <= end]
                rallies.append({
                    "start_frame": int(start_frame + start),
                    "end_frame": int(start_frame + end),
                    "shot_frames": [int(start_frame + sf) for sf in g],
                    "bounce_frames": [int(start_frame + bf) for bf in bframes],
                    "n_shots": len(g),
                })
            logger.info(f"Rallies: {len(rallies)} "
                        f"(longest={max((r['n_shots'] for r in rallies), default=0)} shots)")

            # Merge solo (1-shot) rallies into their nearest neighbour. A solo
            # arises when a strike sits >gap from both neighbours — a genuine
            # isolated hit — but surfacing it as its own point is noise in the
            # highlights. Must match web/src/lib/analysis/highlights.ts
            # mergeSoloRallies so the pipeline JSON and the front-end agree.
            rallies = _merge_solo_rallies(rallies)

        # Shot-type stats
        if shot_by_frame:
            fh = sum(1 for s in shot_by_frame.values() if s["fhb"] == "forehand")
            bh = sum(1 for s in shot_by_frame.values() if s["fhb"] == "backhand")
            logger.info(f"Strokes: {len(shot_by_frame)} hits — forehand={fh}, backhand={bh}")

        # ─── Derived analytics (Session B): per-rally outcome + fatigue ───
        # Enrich each rally with a geometric outcome label (winner / forced_error /
        # unforced_error / neutral). Note: forced_error is LOW confidence — the
        # frontend treats it as such. Both modules are pure functions over the
        # shots/bounces and degrade gracefully (no shots → no outcome, fatigue none).
        stats_shots = [{"frame": int(start_frame + f), "x": s["x"], "y": s["y"],
                        "player_side": s["side"], "stroke": s["fhb"],
                        "quality": s["quality"], "speed_kmh": round(s["speed"], 1),
                        "human_id": s.get("human_id")}
                       for f, s in sorted(shot_by_frame.items())]
        # CHANTIER 2: direction_state tags the bounce-vs-hit confidence from the
        # vx-flip veto ("metric" = vx preserved + vy flip; "unconfirmed" = kept
        # but not direction-confirmed: occluded / sparse / slow / Kalman path /
        # demo-override). Presentation only — no detection risk.
        stats_bounces = [{"frame": int(start_frame + f), "x": int(v[0]), "y": int(v[1]),
                          "depth": v[2], "speed_kmh": round(v[3], 1), "quality": v[4],
                          "direction_state": direction_state_by_frame.get(f, "unconfirmed")}
                         for f, v in sorted(bounce_by_frame.items())]
        for r in rallies:
            r["outcome"] = classify_rally_outcome(r, stats_shots, stats_bounces, fps)
            # In match mode, tag the rally with the stable human_id of its last
            # striker (the rally-ending player) on top of the near/far side. The
            # outcome label still uses player_side (handled by rally_outcome.py).
            if match_mode:
                last_shot_frame = r["shot_frames"][-1] if r.get("shot_frames") else None
                last_rel = (last_shot_frame - start_frame) if last_shot_frame is not None else None
                if last_rel is not None and last_rel in shot_by_frame:
                    r["human_id"] = shot_by_frame[last_rel].get("human_id")
                else:
                    r["human_id"] = None
        stats_fatigue = analyze_fatigue(stats_shots, [start_frame, end_frame], fps)

        # ─── Structured stats JSON (the actionable data output) ───
        stats = {
            "video": video_path,
            "fps": fps,
            "frame_range": [start_frame, end_frame],
            "speed_source": speed_source,
            "match_mode": match_mode,
            "match": ({"side_swaps": (match_result["swaps"] if match_result else []),
                       "n_swaps": (len(match_result["swaps"]) if match_result else 0)}
                      if match_mode else None),
            "homography_confidence": (court_homography.get("confidence")
                                      if court_homography else "none"),
            # 3x3 image-px -> court-meters homography (row-major; origin at the
            # court CENTER, +Y toward the far baseline — the exact convention
            # vision/minimap.py and the web front's projection.ts share). The
            # front no longer needs to synthesize an approximate H: when this is
            # present the minimap/placement views are metric-exact.
            "homography_H": (np.asarray(court_homography["H"]).tolist()
                             if court_homography and court_homography.get("H") is not None
                             else None),
            "summary": {
                "total_bounces": n,
                "depth": {"deep": deep_n, "mid": mid_n, "short": short_n},
                "speed_kmh": {"avg": float(np.mean(speeds_arr)),
                              "max": float(np.max(speeds_arr)),
                              "min": float(np.min(speeds_arr))},
                "quality": {"avg": float(np.mean(quals_arr)),
                            "max": int(np.max(quals_arr)),
                            "high_count": int(high_q)},
                "strokes": ({"forehand": sum(1 for s in shot_by_frame.values() if s["fhb"] == "forehand"),
                             "backhand": sum(1 for s in shot_by_frame.values() if s["fhb"] == "backhand")}
                            if shot_by_frame else {}),
            },
            "shot_guard": {
                "enabled": shot_guard_enabled,
                "court_frames": int(sum(court_visible_mask)),
                "total_frames": int(len(court_visible_mask)),
                "cut_frames": int(len(court_visible_mask) - sum(court_visible_mask)) if shot_guard_enabled else 0,
                "non_court_ratio": (1.0 - sum(court_visible_mask) / max(1, len(court_visible_mask)))
                                   if shot_guard_enabled else 0.0,
                "non_court_spans": non_court_spans_rel,
            },
            "bounces": stats_bounces,
            "shots": stats_shots,
            "rallies": rallies,
            "fatigue": stats_fatigue,
        }
        stats_path = str(Path(output_path).with_suffix("")) + "_stats.json"
        try:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Stats written to: {stats_path} "
                        f"(fatigue: {stats_fatigue['confidence']}, "
                        f"rallies: {len(rallies)})")
        except Exception as exc:  # never let stats emission break the video output
            logger.warning(f"Stats JSON not written: {exc}")

        # ─── Clip-index sidecar (annotated↔original frame mapping for the front) ───
        # Always emitted (identity mapping when shot-guard is off / nothing cut), so
        # the front can treat "no sidecar" and "identity sidecar" the same.
        clip_path, clip_index = write_clip_index(output_path, court_visible_mask,
                                                 fps, start_frame)
        if clip_path:
            logger.info(f"Clip-index written to: {clip_path} "
                        f"({clip_index['annotated_total']}/{clip_index['original_total']} kept)")

    # ─── Re-attach the source audio, CUT to the same kept spans (perfect sync) ───
    # The pipeline's VideoWriter produces a silent video; we mux the original audio
    # back on, trimmed to the kept time intervals so it never drifts past a cut.
    # With shot-guard off the mask is all-True → one full span → the whole audio.
    _remux_audio(video_path, output_path, court_visible_mask, fps, start_frame)

    logger.info(f"Done! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
