"""Rally outcome classification (geometric, no ML).

Labels each rally as one of:
    winner / forced_error / unforced_error / neutral

The decision is built only from geometric, defensible signals available in the
pipeline stats — shot speed/quality, bounce depth/position, and timing. Where a
signal is unreliable we return "neutral" rather than guess (see the honest
caveats in classify_rally_outcome's docstring).

Public API
----------
    classify_rally_outcome(rally, shots, bounces, fps) -> dict

``rally`` is a rally dict from the stats JSON::

    {"start_frame": int, "end_frame": int,
     "shot_frames":   [int, ...],   # absolute frames
     "bounce_frames": [int, ...],   # absolute frames
     "n_shots": int}

``shots`` / ``bounces`` are the full clip lists (each entry has at least
``frame``; shots also carry ``player_side``/``speed_kmh``/``quality``, bounces
carry ``x``/``y``/``depth``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# ─── Ratio thresholds (no absolute pixel / km/h values) ──────────────────────
# A near shot counts as a "big" attacking shot if its speed is at least this
# fraction above the rally's own mean shot speed.
WINNER_SPEED_RATIO = 1.10            # ≥ 110 % of rally mean speed
# A "deep" placement after a big shot is what makes a winner (opponent can't reach).
# Quality threshold below which a near shot is "weak" (poorly struck).
WEAK_QUALITY = 45                    # < 45/100 quality
# A rally "ends shortly after" a shot if the next event (bounce or rally end) is
# within this many seconds — i.e. the opponent did not rally back.
NO_RETURN_GAP_S = 1.2
# A bounce whose x or y lies more than this many (rally) std devs from the rally
# bounce centroid is treated as "out of court".
OUTLIER_STDEV = 2.5
# "Very short" placement = the bounce lands much nearer the hitter than the
# rally's typical bounce depth (proxied by y for a near-far camera axis).
SHORT_DEPTH_RATIO = 0.6              # y within 60 % of the rally's min-y edge


def _index_by_frame(events: Sequence[dict], start_frame: int, end_frame: int) -> list[dict]:
    """Return events whose frame is within [start_frame, end_frame], sorted."""
    in_range = [e for e in events if start_frame <= e.get("frame", -1) <= end_frame]
    return sorted(in_range, key=lambda e: e["frame"])


def _last_near_shot(rally_shots: Sequence[dict]):
    """Last shot in the rally hit by the near player, or None."""
    near = [s for s in rally_shots if s.get("player_side") == "near"]
    return near[-1] if near else None


def _last_shot(rally_shots: Sequence[dict]):
    """The chronologically last shot of the rally (any side), or None."""
    return rally_shots[-1] if rally_shots else None


def _bounce_after(bounces: Sequence[dict], frame: int):
    """First bounce strictly after ``frame`` within the rally, or None."""
    later = [b for b in bounces if b.get("frame", -1) > frame]
    return min(later, key=lambda b: b["frame"]) if later else None


def _is_out_of_court(bounce: dict, rally_bounces: Sequence[dict]) -> bool:
    """True if a bounce lands well outside the rally's bounce cluster.

    Uses the rally's own bounce centroid + spread (ratio / std-dev based), so it
    adapts to any resolution and camera. Falls back to False (not out) when there
    are too few bounces to establish a spread.
    """
    if len(rally_bounces) < 3:
        return False
    xs = np.array([b["x"] for b in rally_bounces], dtype=float)
    ys = np.array([b["y"] for b in rally_bounces], dtype=float)
    sx, sy = float(xs.std()), float(ys.std())
    if sx <= 0 and sy <= 0:
        return False
    dx = (bounce["x"] - float(xs.mean())) / sx if sx > 0 else 0.0
    dy = (bounce["y"] - float(ys.mean())) / sy if sy > 0 else 0.0
    return (dx * dx + dy * dy) > OUTLIER_STDEV * OUTLIER_STDEV


def _is_very_short(bounce: dict, rally_bounces: Sequence[dict]) -> bool:
    """True if a bounce is very short (near the net / hitter), via depth or geometry."""
    depth = str(bounce.get("depth", "")).lower()
    if depth == "short":
        return True
    if len(rally_bounces) >= 3:
        ys = np.array([b["y"] for b in rally_bounces], dtype=float)
        ymin, ymax = float(ys.min()), float(ys.max())
        # On a near-baseline camera the near player's own side has the smallest y;
        # a bounce there = the ball died near the hitter (net/short).
        span = ymax - ymin
        if span > 0 and bounce["y"] <= ymin + SHORT_DEPTH_RATIO * span:
            return True
    return False


def classify_rally_outcome(rally: dict, shots: Sequence[dict],
                           bounces: Sequence[dict], fps: float) -> dict:
    """Classify the outcome of a single rally.

    Returns
    -------
    dict ::
        {"outcome": "winner"|"forced_error"|"unforced_error"|"neutral",
         "defining_frame": int | None,   # frame to point the UI at
         "reason": str}                  # human-readable justification

    Decision logic (in order, first match wins):

    1. **winner** — the rally's last *near* shot is a big attacking shot
       (speed ≥ WINNER_SPEED_RATIO × rally mean) AND the very next bounce lands
       deep AND no return shot follows within NO_RETURN_GAP_S seconds. The near
       player hit a shot the opponent could not put back.

    2. **unforced_error** — the last near shot is weak (quality < WEAK_QUALITY)
       AND the resulting bounce is very short (net) or out of court. The near
       player made the mistake themselves.

    3. **forced_error** — INTENTIONALLY RARE / UNRELIABLE geometrically. We can
       only flag this when the rally *ends on an opponent shot* after a big
       attacking shot *by the opponent* and there is no clean winner signal.
       Because we cannot reliably tell "the near player was stretched" from
       "the near player just mishit", we return "neutral" unless the geometric
       evidence is unusually clear. Treat any "forced_error" label as low
       confidence.

    4. **neutral** — otherwise.

    Honest limitations
    ------------------
    * Without per-shot reach/pose difficulty, "forced_error" is largely a guess;
      we prefer "neutral". Do not weight forced_error heavily downstream.
    * "winner" depends on bounce depth being accurate ("deep") — when the depth
      classifier is uncertain this degrades to neutral, which is the safe choice.
    """
    start = int(rally["start_frame"])
    end = int(rally["end_frame"])
    gap_frames = NO_RETURN_GAP_S * fps

    r_shots = _index_by_frame(shots, start, end)
    r_bounces = _index_by_frame(bounces, start, end)

    speeds = [float(s["speed_kmh"]) for s in r_shots
              if s.get("speed_kmh") is not None]
    mean_speed = float(np.mean(speeds)) if speeds else 0.0

    last_near = _last_near_shot(r_shots)
    last_shot = _last_shot(r_shots)

    # ── 1. Winner ───────────────────────────────────────────────────────────
    if last_near is not None and mean_speed > 0:
        is_big = float(last_near.get("speed_kmh", 0.0)) >= WINNER_SPEED_RATIO * mean_speed
        nxt = _bounce_after(r_bounces, last_near["frame"])
        # Did anyone hit a return shot after the last near shot, soon enough?
        returns = [s for s in r_shots
                   if s["frame"] > last_near["frame"]
                   and s["frame"] - last_near["frame"] <= gap_frames]
        ends_soon = (end - last_near["frame"]) <= gap_frames
        if is_big and nxt is not None:
            deep = str(nxt.get("depth", "")).lower() == "deep"
            if deep and not returns:
                return {"outcome": "winner",
                        "defining_frame": int(last_near["frame"]),
                        "reason": "near player hit a big shot (≥%.0f%% rally mean) "
                                  "followed by a deep placement with no return"
                                  % (100 * WINNER_SPEED_RATIO)}

    # ── 2. Unforced error (near player's own mistake) ───────────────────────
    if last_near is not None and last_shot is not None:
        # Only attribute a fault to near if the LAST shot of the rally was near's
        # (i.e. the rally ended on their racquet / their shot's resulting bounce).
        rally_ended_on_near = last_shot["frame"] == last_near["frame"]
        weak = float(last_near.get("quality", 100.0)) < WEAK_QUALITY
        if rally_ended_on_near and weak:
            nxt = _bounce_after(r_bounces, last_near["frame"])
            bad_placement = False
            if nxt is not None:
                bad_placement = _is_very_short(nxt, r_bounces) or _is_out_of_court(nxt, r_bounces)
            # No opponent return = the rally died on near's shot.
            returns = [s for s in r_shots if s["frame"] > last_near["frame"]]
            if bad_placement and not returns:
                return {"outcome": "unforced_error",
                        "defining_frame": int(last_near["frame"]),
                        "reason": "rally ended on a weak near shot (quality < %d) "
                                  "with a poor placement (short/out), no opponent return"
                                  % WEAK_QUALITY}

    # ── 3. Forced error — deliberately conservative ─────────────────────────
    # The rally ended on an opponent shot (not near), and that opponent shot was
    # a big attacking shot that near could not answer. Even then this is soft,
    # so we keep it but document that it is low-confidence.
    if last_shot is not None and last_near is not None:
        ended_on_opponent = last_shot.get("player_side") != "near"
        if ended_on_opponent and mean_speed > 0:
            opp_big = float(last_shot.get("speed_kmh", 0.0)) >= WINNER_SPEED_RATIO * mean_speed
            nxt = _bounce_after(r_bounces, last_shot["frame"])
            if opp_big and nxt is not None and str(nxt.get("depth", "")).lower() == "deep":
                returns = [s for s in r_shots
                           if s["frame"] > last_shot["frame"]
                           and s["frame"] - last_shot["frame"] <= gap_frames]
                if not returns:
                    return {"outcome": "forced_error",
                            "defining_frame": int(last_shot["frame"]),
                            "reason": "rally ended on a big opponent shot near could "
                                      "not return (low confidence — geometric guess)"}

    # ── 4. Neutral ──────────────────────────────────────────────────────────
    return {"outcome": "neutral",
            "defining_frame": int(last_shot["frame"]) if last_shot else None,
            "reason": "no decisive geometric signal (rally ended without a clear "
                      "winner/error signature)"}
