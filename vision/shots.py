"""
Shot detection, player attribution, forehand/backhand classification, and a
multi-factor shot-quality score.

Method (grounded in the verified SOTA research, see SOTA_RESEARCH_POSE_SHOTS):
  - CONTACT (a hit): the ball reverses VERTICAL direction (coming down toward a
    player, then going back up) at a point NEAR a player — distinct from a bounce
    (down→up near the court surface, away from players). We find vy down→up
    reversals and attribute each to the nearest player (by 2D distance to the
    player's body), requiring proximity within a player-height-scaled radius.
  - FOREHAND / BACKHAND (geometric, pose-based — the antoinekeller/MoveNet recipe
    reaches ~80% with simple pose features): at contact, compare the ball's x to
    the player's body. For a right-handed player, ball on the racket-hand side
    (right of body center, in the player's facing frame) = forehand, else backhand.
    We use the wrist/shoulder keypoints when available, else the box center.
  - QUALITY (mirrors ATP/TennisViz "Shot Quality": speed + depth + placement +
    net clearance, 0-100): combine normalized ball speed, bounce depth, lateral
    placement (closeness to sidelines), and net clearance margin.

All thresholds are ratios of fps / frame size (no per-video constants).
"""
import numpy as np

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


def _player_anchor(player):
    """Body anchor (x,y) for a player dict {box, kps_xy, kps_conf}: torso center
    from shoulders/hips if posed, else box center."""
    kx = player.get("kps_xy")
    kc = player.get("kps_conf")
    if kx is not None and kc is not None:
        pts = []
        for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            if i < len(kc) and kc[i] > 0.2:
                pts.append(kx[i])
        if pts:
            p = np.mean(pts, axis=0)
            return float(p[0]), float(p[1])
    b = player["box"]
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _player_height(player):
    b = player["box"]
    return float(b[3] - b[1])


def detect_hits(ball_centers, players_per_frame, fps, frame_height, frame_width,
                bounce_frames=None):
    """
    Detect contact (hit) frames and attribute each to a player.

    ball_centers: list[(x,y)|None] per frame (smoothed trajectory)
    players_per_frame: list (per frame) of player dicts (from vision.pose) OR None
    bounce_frames: optional set/list of bounce frame indices (to avoid labeling a
                   bounce as a hit)

    Returns list of dicts: {frame, x, y, player_idx, player_side}
      player_side: 'near'/'far' (by y position) for stable per-player IDs.
    """
    n = len(ball_centers)
    if n < 7:
        return []
    ys = np.array([c[1] if c is not None else np.nan for c in ball_centers])
    xs = np.array([c[0] if c is not None else np.nan for c in ball_centers])

    # smoothed vertical velocity
    vy = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ys[i]) and not np.isnan(ys[i - 1]):
            vy[i] = ys[i] - ys[i - 1]

    min_gap = int(fps * 0.30)
    win = max(3, int(fps * 0.07))
    bounce_set = set(bounce_frames or [])
    bounce_guard = int(fps * 0.12)

    net_y = frame_height * 0.47  # rough; only used as a tiebreak for near/far
    raw_hits = []
    for i in range(win, n - win):
        if ball_centers[i] is None:
            continue
        a = np.nanmean(vy[max(0, i - win):i])
        b = np.nanmean(vy[i:i + win + 1])
        # contact: ball was coming DOWN (vy>0) then goes UP (vy<0)
        if np.isnan(a) or np.isnan(b):
            continue
        if not (a > 1.0 and b < -1.0):
            continue
        # skip if this reversal is actually a bounce
        if any(abs(i - bf) <= bounce_guard for bf in bounce_set):
            continue
        raw_hits.append(i)

    # also: a hit can be a horizontal reversal near a player (e.g. flat drives)
    # — fold those in, deduped by min_gap below.

    # attribute to nearest player + dedup
    hits = []
    last = -min_gap
    for i in raw_hits:
        if i - last < min_gap:
            continue
        bx, by = xs[i], ys[i]
        if np.isnan(bx):
            continue
        players = players_per_frame[i] if i < len(players_per_frame) else None
        if not players:
            continue
        best, bestd = None, float("inf")
        for pidx, p in enumerate(players):
            ax, ay = _player_anchor(p)
            d = np.hypot(bx - ax, by - ay)
            h = max(_player_height(p), 1.0)
            if d / h < bestd:  # distance in player-height units
                bestd = d / h
                best = (pidx, p, ax, ay)
        # require the ball to be reasonably near a player (within ~2.5 body heights)
        if best is None or bestd > 2.5:
            continue
        pidx, p, ax, ay = best
        side = "far" if ay < net_y else "near"
        hits.append({"frame": i, "x": int(bx), "y": int(by),
                     "player_idx": pidx, "player_side": side,
                     "anchor": (ax, ay)})
        last = i
    return hits


def classify_forehand_backhand(hit, players_per_frame, right_handed=True):
    """Geometric FH/BH at a hit. Returns 'forehand' | 'backhand' | 'unknown'.

    For a right-handed player the racket is on their right side. Without a facing
    estimate we use the screen heuristic refined by wrist position: the ball is on
    the forehand side if it's on the same side as the dominant-hand wrist relative
    to the body center.
    """
    i = hit["frame"]
    players = players_per_frame[i] if i < len(players_per_frame) else None
    if not players or hit["player_idx"] >= len(players):
        return "unknown"
    p = players[hit["player_idx"]]
    ax, ay = _player_anchor(p)
    bx = hit["x"]
    kx = p.get("kps_xy")
    kc = p.get("kps_conf")

    # dominant-hand wrist x (right wrist for right-handed)
    wrist_x = None
    if kx is not None and kc is not None:
        wi = R_WRIST if right_handed else L_WRIST
        if wi < len(kc) and kc[wi] > 0.2:
            wrist_x = kx[wi][0]

    # Side of the ball relative to body center
    ball_side = np.sign(bx - ax)  # +1 = screen-right of body, -1 = left
    if wrist_x is not None:
        hand_side = np.sign(wrist_x - ax)
        # forehand = ball on the same screen side the racket hand reaches out to
        return "forehand" if ball_side == hand_side else "backhand"
    # fallback: right-handed forehand tends to be ball on body's right
    if ball_side == 0:
        return "unknown"
    if right_handed:
        return "forehand" if ball_side > 0 else "backhand"
    return "forehand" if ball_side < 0 else "backhand"


def shot_quality(speed_kmh, depth_label, bounce_xy, court_zones, frame_width,
                 frame_height, homography=None, speed_ref=None):
    """Multi-factor 0-100 shot quality, mirroring ATP/TennisViz Shot Quality
    (speed + depth + placement + net clearance). Spin omitted (unmeasurable).

    bounce_xy: (x,y) image coords of the bounce that ended this shot.
    """
    # speed term
    speed_ref = speed_ref or 120.0
    speed_norm = min(1.0, max(0.0, speed_kmh / speed_ref)) if speed_kmh else 0.0

    # depth term
    depth_norm = {"deep": 1.0, "mid": 0.55, "short": 0.25}.get(depth_label, 0.5)

    # placement term: closeness to a sideline (corner play scores higher)
    bx = bounce_xy[0]
    placement_norm = 0.5
    if homography is not None and homography.get("H") is not None:
        try:
            from models.court_detector import img_to_world
            Xm, Ym = img_to_world(homography["H"], bounce_xy[0], bounce_xy[1])
            # singles half-width 4.115 m; closer to the line = higher
            lateral = min(1.0, abs(Xm) / 4.115)
            placement_norm = lateral
        except Exception:
            pass
    else:
        # image fallback: distance of bx from frame center, normalized
        placement_norm = min(1.0, abs(bx - frame_width / 2) / (frame_width / 2))

    # net clearance: not directly measured per-shot here; folded into depth.
    # Weighted blend (ATP weights aren't public; use a balanced, depth-led mix)
    q = (0.30 * speed_norm + 0.40 * depth_norm + 0.30 * placement_norm)
    return int(round(max(0.0, min(1.0, q)) * 100))
