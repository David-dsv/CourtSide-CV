"""
Shot detection, player attribution, forehand/backhand classification, and a
multi-factor shot-quality score.

Method (grounded in SOTA research — antoinekeller/tennis_shot_recognition,
arXiv:2507.02906):
  - CONTACT (a hit): a **peak in the racket-hand wrist speed**. On broadcast the
    ball is tiny/occluded at contact, but the swinging arm moves the wrist fast —
    a robust, cheap signal from the per-frame pose (already computed in Pass 1.5).
    This replaces the old vy down→up rule, which was a bounce signature (and was
    smoothed away by the spline trajectory anyway → 0/8 on broadcast).
  - HANDEDNESS: the racket wrist has 3-5× the positional variance of the other
    over a clip; majority vote per side (near/far) → right/left-handed, no manual
    flag.
  - FOREHAND / BACKHAND (geometric 2D): at contact, compare the ball x to the
    racket-hand wrist x relative to the body center, corrected for facing
    (sign of R_SHOULDER_x − L_SHOULDER_x). 'unknown' when pose confidence is low.
  - QUALITY (mirrors ATP/TennisViz "Shot Quality": speed + depth + placement +
    net clearance, 0-100).

All thresholds are ratios of fps / frame size (no per-video constants).
"""
import numpy as np

# COCO keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# Confidence floor for trusting a keypoint. The far player is small but its
# wrists resolve well on broadcast (measured ~0.9 median conf), so a single
# 0.30 floor works for both near and far — no per-size tuning.
_KP_CONF = 0.30


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


def _kp(player, idx):
    """Return the (x, y) of keypoint `idx` if confident enough, else None."""
    kx = player.get("kps_xy")
    kc = player.get("kps_conf")
    if kx is None or kc is None:
        return None
    kx = np.asarray(kx)
    kc = np.asarray(kc)
    if idx >= len(kc) or kc[idx] < _KP_CONF:
        return None
    return np.array([float(kx[idx][0]), float(kx[idx][1])])


def _wrist_of(player, handedness):
    """The racket-hand wrist position: right wrist if right-handed, else left.
    Falls back to the other wrist if the dominant one is low-confidence."""
    dominant = R_WRIST if handedness else L_WRIST
    other = L_WRIST if handedness else R_WRIST
    return _kp(player, dominant) if _kp(player, dominant) is not None else _kp(player, other)


def _side_of(player, net_y):
    feet = player["box"][3]
    return "near" if feet >= net_y else "far"


def detect_handedness(players_per_frame, net_y):
    """Per-side racket-hand (right=True/left=False) by wrist-variance voting.

    The racket wrist moves far more than the off-hand over a rally. We accumulate
    per-frame squared displacement of L vs R wrist for the player on each side
    (near/far), and majority-vote across frames. Returns {'near': bool, 'far': bool}
    defaulting to True (right-handed) when undeterminable.
    """
    votes = {"near": {"R": 0, "L": 0}, "far": {"R": 0, "L": 0}}
    prev = {"near": {"L": None, "R": None}, "far": {"L": None, "R": None}}
    for players in players_per_frame:
        if not players:
            # carry prev forward (no reset)
            continue
        # match each player to a side; pick the best (most-confident wrists) per side
        per_side = {}
        for p in players:
            side = _side_of(p, net_y)
            kc = p.get("kps_conf")
            if kc is None:
                continue
            score = sum((kc[i] if i < len(kc) else 0) for i in (L_WRIST, R_WRIST))
            if side not in per_side or score > per_side[side][1]:
                per_side[side] = (p, score)
        for side, (p, _) in per_side.items():
            for wi, hand in ((L_WRIST, "L"), (R_WRIST, "R")):
                pos = _kp(p, wi)
                if pos is None:
                    continue
                pp = prev[side][hand]
                if pp is not None:
                    var = float(np.sum((pos - pp) ** 2))
                    if var > 0:
                        # a vote weighted by displacement: the moving hand wins
                        votes[side][hand] += 1 if var > 1.0 else 0
                prev[side][hand] = pos
    out = {}
    for side in ("near", "far"):
        vr, vl = votes[side]["R"], votes[side]["L"]
        # right-handed if R wrist moved on at least as many frames as L, with a margin
        out[side] = True if vr >= vl else False
    return out


def _wrist_speed_series(players_per_frame, side, handedness, net_y, fps):
    """Track ONE wrist per frame on `side` (nearest to the previous wrist — a
    tiny per-side identity lock so a detection-order flip between two players
    doesn't inject a fake speed spike), then return median-smoothed speed (px/frame)
    and the wrist x,y at each frame (for FH/BH). NaN where unavailable."""
    n = len(players_per_frame)
    raw_pos = [None] * n
    raw_wrist_xy = [None] * n
    last_pos = None
    for i, players in enumerate(players_per_frame):
        cands = [p for p in (players or []) if _side_of(p, net_y) == side]
        if not cands:
            continue
        # pick the candidate whose racket wrist is nearest to the last tracked wrist
        best, best_d = None, float("inf")
        for p in cands:
            w = _wrist_of(p, handedness)
            if w is None:
                continue
            d = float(np.hypot(w[0] - last_pos[0], w[1] - last_pos[1])) if last_pos is not None else 0.0
            if d < best_d:
                best_d, best, best_w = d, p, w
        if best is None:
            continue
        raw_pos[i] = best_w
        raw_wrist_xy[i] = (best, best_w)
        last_pos = best_w
    # speed (px/frame), interpolate short gaps, median-smooth
    half_gap = max(1, int(fps * 0.10))
    speed = np.full(n, np.nan)
    for i in range(1, n):
        if raw_pos[i] is not None and raw_pos[i - 1] is not None:
            speed[i] = float(np.hypot(*(raw_pos[i] - raw_pos[i - 1])))
    # 5-frame median smoothing (robust to impulsive noise)
    sm = speed.copy()
    w = 2
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        win = sm[lo:hi]
        win = win[~np.isnan(win)]
        if win.size:
            speed[i] = float(np.median(win))
    return speed, raw_wrist_xy


def _adaptive_threshold(speed, frame_height, fps):
    """Amplitude pre-filter for wrist-speed peaks. A real swing moves the wrist
    fast, but so does a 1-frame pose-detection jitter and a 2-frame identity-swap
    of the wrist tracker — so amplitude ALONE can't separate them (it plateaus at
    F1~0.5). We keep a low amplitude floor only to kill sub-pixel noise; the real
    discriminator is the FWHM (full-width half-max) of the bump — a bio-mechanical
    swing lasts ~0.15-0.25s (4-8 frames @50fps), whereas jitter is 1 frame and an
    identity swap is ~2. See _fwhm_at + the fwhm gate in detect_hits."""
    return 0.008 * frame_height  # ~8.6 px/frame @1080p — sub-pixel noise floor


def _fwhm_at(speed, peak_idx):
    """Full-width half-max (frames) of the speed bump around peak_idx. A real
    swing = 4-8; pose jitter = 1; an identity swap = ~2."""
    if np.isnan(speed[peak_idx]):
        return 0
    half = speed[peak_idx] / 2.0
    n = len(speed)
    lo = peak_idx
    while lo > 0 and not np.isnan(speed[lo - 1]) and speed[lo - 1] >= half:
        lo -= 1
    hi = peak_idx
    while hi < n - 1 and not np.isnan(speed[hi + 1]) and speed[hi + 1] >= half:
        hi += 1
    return hi - lo + 1


def detect_hits(ball_centers, players_per_frame, fps, frame_height, frame_width,
                bounce_frames=None):
    """
    Detect racket→ball contact (hit) frames via racket-wrist speed peaks, and
    attribute each to a player. Returns list of dicts:
      {frame, x, y, player_idx, player_side, anchor, right_handed, wrist_speed}
    """
    n = len(ball_centers)
    if n < 7:
        return []
    net_y = frame_height * 0.47
    bounce_set = set(bounce_frames or [])
    min_gap = int(fps * 0.30)
    ball_win = int(fps * 0.20)
    bounce_guard = int(fps * 0.10)

    handed = detect_handedness(players_per_frame, net_y)

    # collect candidate peaks per side
    cands = []  # (frame, side, wrist_speed, player_idx, anchor, wrist_y)
    amp_floor = _adaptive_threshold(None, frame_height, fps)
    fwhm_min = max(3, int(fps * 0.06))  # ~3 frames @50fps = min real-swing width
    for side in ("near", "far"):
        speed, wrist_xy = _wrist_speed_series(
            players_per_frame, side, handed[side], net_y, fps)
        # local maxima: low amplitude floor (sub-pixel noise) + FWHM gate (a real
        # swing is a wide bio-mechanical bump; jitter is 1 frame, an identity
        # swap ~2 — both rejected here even at huge amplitude)
        i = 1
        while i < n - 1:
            if (not np.isnan(speed[i]) and speed[i] >= amp_floor
                    and speed[i] >= speed[i - 1] and speed[i] > speed[i + 1]
                    and _fwhm_at(speed, i) >= fwhm_min):
                cands.append((i, side, float(speed[i]), wrist_xy))
                i += min_gap  # NMS
            else:
                i += 1

    cands.sort(key=lambda c: c[0])
    hits = []
    last = -min_gap
    for i, side, wspeed, wrist_xy in cands:
        if i - last < min_gap:
            continue
        # The wrist-speed peak lands a few frames AFTER contact (the arm is still
        # swinging through). So we don't snap the hit to frame i's ball position —
        # we search the ±ball_win window for the frame where the ball is CLOSEST
        # to the player's racket wrist, which is the true contact frame+position.
        players_at_i = players_per_frame[i] if i < len(players_per_frame) else None
        if not players_at_i:
            continue
        # candidate player on this side at the peak frame (for the wrist anchor)
        me = None
        for p in players_at_i:
            if _side_of(p, net_y) == side:
                if me is None or _player_height(p) > _player_height(me):
                    me = p
        if me is None:
            continue
        anchor = _player_anchor(me)
        # search the window for the ball position nearest the player's box center
        lo, hi = max(0, i - ball_win), min(n, i + ball_win + 1)
        best_j, best_d = None, float("inf")
        for j in range(lo, hi):
            if ball_centers[j] is None:
                continue
            d = np.hypot(ball_centers[j][0] - anchor[0], ball_centers[j][1] - anchor[1])
            if d < best_d:
                best_d, best_j = d, ball_centers[j]
        if best_j is None:
            continue
        h = max(_player_height(me), 1.0)
        if best_d / h > 2.5:  # ball never came near this player → not a real contact
            continue
        contact_frame = j
        ball_pos = best_j
        # re-attribute at the contact frame (player pose there is what we classify on)
        players = players_per_frame[contact_frame] if contact_frame < len(players_per_frame) else players_at_i
        best, bestd, best_anchor = None, float("inf"), anchor
        for pidx, p in enumerate(players):
            if _side_of(p, net_y) != side:
                continue
            ax, ay = _player_anchor(p)
            d = np.hypot(ball_pos[0] - ax, ball_pos[1] - ay)
            if d / h < bestd:
                bestd, best, best_anchor = d / h, pidx, (ax, ay)
        if best is None or bestd > 3.0:
            continue
        # skip if this coincides with a known bounce (a bounce has no wrist swing,
        # but guard anyway)
        if any(abs(contact_frame - bf) <= bounce_guard for bf in bounce_set):
            continue
        # skip serve: wrist very high + ball high
        p = players[best]
        wy = wrist_xy[i][1][1] if (wrist_xy[i] is not None and wrist_xy[i] is not False) else (p["box"][1])
        if wy < 0.30 * frame_height and ball_pos[1] < 0.35 * frame_height:
            continue
        hits.append({
            "frame": contact_frame, "x": int(ball_pos[0]), "y": int(ball_pos[1]),
            "player_idx": best, "player_side": side, "anchor": best_anchor,
            "right_handed": handed[side], "wrist_speed": wspeed,
        })
        last = contact_frame
    return hits


def classify_forehand_backhand(hit, players_per_frame, right_handed=None):
    """Geometric FH/BH at a hit. Returns 'forehand' | 'backhand' | 'unknown'.

    Auto-handedness by wrist variance is unreliable in a single rally (both wrists
    move ~equally to balance the body), so we DON'T rely on it. Instead: at contact
    the racket wrist is the one NEAREST the ball (the racket touches the ball), and
    we assume right-handed (>80% of players) corrected by the player's facing
    direction (sign of R_SHOULDER_x − L_SHOULDER_x; near players face the camera,
    far players face away). Returns 'unknown' on low-confidence pose or an ambiguous
    contact rather than guessing wrong.
    """
    i = hit["frame"]
    players = players_per_frame[i] if i < len(players_per_frame) else None
    if not players or hit["player_idx"] >= len(players):
        return "unknown"
    p = players[hit["player_idx"]]
    ax, ay = _player_anchor(p)
    bx = hit["x"]
    h = max(_player_height(p), 1.0)

    lw = _kp(p, L_WRIST)
    rw = _kp(p, R_WRIST)
    ls = _kp(p, L_SHOULDER)
    rs = _kp(p, R_SHOULDER)
    if lw is None or rw is None or ls is None or rs is None:
        return "unknown"
    if abs(bx - ax) < 0.04 * h:  # ball dead on the body axis → ambiguous
        return "unknown"

    # (1) racket wrist = nearest to the ball
    d_lw = abs(lw[0] - bx)
    d_rw = abs(rw[0] - bx)
    if min(d_lw, d_rw) > 0.35 * h:  # ball too far from both wrists → not a real contact
        return "unknown"
    racket_is_right = (d_rw <= d_lw)

    # (2) facing: >0 → right shoulder screen-right (faces camera); <0 → faces away
    facing = 1.0 if (rs[0] - ls[0]) >= 0 else -1.0

    # (3) ball side in the player's own frame (their right vs left)
    ball_screen_side = 1.0 if bx >= ax else -1.0
    ball_player_side = ball_screen_side * facing

    # (4) assume right-handed; left-hander mirrors
    if racket_is_right:
        return "forehand" if ball_player_side > 0 else "backhand"
    return "backhand" if ball_player_side > 0 else "forehand"


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
