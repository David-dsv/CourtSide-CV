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
  - FOREHAND / BACKHAND (geometric 2D COMPOSITE, see classify_forehand_backhand):
    ball-at-contact side (ball_x − anchor_x)/h as the primary signal, a pre-contact
    SWING-window cross-body GUARD that ABSTAINS on a confident disagreement, an
    abstention band near the body axis, and handedness as a voted per-side PRIOR.
    The old per-frame "facing" correction is GONE — it inverted the class (~6/8 →
    2/8); removing it is the measured quick-win. 'unknown' on low-confidence pose /
    an ambiguous or cross-body contact.
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
                bounce_frames=None, wrist_prox_max=float("inf"), far_aware=False):
    """
    Detect racket→ball contact (hit) frames via racket-wrist speed peaks, and
    attribute each to a player. Returns list of dicts:
      {frame, x, y, player_idx, player_side, anchor, right_handed, wrist_speed}

    far_aware (default False = byte-identical legacy behavior, used by the event
    ARBITRATION path so classify_events sees an unchanged candidate set): when True,
    two far-player fixes are enabled — (1) NMS is PER SIDE so a near contact doesn't
    suppress a far contact a few frames later, and (2) the serve gate is PLAYER-
    relative (wrist above the player's own head) instead of FRAME-relative (which
    flagged every legitimately-high far contact as a serve and dropped all far hits).
    This is opt-in because it CHANGES the live hit set, which would perturb the
    methodo arbitration; it is used ONLY to source well-attributed near+far contacts
    for FH/BH labeling, never for the arbitration votes.

    A wrist-speed peak alone over-fires: the arm also swings on a split-step,
    a follow-through, or a pose-tracker identity flip — none of which are a real
    racket→ball contact. The discriminator (now that the FAR player's pose is
    dense — far-select 0.9%→84.9%) is BALL↔RACKET-WRIST PROXIMITY: a real contact
    has the ball reach the racket wrist. We require, over the ±ball_win contact
    window, the MINIMUM ball↔wrist distance (in player-height units) to fall below
    ``wrist_prox_max``. The min is taken over the window — not at the wrist-speed
    peak frame, which LAGS contact by a few frames (the arm swings through), so
    the ball is already leaving at the peak. A None wrist (pose miss) ABSTAINS
    (the candidate is kept) rather than dropping — proximity can only ADD
    precision here, never silently cut recall on an unreadable pose.

    wrist_prox_max is a fraction of the player's bounding-box HEIGHT (scale-free:
    an arm+racket spans ~1.0-1.2·height, and the windowed-MIN over the contact
    window pulls a true contact well under that, so 1.2 means "the ball came within
    roughly a body-length of the racket at closest approach"). It DEFAULTS TO
    float('inf') — i.e. the gate is OFF and detect_hits is byte-identical to before
    for every existing caller (the --event-methodo path, felix, etc.). The LEGACY
    prod path opts in with wrist_prox_max≈1.2: on the demo3 legacy cache 1.0-1.2 is
    a stable plateau (hit FP 7→2, no real near-side contact lost, confusion 0/0);
    the 1.5-2.0 band is AVOIDED (it kills a hit that was shadowing a bounce-as-hit,
    re-introducing confusion_B→H). Keeping the default OFF is what makes this change
    safe for the methodo path (its firewall owns hit arbitration there).
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
    # Per-side amplitude floor, scaled by the side's APPARENT player size.
    # Wrist speeds are measured in pixels, and pixels scale with the player's
    # projected size: the far player's whole swing spans about half the pixels
    # of the near player's (measured demo3: far median wrist speed 3.6 px/f vs
    # near 7.4). A single frame-scaled floor is therefore blind to the far
    # player — the far GT swing @141 peaks at 7.57 px/f with FWHM 19 (an
    # unambiguous bio-mechanical swing) UNDER the 8.64 floor. Scaling the floor
    # by (side's median box height / tallest side's median box height) keeps
    # the LARGEST player's floor byte-identical (near behavior unchanged) and
    # lowers the smaller player's floor by pure perspective proportionality —
    # no new constant. Jitter stays rejected by the FWHM gate (a 1-2 frame
    # spike can't be 3+ frames wide), which was always the real discriminator.
    med_box_h = {}
    for side in ("near", "far"):
        hs = []
        for slot in players_per_frame:
            for p in (slot or []):
                if p is not None and _side_of(p, net_y) == side:
                    hs.append(_player_height(p))
        med_box_h[side] = float(np.median(hs)) if hs else 0.0
    tallest = max(med_box_h.values()) or 1.0
    for side in ("near", "far"):
        scale = (med_box_h[side] / tallest) if med_box_h[side] > 0 else 1.0
        side_floor = amp_floor * scale
        speed, wrist_xy = _wrist_speed_series(
            players_per_frame, side, handed[side], net_y, fps)
        # local maxima: low amplitude floor (sub-pixel noise) + FWHM gate (a real
        # swing is a wide bio-mechanical bump; jitter is 1 frame, an identity
        # swap ~2 — both rejected here even at huge amplitude)
        i = 1
        while i < n - 1:
            if (not np.isnan(speed[i]) and speed[i] >= side_floor
                    and speed[i] >= speed[i - 1] and speed[i] > speed[i + 1]
                    and _fwhm_at(speed, i) >= fwhm_min):
                cands.append((i, side, float(speed[i]), wrist_xy))
                i += min_gap  # NMS
            else:
                i += 1

    cands.sort(key=lambda c: c[0])
    hits = []
    # NMS: PER SIDE when far_aware (a rally alternates near↔far, so a near contact
    # must not suppress a far contact a few frames later — the global cursor silently
    # dropped every far hit whenever a near peak preceded it within min_gap, leaving
    # FH/BH blind on the far player). GLOBAL otherwise (legacy arbitration behavior).
    last_by_side = {"near": -min_gap, "far": -min_gap}

    def _refractory(side):
        return last_by_side[side] if far_aware else max(last_by_side.values())

    def _mark(side, frame):
        if far_aware:
            last_by_side[side] = frame
        else:
            last_by_side["near"] = last_by_side["far"] = frame

    for i, side, wspeed, wrist_xy in cands:
        if i - _refractory(side) < min_gap:
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
        best_j, best_d, best_fr = None, float("inf"), None
        for j in range(lo, hi):
            if ball_centers[j] is None:
                continue
            d = np.hypot(ball_centers[j][0] - anchor[0], ball_centers[j][1] - anchor[1])
            if d < best_d:
                best_d, best_j, best_fr = d, ball_centers[j], j
        if best_j is None:
            continue
        h = max(_player_height(me), 1.0)
        if best_d / h > 2.5:  # ball never came near this player → not a real contact
            continue
        # WRIST-PROXIMITY GATE: a real contact has the ball reach the RACKET WRIST,
        # not just the torso. The per-frame racket wrist comes from the wrist_xy
        # series (a (player, wrist) tuple, or None where the pose missed it). We
        # take the MIN ball↔wrist distance over the window (contact LAGS the speed
        # peak). A None wrist on the whole window ABSTAINS (kept) — proximity only
        # adds precision, it never cuts recall on an unreadable pose. The torso
        # search above already located the contact; this gate only filters phantom
        # swings (split-step / follow-through / a tracker identity flip) where the
        # ball never came near the racket.
        wmin, wmin_fr = float("inf"), None
        for j in range(lo, hi):
            if ball_centers[j] is None:
                continue
            wj = wrist_xy[j] if 0 <= j < len(wrist_xy) else None
            if wj is None or wj is False:
                continue
            wpos = wj[1]
            dw = np.hypot(ball_centers[j][0] - wpos[0], ball_centers[j][1] - wpos[1])
            if dw < wmin:
                wmin, wmin_fr = dw, j
        if wmin != float("inf") and wmin / h > wrist_prox_max:
            continue  # ball never reached the racket wrist → phantom swing, not a hit
        # CONTACT FRAME = the argmin of the ball↔wrist search (the frame where the
        # ball actually reaches the racket), falling back to the ball↔torso argmin
        # when no wrist was readable in the window. This was `contact_frame = j`
        # before — the leftover loop variable, i.e. the LAST frame of the window
        # (~+0.2s after the peak, itself already lagging contact): every hit
        # candidate carried a systematic ~10-15 frame LATE bias. That bias is why
        # placed hits shadowed the true contacts (pred @487 for the GT hit @468,
        # pred @65 for a swing whose ball-closest frame is ~15f earlier) — the
        # events DP then had to choose between a mistimed candidate and none.
        contact_frame = wmin_fr if wmin_fr is not None else best_fr
        ball_pos = (ball_centers[contact_frame]
                    if ball_centers[contact_frame] is not None else best_j)
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
        # skip serve. far_aware: PLAYER-relative (wrist above the player's OWN head,
        # ball at/above the box top) — the frame-fraction test flagged every
        # legitimately-high far contact as a serve and dropped all far hits. Legacy
        # (default): the original frame-fraction test, kept byte-identical for the
        # arbitration path.
        p = players[best]
        wy = wrist_xy[i][1][1] if (wrist_xy[i] is not None and wrist_xy[i] is not False) else (p["box"][1])
        if far_aware:
            box_top = p["box"][1]
            ph = max(_player_height(p), 1.0)
            if wy < box_top - 0.05 * ph and ball_pos[1] < box_top + 0.10 * ph:
                continue
        else:
            if wy < 0.30 * frame_height and ball_pos[1] < 0.35 * frame_height:
                continue
        hits.append({
            "frame": contact_frame, "x": int(ball_pos[0]), "y": int(ball_pos[1]),
            "player_idx": best, "player_side": side, "anchor": best_anchor,
            "right_handed": handed[side], "wrist_speed": wspeed,
        })
        _mark(side, contact_frame)
    return hits


# ── FH/BH classification thresholds (all in player-HEIGHT units → scale-free) ──
# These are geometric bands, NOT pixel constants: dx/swing are divided by the
# player's box height h, so an "axis-ambiguous" zone of 0.06·h means "the ball was
# within 6% of a body-length of the body axis" — the same physical band on a 260px
# near player and a 100px far player. Provisional values from the demo3 GT (8
# shots, mono-clip); kept conservative and revalidated the moment a 2nd GT exists.
_FHB_ABSTAIN = 0.06   # |ball_dx| under this ⇒ ambiguous (on the body axis) ⇒ unknown
_FHB_DISAGREE = 0.05  # ball_dx & swing must both clear this for the cross-body abstain


def _swing_meandx(players_per_frame_window, frame, player_side, net_y, fps):
    """Pre-contact swing direction of the ACTIVE wrist relative to the body anchor,
    averaged over a short window BEFORE the contact (screen frame, /height).

    The contact instant only tells you where the ball is; the SWING tells you which
    side the stroke came from — the discriminator for a cross-body contact (a
    backhand whose ball-at-contact reads on the forehand side because the ball has
    already left the strings). Over the window [frame-w, frame-1] we collect BOTH
    wrists' (x-anchor)/h on the side-player, then return the mean of the side whose
    wrist travelled the most (the "active" / racket wrist by greatest path — the
    doc's selector). This is handedness-AGNOSTIC, which matters on the far player:
    the right-wrist-only track was the noisy signal that wrongly flipped f141; the
    active wrist resolves it. None if the window has no usable pose.

    Window = round(0.12*fps) frames (~6 @50fps) — a swing's pre-contact arc; fps-
    scaled, no per-video constant. The doc measured [-6,-1] @50fps = 8/8; we express
    it as a ratio of fps and take the strictly-pre-contact half (post-contact adds
    follow-through noise).
    """
    w = max(3, int(round(fps * 0.12)))
    n = len(players_per_frame_window)
    # per-wrist series of (relative-x, absolute-xy) so we can pick the active one
    series = {L_WRIST: [], R_WRIST: []}
    last_xy = {L_WRIST: None, R_WRIST: None}
    path = {L_WRIST: 0.0, R_WRIST: 0.0}
    for j in range(max(0, frame - w), frame):
        slot = players_per_frame_window[j] if j < n else None
        if not slot:
            continue
        # the side-player (tallest on this side, matching detect_hits' anchor pick)
        me = None
        for p in slot:
            if _side_of(p, net_y) != player_side:
                continue
            if me is None or _player_height(p) > _player_height(me):
                me = p
        if me is None:
            continue
        ax, _ = _player_anchor(me)
        h = max(_player_height(me), 1.0)
        for wi in (L_WRIST, R_WRIST):
            wr = _kp(me, wi)
            if wr is None:
                continue
            series[wi].append((wr[0] - ax) / h)
            if last_xy[wi] is not None:
                path[wi] += float(np.hypot(wr[0] - last_xy[wi][0],
                                           wr[1] - last_xy[wi][1]))
            last_xy[wi] = wr
    # the active wrist = greater total path; fall back to whichever has samples
    active = None
    if series[L_WRIST] and series[R_WRIST]:
        active = R_WRIST if path[R_WRIST] >= path[L_WRIST] else L_WRIST
    elif series[L_WRIST]:
        active = L_WRIST
    elif series[R_WRIST]:
        active = R_WRIST
    if active is None or not series[active]:
        return None
    return float(np.mean(series[active]))


def _ball_dx_at(hit, players_per_frame):
    """(ball_x - anchor_x)/h for a hit's contact player, or None if unreadable.
    The screen-side of the ball at contact — the primary FH/BH signal AND the basis
    of the handedness vote (forehands dominate ⇒ the mode of this sign is the
    player's forehand side)."""
    i = hit["frame"]
    players = players_per_frame[i] if i < len(players_per_frame) else None
    if not players or hit.get("player_idx", 0) >= len(players):
        return None
    p = players[hit["player_idx"]]
    ax, ay = _player_anchor(p)
    h = max(_player_height(p), 1.0)
    return (hit["x"] - ax) / h


# Handedness prior: flip to left-handed only on a CLEAR ball-side mode (a strict
# super-majority), else keep the right-handed default. A 1-vote margin on a handful
# of shots is noise (the doc proved mono-clip handedness is unstable), so the bar
# to override the population default is deliberately high. Scale-free: it's a vote
# RATIO, not a count.
_HAND_MIN_VOTES = 4       # need at least this many confident shots to even consider
_HAND_SUPERMAJ = 0.75     # and ≥75% on one side to flip off the right-handed default


def estimate_handedness_prior(hits, players_per_frame, fps, frame_height):
    """Per-side handedness PRIOR from the MODE of the ball-side at contact.

    The doc's key finding: per-frame facing AND per-clip wrist-variance are both
    unreliable for handedness on a single rally. The robust population prior: a
    player hits MORE forehands than backhands, so the MODE of the ball-at-contact
    side over their shots = their forehand side, and the forehand side fixes
    handedness — a right-hander's forehand sits on screen-RIGHT (ball_dx>0) on the
    near end. We vote per court SIDE (near/far) because screen-direction inverts
    between the two ends, and each end's mapping is self-consistent on its own
    frames.

    Returns {'near': hand, 'far': hand} with hand ∈ {True (right), False (left),
    None}. CRITICAL: None means "use the right-handed default" (NOT abstain the
    shot) — ~80%+ of players are right-handed, so the safe fallback is right, not a
    refusal. We only return False (left-handed) on a strict super-majority of a
    decent sample (≥_HAND_MIN_VOTES shots, ≥_HAND_SUPERMAJ on the minority/forehand
    side opposite to right) — otherwise mono-clip noise would wrongly flip a
    right-hander (the bug that took the composite to 1/7). This is a PRIOR, not a
    per-shot signal; it only sets the screen→absolute mapping.
    """
    side_votes = {"near": [], "far": []}
    for h in hits:
        side = h.get("player_side")
        if side not in ("near", "far"):
            continue
        bdx = _ball_dx_at(h, players_per_frame)
        if bdx is None or abs(bdx) < _FHB_ABSTAIN:
            continue  # axis-ambiguous shots don't vote
        side_votes[side].append(1 if bdx > 0 else -1)
    out = {}
    for side in ("near", "far"):
        votes = side_votes[side]
        out[side] = None  # default → right-handed mapping downstream
        if len(votes) < _HAND_MIN_VOTES:
            continue
        pos = sum(1 for v in votes if v > 0)
        frac_pos = pos / len(votes)
        # forehand side = the screen side a strict majority of shots land on.
        # right-handed (forehand on screen-right) is the default, so only a strong
        # LEFT lean (forehand on screen-left ⇒ left-handed) overrides it.
        if frac_pos >= _HAND_SUPERMAJ:
            out[side] = True          # forehand on the right ⇒ right-handed
        elif (1 - frac_pos) >= _HAND_SUPERMAJ:
            out[side] = False         # forehand on the left ⇒ left-handed
        # else: ambiguous lean ⇒ keep the right-handed default (None)
    return out


def classify_forehand_backhand(hit, players_per_frame, right_handed=None,
                               players_per_frame_window=None, handedness=None,
                               fps=50.0, frame_height=None):
    """Geometric FH/BH at a hit. Returns 'forehand' | 'backhand' | 'unknown'.

    COMPOSITE (informed by docs/research/forehand-backhand.md §3 + LIVE measurement):
      1. PRIMARY = ball-at-contact side: ``ball_dx = (ball_x - anchor_x)/h`` (screen
         frame, /player-height). Measured 7/8 alone — the lone miss is a cross-body
         contact, which we ABSTAIN on rather than guess (see 2).
      2. CROSS-BODY GUARD = pre-contact swing → ABSTAIN (not override). When the
         swing direction confidently disagrees with the contact side, the call is
         too ambiguous to make, so we return 'unknown'. The doc's §3 OVERRIDE-to-swing
         got 8/8 on the cache but introduced a far-court CONFUSION live (the small far
         player's swing is noise; trusting it flips a correct forehand to backhand,
         and no geometric signal separates a true near cross-body from a false far
         flicker). Abstaining on disagreement holds confusion at 0 on BOTH the cache
         and live — the priority — at the cost of abstaining on the rare cross-body.
      3. ABSTENTION near the body axis (|ball_dx| < band) → 'unknown'.
      4. HANDEDNESS as a per-side PRIOR (estimate_handedness_prior), NOT the old
         per-frame facing correction (which inverted the class ~6/8 → 2/8). The
         prior maps screen-side → forehand/backhand; default right-handed when the
         vote margin is too low.

    The old per-frame ``facing`` multiplier is GONE — it was the bug (removing it is
    the measured 2/8 → 7/8 quick-win). No wrist-proximity gate either: detect_hits
    already validated the contact, and a redundant gate here abstained on far shots
    whose ball detect_hits snapped slightly off the tiny far player.

    Args:
        hit: {frame, x (ball x), player_idx, player_side, ...}.
        players_per_frame: the per-frame pose list ``hit['player_idx']`` indexes
            (the SAME list detect_hits used).
        players_per_frame_window: per-frame pose list for the swing window (usually
            the same as players_per_frame). If None, the cross-body guard is skipped.
        handedness: optional {'near': bool|None, 'far': bool|None} prior from
            estimate_handedness_prior. None / a None side ⇒ right-handed default.
        right_handed: legacy/no-op (kept for signature compatibility).
    """
    i = hit["frame"]
    players = players_per_frame[i] if i < len(players_per_frame) else None
    if not players or hit["player_idx"] >= len(players):
        return "unknown"
    p = players[hit["player_idx"]]
    ax, ay = _player_anchor(p)
    bx = hit["x"]
    h = max(_player_height(p), 1.0)
    side = hit.get("player_side")
    if side not in ("near", "far"):
        side = _side_of(p, frame_height * 0.47) if frame_height else "near"

    # a wrist must exist at all (shoulders give us nothing now that facing is gone)
    lw = _kp(p, L_WRIST)
    rw = _kp(p, R_WRIST)
    if lw is None and rw is None:
        return "unknown"

    # (1) primary: ball side at contact (screen frame, /height)
    ball_dx = (bx - ax) / h
    if abs(ball_dx) < _FHB_ABSTAIN:
        return "unknown"  # ball on the body axis → ambiguous, refuse honestly

    # (2) cross-body GUARD via the pre-contact swing → ABSTAIN, don't override.
    # A genuine cross-body backhand has the swing come from the OTHER side than the
    # ball-at-contact (the ball already left the strings). The original spec OVERRODE
    # to the swing here; measurement killed that: the swing is reliable on the near
    # player but NOISY on the small far player (no geometric separator — a true near
    # cross-body and a false far swing-flicker are indistinguishable by magnitude or
    # sign-consistency). Trusting it flips a correct strong far forehand to backhand
    # (a CONFUSION — the worst outcome). So on a confident disagreement we ABSTAIN
    # (honest 'unknown') rather than override: that converts both the true cross-body
    # (cache f525) and the false far flicker (live f149) to abstentions, holding
    # confusion at 0 either way — the prompt's priority. The contact owns the call
    # everywhere else.
    if players_per_frame_window is not None:
        net_y = (frame_height * 0.47) if frame_height else (ay + 1.0)
        sw = _swing_meandx(players_per_frame_window, i, side, net_y, fps)
        if (sw is not None and abs(sw) > _FHB_DISAGREE
                and abs(ball_dx) > _FHB_DISAGREE
                and (ball_dx > 0) != (sw > 0)):
            return "unknown"  # swing fights the contact → too ambiguous to call

    base_side = 1 if ball_dx > 0 else -1

    # NOTE: no wrist-proximity gate here. detect_hits already validated this is a real
    # contact (its own torso + wrist-prox gates); a redundant gate here only ABSTAINED
    # on far contacts whose ball position detect_hits snapped a little off the small
    # far player (the live far-abstention bug). Contact validation is detect_hits' job.

    # (3) screen-side → absolute FH/BH via the handedness prior for this court side.
    # right-hander: forehand on screen-RIGHT (base_side>0) ⇒ forehand. left-hander
    # mirrors. None / missing prior ⇒ right-handed default (>80% of players).
    hand = None
    if handedness is not None:
        hand = handedness.get(side)
    is_right = True if hand is None else bool(hand)
    fh_is_right_side = is_right  # forehand sits on the player's racket-hand side
    return "forehand" if (base_side > 0) == fh_is_right_side else "backhand"


def classify_shots(hits, players_per_frame, fps, frame_height,
                   players_per_frame_window=None):
    """Batch FH/BH for all detected hits: estimate the per-side handedness prior
    ONCE from the full hit set, then classify each hit with the composite. Returns
    a list of labels aligned with ``hits``. This is the orchestrator the pipeline
    calls so the handedness prior sees every shot (the more hits, the better the
    vote margin). ``players_per_frame_window`` defaults to ``players_per_frame``.
    """
    win = players_per_frame_window if players_per_frame_window is not None else players_per_frame
    # vote the prior on the SAME list player_idx indexes (the contact list), not the
    # window — the ball_dx that decides the forehand-side must read the contact pose.
    handed = estimate_handedness_prior(hits, players_per_frame, fps, frame_height)
    return [
        classify_forehand_backhand(
            h, players_per_frame, players_per_frame_window=win,
            handedness=handed, fps=fps, frame_height=frame_height)
        for h in hits
    ], handed


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
