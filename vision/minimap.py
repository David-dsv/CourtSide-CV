"""
Premium 2D top-down "COURT RADAR" — a broadcast-quality animated minimap.

Draws a schematic ITF court (singles + doubles lines, service boxes, net) inside a
glassmorphism panel in a corner of the frame, and projects the ball trajectory,
bounce locations (color-coded by depth, with a soft placement heatmap) and the two
players onto it. The look targets Hawk-Eye / SwingVision radar overlays: a cool-blue
court with a subtle gradient, glowing comet-style ball trail, pulsing live ball, and
crisp TrueType chrome — all via the cinematic ``vision.fx`` library.

Image→court projection:
  - PRIMARY: when a court homography H (image px → court meters) is available
    (a one-time calibration, or a broadcast keypoint solve), project exactly via H.
    The points then spread across the FULL court depth (metric-honest).
  - FALLBACK: when H is unavailable (oblique/amateur, court not localized), use a
    geometric approximation from the ball's position in the frame — the minimap is
    then indicative, not metric (flagged with a '~' in the title).

Court frame (meters): origin at court center, +X toward right doubles sideline
(half-width 5.485 m), +Y toward the far baseline (half-length 11.885 m). This
matches models.court_detector.WORLD_KEYPOINTS.
"""
import numpy as np
import cv2

from vision import fx

# ITF dimensions (meters)
DOUBLES_HALF_W = 5.485
SINGLES_HALF_W = 4.115
HALF_LEN = 11.885
SERVICE_FROM_NET = 6.40

# meters of margin around the court inside the radar viewport (keeps points just
# behind a baseline on-screen without wasting much real estate)
PAD_M = 0.8

# ── palette (BGR) ──────────────────────────────────────────────────────────
COURT_DEEP = (96, 58, 26)        # cool deep blue (far / top)
COURT_NEAR = (140, 92, 44)       # lighter blue (near / bottom) → subtle gradient
COURT_FILL = (118, 74, 34)       # mean court tone
LINE_COLOR = (235, 240, 245)     # near-white court lines
LINE_SOFT = (200, 215, 225)      # inner lines slightly dimmer
NET_COLOR = (245, 250, 255)
TRAIL_COLOR = (120, 235, 90)     # green comet
BALL_COLOR = (90, 255, 180)      # bright yellow-green live ball
ACCENT = (255, 200, 110)         # cyan/amber accent for chrome (BGR)

# bounce colors by depth (BGR): deep=red, mid=amber, short=green
BOUNCE_COLORS = {
    "deep":  (60, 60, 255),
    "mid":   (40, 190, 255),
    "short": (90, 230, 120),
}

# default player colors (BGR) — match run_pipeline_8s.PLAYER_COLORS
PLAYER_COLORS = {1: (0, 191, 255), 2: (255, 160, 50)}

# supersampling factor — render the court tile at SS× then downscale for crisp AA
_SS = 2


def _court_to_tile(Xm, Ym, w, h):
    """Court meters → tile pixel. +Y (far baseline) maps to the TOP, +X to the
    right. Returns float (sub-pixel) coords for smooth AA drawing."""
    span_x = (DOUBLES_HALF_W + PAD_M)
    span_y = (HALF_LEN + PAD_M)
    px = w / 2.0 + (Xm / span_x) * (w / 2.0)
    py = h / 2.0 - (Ym / span_y) * (h / 2.0)
    return px, py


def _ipt(p):
    """Float point → int tuple for cv2 (uses sub-pixel shift-free rounding)."""
    return (int(round(p[0])), int(round(p[1])))


def render_court_base(w, h):
    """Return a BGR court tile (court fill + lines), with a cool-blue gradient.

    No glow here — pure court geometry. Trajectory/markers/glow are layered on top
    by :func:`draw_minimap`. Lines are anti-aliased; the caller renders this at a
    supersampled size and downscales for extra crispness.
    """
    img = np.empty((h, w, 3), np.float32)

    # vertical gradient backdrop (cool, dark) so the panel reads as a screen
    top_bg = np.array((46, 30, 18), np.float32)
    bot_bg = np.array((60, 40, 24), np.float32)
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    img[:] = top_bg * (1 - ramp) + bot_bg * ramp

    # court rectangle (doubles) with a top→bottom gradient (far darker, near lighter)
    tl = _court_to_tile(-DOUBLES_HALF_W, +HALF_LEN, w, h)
    br = _court_to_tile(+DOUBLES_HALF_W, -HALF_LEN, w, h)
    cx0, cy0 = int(round(tl[0])), int(round(tl[1]))
    cx1, cy1 = int(round(br[0])), int(round(br[1]))
    cx0, cx1 = sorted((cx0, cx1))
    cy0, cy1 = sorted((cy0, cy1))
    ch = max(1, cy1 - cy0)
    cramp = np.linspace(0.0, 1.0, ch, dtype=np.float32)[:, None, None]
    court_block = (np.array(COURT_DEEP, np.float32) * (1 - cramp)
                   + np.array(COURT_NEAR, np.float32) * cramp)
    img[cy0:cy1, cx0:cx1] = court_block

    img = np.clip(img, 0, 255).astype(np.uint8)

    th = max(1, int(round(w / 130)))      # base line thickness (scales with tile)
    th_thin = max(1, th)
    th_net = max(2, th + 1)

    def line(p1, p2, color=LINE_COLOR, thick=th_thin):
        a = _ipt(_court_to_tile(*p1, w, h))
        b = _ipt(_court_to_tile(*p2, w, h))
        cv2.line(img, a, b, color, thick, cv2.LINE_AA)

    # doubles box
    line((-DOUBLES_HALF_W, +HALF_LEN), (+DOUBLES_HALF_W, +HALF_LEN))
    line((-DOUBLES_HALF_W, -HALF_LEN), (+DOUBLES_HALF_W, -HALF_LEN))
    line((-DOUBLES_HALF_W, +HALF_LEN), (-DOUBLES_HALF_W, -HALF_LEN))
    line((+DOUBLES_HALF_W, +HALF_LEN), (+DOUBLES_HALF_W, -HALF_LEN))
    # singles sidelines
    line((-SINGLES_HALF_W, +HALF_LEN), (-SINGLES_HALF_W, -HALF_LEN), LINE_SOFT)
    line((+SINGLES_HALF_W, +HALF_LEN), (+SINGLES_HALF_W, -HALF_LEN), LINE_SOFT)
    # service lines
    line((-SINGLES_HALF_W, +SERVICE_FROM_NET), (+SINGLES_HALF_W, +SERVICE_FROM_NET), LINE_SOFT)
    line((-SINGLES_HALF_W, -SERVICE_FROM_NET), (+SINGLES_HALF_W, -SERVICE_FROM_NET), LINE_SOFT)
    # center service line
    line((0, +SERVICE_FROM_NET), (0, -SERVICE_FROM_NET), LINE_SOFT)
    # center marks on the baselines (small ticks)
    line((0, +HALF_LEN), (0, +HALF_LEN - 0.4), LINE_SOFT)
    line((0, -HALF_LEN), (0, -HALF_LEN + 0.4), LINE_SOFT)

    # net: a soft dark band (the net's shadow) + a bright crisp line over it, so
    # the divider reads instantly. Drawn slightly wider than the court so the
    # posts overhang the doubles sidelines like a real net.
    nl = _ipt(_court_to_tile(-DOUBLES_HALF_W - 0.25, 0, w, h))
    nr = _ipt(_court_to_tile(+DOUBLES_HALF_W + 0.25, 0, w, h))
    band = img.copy()
    cv2.line(band, nl, nr, (20, 14, 8), max(3, th_net * 4), cv2.LINE_AA)
    band = cv2.GaussianBlur(band, (1, (max(3, th_net * 4) | 1)), 0)
    img[:] = cv2.addWeighted(img, 0.55, band, 0.45, 0)
    cv2.line(img, nl, nr, NET_COLOR, th_net, cv2.LINE_AA)
    # net posts (small ticks at the ends)
    cv2.circle(img, nl, max(2, th_net), NET_COLOR, -1, cv2.LINE_AA)
    cv2.circle(img, nr, max(2, th_net), NET_COLOR, -1, cv2.LINE_AA)
    return img


def project_to_court(x, y, frame_w, frame_h, homography=None):
    """Image pixel (x,y) → court meters (Xm, Ym), or None if unprojectable.

    Uses the homography ONLY when it is trustworthy AND the point lies in front of
    the homography's horizon (on a grazing oblique angle the horizon line cuts
    THROUGH the court — points behind it project to infinity / flip to the wrong
    side, which collapsed everything onto one half of the radar). When the
    homography is low-confidence or the point is past the horizon, fall back to a
    geometric image-Y → court-depth map, calibrated on the real baseline pixel
    rows when a calibration is available (far more accurate than a fixed band)."""
    H = homography.get("H") if homography is not None else None
    conf = homography.get("confidence") if homography is not None else None
    if H is not None and conf in ("high", "medium"):
        # horizon guard: w = H[2]·[x,y,1] must be safely positive (same side as the
        # control points). Near/behind the horizon the projection is unstable.
        w = float(H[2, 0] * x + H[2, 1] * y + H[2, 2])
        if w > 1e-3:
            from models.court_detector import img_to_world
            Xm, Ym = img_to_world(H, x, y)
            if abs(Ym) <= HALF_LEN + 2.0 and abs(Xm) <= DOUBLES_HALF_W + 2.0:
                return Xm, Ym
        # else fall through to the geometric map (don't trust a flipped point)

    # ── geometric fallback ──
    # Map image-Y to court depth. Anchor on the calibration's known baseline rows
    # when available (e.g. felix: far baseline ≈ y0, near baseline ≈ y1) so the
    # spread matches the real court; else use a generic broadcast band.
    top, bot = _depth_band(homography, frame_h)
    yr = y / frame_h
    depth = (yr - top) / (bot - top + 1e-9)          # 0 = far baseline, 1 = near
    depth = max(0.0, min(1.0, depth))
    Ym = HALF_LEN - depth * (2 * HALF_LEN)           # +HALF_LEN (far) → -HALF_LEN (near)
    xr = x / frame_w - 0.5
    persp = 0.55 + 0.9 * depth                        # near rows span more width
    Xm = (xr / 0.5) * DOUBLES_HALF_W * persp
    Xm = max(-DOUBLES_HALF_W - PAD_M, min(DOUBLES_HALF_W + PAD_M, Xm))
    return Xm, Ym


def _depth_band(homography, frame_h):
    """Return (top_ratio, bot_ratio): the image-Y rows (as fractions of frame
    height) of the FAR and NEAR baselines, derived from calibration landmark
    pixels when present, else a generic broadcast band."""
    if homography is not None and homography.get("landmark_px"):
        lp = homography["landmark_px"]
        far_ys = [lp[k][1] for k in lp if k.startswith("far_")]
        near_ys = [lp[k][1] for k in lp if k.startswith("near_")]
        if far_ys and near_ys:
            top = min(far_ys) / frame_h
            bot = max(near_ys) / frame_h
            if bot - top > 0.05:
                return top, bot
    return 0.30, 0.84


def _on_radar(Xm, Ym):
    """True if a court-meter point falls within the radar viewport (court + margin
    + a little slack so points just behind the baseline still render at the edge)."""
    return (abs(Xm) <= DOUBLES_HALF_W + PAD_M + 2.5 and
            abs(Ym) <= HALF_LEN + PAD_M + 3.0)


def _smooth_arc(tile_pts, per_seg=10):
    """Catmull-Rom upsample an ordered list of (px,py) tile points into a smooth
    flowing arc (Hawk-Eye 'shot spot' look). With <3 points there's nothing to
    spline, so return the points unchanged. ``per_seg`` is the number of
    interpolated samples emitted per original segment."""
    pts = [(float(p[0]), float(p[1])) for p in tile_pts if p is not None]
    if len(pts) < 3:
        return pts
    P = np.asarray(pts, np.float32)
    # pad the ends by reflecting so the first/last segments are also splined
    ext = np.vstack([P[0] + (P[0] - P[1]), P, P[-1] + (P[-1] - P[-2])])
    out = []
    n = len(P)
    for i in range(n - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for j in range(per_seg):
            t = j / float(per_seg)
            t2, t3 = t * t, t * t * t
            # Catmull-Rom basis (uniform)
            q = 0.5 * ((2 * p1)
                       + (-p0 + p2) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                       + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((float(q[0]), float(q[1])))
    out.append((float(P[-1][0]), float(P[-1][1])))   # ensure the head is exact
    return out


def _draw_heat(tile, pts_tile, colors):
    """Soft additive placement-heat blobs at bounce tile points (cheap, glanceable).
    pts_tile: list of (px,py); colors: matching list of BGR. Modifies tile in place."""
    if not pts_tile:
        return
    h, w = tile.shape[:2]
    glow = np.zeros_like(tile)
    r = max(4, int(w / 14))
    for (p, col) in zip(pts_tile, colors):
        cv2.circle(glow, _ipt(p), r, tuple(int(c) for c in col), -1, cv2.LINE_AA)
    k = (r | 1)
    glow = cv2.GaussianBlur(glow, (k, k), sigmaX=r * 0.55)
    tile[:] = np.clip(tile.astype(np.float32) + glow.astype(np.float32) * 0.40, 0, 255).astype(np.uint8)


def draw_minimap(frame, ball_trail_img, bounces_img, frame_w, frame_h,
                 homography=None, players_img=None, scale=0.30, margin=None,
                 pulse=0.0, ground_path=None):
    """Composite the premium COURT RADAR onto the bottom-right of ``frame``.

    Parameters
    ----------
    frame : HxWx3 uint8 BGR — modified in place (and returned).
    ball_trail_img : list of recent (x, y) image points (the live ball trail).
        Used ONLY as the legacy fallback when ``ground_path`` is None — a ball in
        flight sits ABOVE its ground point in the image, so projecting in-flight
        positions mirrors image→2D instead of translating 3D→2D (the ball appears
        to recede on the radar during the descending arc). Prefer ``ground_path``.
    bounces_img : list of (x, y, depth) image points (all bounces so far);
        ``depth`` is one of 'deep' | 'mid' | 'short'.
    frame_w, frame_h : frame dimensions.
    homography : court homography dict {H, ...} (image px → court meters), or None.
    players_img : OPTIONAL list of (x, y, player_id) feet image points; drawn as
        colored dots (PLAYER_COLORS). Backward-compatible — omit to skip.
    scale : radar height as a fraction of the frame height.
    margin : panel margin from the frame edge in px (default: ~2.2% of height).
    pulse : animation phase (radians) for the live-ball breathing + ring.
    ground_path : OPTIONAL ordered list of (x_img, y_img, kind) GROUND-level points
        (kind in {'bounce','hit'}) in temporal order. When given, the radar plots a
        smoothed arc through these metrically-trustworthy ground contacts instead of
        the in-flight comet — the only points that translate honestly to a top-down
        map. The live-ball dot then sits at the most recent ground point. Backward-
        compatible: omit (or pass None) to keep the legacy ``ball_trail_img`` comet.
    """
    H, Wf = frame.shape[:2]
    if margin is None:
        margin = int(frame_h * 0.022)

    # "exact" (green dot, no ~) only when a homography is present AND it isn't
    # flagged low-confidence (a grazing-oblique calibration is metric but its
    # far-field depth is approximate — flag it honestly with ~).
    exact = (homography is not None and homography.get("H") is not None
             and homography.get("confidence") != "low")

    # ── panel geometry (glassmorphism container) ──────────────────────────────
    # court aspect inside the viewport
    span_x = DOUBLES_HALF_W + PAD_M
    span_y = HALF_LEN + PAD_M
    court_ar = span_x / span_y                          # width / height

    pad = max(8, int(frame_h * 0.012))                  # inner padding inside glass
    title_h = max(16, int(frame_h * 0.030))             # space for the title row
    legend_h = max(12, int(frame_h * 0.024))            # space for the legend row

    # radar tile drawing area (the court), height-driven
    tile_h = int(frame_h * scale)
    tile_w = int(round(tile_h * court_ar))
    panel_w = tile_w + 2 * pad
    panel_h = tile_h + title_h + legend_h + 2 * pad
    panel_w = min(panel_w, Wf - 2 * margin)
    panel_h = min(panel_h, H - 2 * margin)

    px0 = Wf - panel_w - margin
    py0 = H - panel_h - margin

    # frosted-glass backing pane
    fx.glass_panel(frame, px0, py0, panel_w, panel_h,
                   tint=(40, 26, 16), alpha=0.52,
                   border=(200, 175, 130), radius=max(10, int(panel_h * 0.06)))

    # ── render the court radar onto a supersampled tile ───────────────────────
    tw, th = tile_w * _SS, tile_h * _SS
    tile = render_court_base(tw, th)

    def to_tile(Xm, Ym):
        return _court_to_tile(Xm, Ym, tw, th)

    # project helper (image → court meters → tile px), with on-radar gating
    def proj_tile(x, y):
        Xm, Ym = project_to_court(x, y, frame_w, frame_h, homography)
        if not _on_radar(Xm, Ym):
            return None
        # clamp to viewport so near-baseline points sit on the edge, not off-tile
        Xm = max(-span_x, min(span_x, Xm))
        Ym = max(-span_y, min(span_y, Ym))
        return to_tile(Xm, Ym)

    # 1) bounce placement heat layer (subtle), under everything
    bounce_tpts, bounce_cols, bounce_depths = [], [], []
    for b in bounces_img:
        x, y, depth = b[0], b[1], b[2]
        p = proj_tile(x, y)
        if p is None:
            continue
        bounce_tpts.append(p)
        bounce_cols.append(BOUNCE_COLORS.get(depth, (255, 255, 255)))
        bounce_depths.append(depth)
    if len(bounce_tpts) >= 2:
        _draw_heat(tile, bounce_tpts, bounce_cols)

    # 2) ball trajectory — a GROUND-PATH arc when ground_path is provided (bounces +
    #    hit/contact points, temporally ordered, smoothed into Hawk-Eye 'shot spot'
    #    arcs), else the legacy in-flight comet (kept for backward compatibility).
    trail_pts = []
    if ground_path is not None:
        gp_pts = []
        for item in ground_path:
            x, y = item[0], item[1]
            p = proj_tile(x, y)
            if p is not None:
                gp_pts.append(p)
        # smooth the ground contacts into flowing arcs (upsample per segment)
        trail_pts = _smooth_arc(gp_pts, per_seg=10)
    else:
        for (x, y) in ball_trail_img:
            p = proj_tile(x, y)
            if p is not None:
                trail_pts.append(p)
    if len(trail_pts) >= 2:
        fx.comet_trail(tile, trail_pts, base_color=TRAIL_COLOR,
                       max_thickness=max(3, int(tw / 50)), glow=True)

    # 3) bounce markers — glowing depth-colored dots with a white core
    br = max(3, int(tw / 60))
    for (p, col) in zip(bounce_tpts, bounce_cols):
        fx.glow_marker(tile, p, color=col, radius=br, pulse=0.0)

    # 4) players — larger glowing pucks with a white ring + ID, so they're clearly
    #    distinct from bounce dots (optional kwarg; off by default).
    if players_img:
        pr = max(5, int(tw / 34))
        for item in players_img:
            x, y, pid = item[0], item[1], item[2]
            p = proj_tile(x, y)
            if p is None:
                continue
            pid = int(pid)
            col = PLAYER_COLORS.get(pid, (220, 220, 220))
            # soft footprint halo under the puck
            shadow = np.zeros_like(tile)
            cv2.circle(shadow, _ipt(p), int(pr * 2.0), tuple(int(c) for c in col),
                       -1, cv2.LINE_AA)
            shadow = cv2.GaussianBlur(shadow, ((pr | 1), (pr | 1)), 0)
            tile[:] = np.clip(tile.astype(np.float32)
                              + shadow.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
            fx.glow_marker(tile, p, color=col, radius=pr, pulse=0.0)
            # crisp white seating ring
            cv2.circle(tile, _ipt(p), int(pr * 1.6), (255, 255, 255), max(1, _SS), cv2.LINE_AA)
            # player ID tag
            fx.draw_text(tile, f"P{pid}", (int(p[0]), int(p[1] - pr * 2.4)),
                         size=max(10, int(pr * 1.3)), color=(255, 255, 255),
                         font="bold", anchor="center")

    # 5) live ball — pulsing bright dot at the head of the trail
    if trail_pts:
        fx.glow_marker(tile, trail_pts[-1], color=BALL_COLOR,
                       radius=max(4, int(tw / 44)), pulse=pulse)

    # downscale the supersampled tile → crisp anti-aliased radar
    radar = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)

    # ── composite the radar into the panel's content area ─────────────────────
    rx = px0 + pad
    ry = py0 + pad + title_h
    # rounded clip so the radar's corners follow the glass pane
    rmask = fx._rounded_rect_mask(tile_w, tile_h, max(4, int(tile_h * 0.05)))
    m3 = (rmask[..., None].astype(np.float32) / 255.0)
    roi = frame[ry:ry + tile_h, rx:rx + tile_w]
    if roi.shape[:2] == radar.shape[:2]:
        frame[ry:ry + tile_h, rx:rx + tile_w] = np.clip(
            roi.astype(np.float32) * (1 - m3) + radar.astype(np.float32) * m3,
            0, 255).astype(np.uint8)
        # thin inner frame around the radar for a "screen" feel
        cv2.rectangle(frame, (rx - 1, ry - 1), (rx + tile_w, ry + tile_h),
                      (150, 130, 100), 1, cv2.LINE_AA)

    # ── chrome: title + accent rule + legend ──────────────────────────────────
    title = "COURT RADAR" if exact else "COURT RADAR ~"
    t_size = max(13, int(title_h * 0.62))
    fx.draw_text(frame, title, (px0 + pad, py0 + pad), size=t_size,
                 color=(255, 255, 255), font="bold", anchor="lt")
    # a small accent indicator (green = metric, amber = approximate)
    dot_c = (110, 230, 120) if exact else ACCENT
    dcy = py0 + pad + t_size // 2
    cv2.circle(frame, (px0 + panel_w - pad - 4, dcy), max(3, t_size // 5),
               dot_c, -1, cv2.LINE_AA)
    # accent rule under the title
    ruley = py0 + pad + title_h - 3
    cv2.line(frame, (px0 + pad, ruley), (px0 + panel_w - pad, ruley),
             (90, 80, 65), 1, cv2.LINE_AA)

    # legend row (depth color key) along the bottom strip
    ly = py0 + panel_h - pad - int(legend_h * 0.72)
    leg_sz = max(10, int(legend_h * 0.52))
    lx = px0 + pad
    for label, depth in (("DEEP", "deep"), ("MID", "mid"), ("SHORT", "short")):
        col = BOUNCE_COLORS[depth]
        cv2.circle(frame, (lx + leg_sz // 2, ly + leg_sz // 2),
                   max(3, leg_sz // 2), col, -1, cv2.LINE_AA)
        cv2.circle(frame, (lx + leg_sz // 2, ly + leg_sz // 2),
                   max(3, leg_sz // 2), (240, 240, 240), 1, cv2.LINE_AA)
        fx.draw_text(frame, label, (lx + leg_sz + 4, ly - 1),
                     size=leg_sz, color=(225, 230, 235), font="regular")
        lx += leg_sz + 6 + fx.text_size(label, leg_sz, font="regular")[0] + 12

    return frame
