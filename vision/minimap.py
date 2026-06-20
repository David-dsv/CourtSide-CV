"""
2D top-down court minimap with the ball trajectory and bounce locations.

Draws a schematic ITF court (singles + doubles lines, service boxes, net) in a
corner of the frame, and projects ball positions + bounces onto it.

Image→court projection:
  - PRIMARY: when a court homography H (image px → court meters) is available
    (broadcast angle), project exactly via H.
  - FALLBACK: when H is unavailable (oblique/amateur, court not localized), use a
    geometric approximation from the ball's position in the frame — the minimap is
    then indicative, not metric (flagged with a '~' in the title).

Court frame (meters): origin at court center, +X toward right doubles sideline
(half-width 5.485 m), +Y toward the far baseline (half-length 11.885 m). This
matches models.court_detector.WORLD_KEYPOINTS.
"""
import numpy as np
import cv2

# ITF dimensions (meters)
DOUBLES_HALF_W = 5.485
SINGLES_HALF_W = 4.115
HALF_LEN = 11.885
SERVICE_FROM_NET = 6.40

# minimap pixel layout
PAD_M = 1.2          # meters of margin around the court in the minimap
TRAIL_COLOR = (90, 255, 150)
BALL_COLOR = (0, 255, 80)
LINE_COLOR = (245, 245, 245)
COURT_FILL = (150, 95, 55)      # BGR — solid bluish court
BOUNCE_COLORS = {"deep": (0, 0, 255), "mid": (0, 200, 255), "short": (0, 230, 120)}


def _court_to_minimap(Xm, Ym, w, h):
    """Court meters → minimap pixel. +Y (far baseline) maps to the TOP of the
    minimap, +X (right) maps to the right."""
    span_x = (DOUBLES_HALF_W + PAD_M)
    span_y = (HALF_LEN + PAD_M)
    px = w / 2 + (Xm / span_x) * (w / 2)
    py = h / 2 - (Ym / span_y) * (h / 2)
    return int(round(px)), int(round(py))


def render_court_base(w, h):
    """Return a BGR minimap image with the court lines drawn (no ball yet)."""
    img = np.full((h, w, 3), 35, np.uint8)            # dark backdrop
    # court fill
    tl = _court_to_minimap(-DOUBLES_HALF_W, +HALF_LEN, w, h)
    br = _court_to_minimap(+DOUBLES_HALF_W, -HALF_LEN, w, h)
    cv2.rectangle(img, tl, br, COURT_FILL, -1)

    def line(p1, p2, thick=1):
        a = _court_to_minimap(*p1, w, h)
        b = _court_to_minimap(*p2, w, h)
        cv2.line(img, a, b, LINE_COLOR, thick, cv2.LINE_AA)

    # doubles box
    line((-DOUBLES_HALF_W, +HALF_LEN), (+DOUBLES_HALF_W, +HALF_LEN))
    line((-DOUBLES_HALF_W, -HALF_LEN), (+DOUBLES_HALF_W, -HALF_LEN))
    line((-DOUBLES_HALF_W, +HALF_LEN), (-DOUBLES_HALF_W, -HALF_LEN))
    line((+DOUBLES_HALF_W, +HALF_LEN), (+DOUBLES_HALF_W, -HALF_LEN))
    # singles sidelines
    line((-SINGLES_HALF_W, +HALF_LEN), (-SINGLES_HALF_W, -HALF_LEN))
    line((+SINGLES_HALF_W, +HALF_LEN), (+SINGLES_HALF_W, -HALF_LEN))
    # service lines
    line((-SINGLES_HALF_W, +SERVICE_FROM_NET), (+SINGLES_HALF_W, +SERVICE_FROM_NET))
    line((-SINGLES_HALF_W, -SERVICE_FROM_NET), (+SINGLES_HALF_W, -SERVICE_FROM_NET))
    # center service line
    line((0, +SERVICE_FROM_NET), (0, -SERVICE_FROM_NET))
    # net (thicker)
    line((-DOUBLES_HALF_W, 0), (+DOUBLES_HALF_W, 0), thick=2)
    return img


def project_to_court(x, y, frame_w, frame_h, homography=None):
    """Image pixel (x,y) → court meters (Xm, Ym). Uses homography if available,
    else a geometric fallback. Returns (Xm, Ym) or None if off-court (clamped)."""
    if homography is not None and homography.get("H") is not None:
        from models.court_detector import img_to_world
        Xm, Ym = img_to_world(homography["H"], x, y)
        return Xm, Ym
    # ── geometric fallback (no homography) ──
    # Assume the visible court roughly spans a vertical band of the frame. Map the
    # ball's vertical position to court depth (top of band = far baseline) and its
    # horizontal position to court width, with a mild perspective correction
    # (objects higher in the frame = farther = compressed horizontally).
    yr = y / frame_h
    # court occupies ~ rows 0.30..0.82 of the frame on a typical mid/low angle
    top, bot = 0.30, 0.84
    depth = (yr - top) / (bot - top)                # 0 = far baseline, 1 = near
    depth = max(0.0, min(1.0, depth))
    Ym = HALF_LEN - depth * (2 * HALF_LEN)          # +HALF_LEN (far) → -HALF_LEN (near)
    # horizontal: center the frame, scale by a perspective factor (wider near cam)
    xr = x / frame_w - 0.5
    persp = 0.55 + 0.9 * depth                       # near rows span more width
    Xm = (xr / 0.5) * DOUBLES_HALF_W * persp
    Xm = max(-DOUBLES_HALF_W - PAD_M, min(DOUBLES_HALF_W + PAD_M, Xm))
    return Xm, Ym


def draw_minimap(frame, ball_trail_img, bounces_img, frame_w, frame_h,
                 homography=None, scale=0.26, margin=18):
    """Composite a minimap onto the bottom-right of `frame` (in place-ish; returns
    the frame).
      ball_trail_img: list of recent (x,y) image points (the live trail)
      bounces_img: list of (x, y, depth) image points (all bounces so far)
    """
    # minimap size proportional to frame
    mh = int(frame_h * scale)
    mw = int(mh * ((DOUBLES_HALF_W + PAD_M) / (HALF_LEN + PAD_M)) * 1.0)
    mw = max(mw, int(frame_w * 0.12))
    base = render_court_base(mw, mh)

    # draw trail
    pts = []
    for (x, y) in ball_trail_img:
        proj = project_to_court(x, y, frame_w, frame_h, homography)
        if proj is not None:
            pts.append(_court_to_minimap(proj[0], proj[1], mw, mh))
    for i in range(1, len(pts)):
        a = i / max(1, len(pts))
        col = tuple(int(c * (0.4 + 0.6 * a)) for c in TRAIL_COLOR)
        cv2.line(base, pts[i - 1], pts[i], col, 3, cv2.LINE_AA)
    if pts:
        cv2.circle(base, pts[-1], 5, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(base, pts[-1], 5, (255, 255, 255), 1, cv2.LINE_AA)

    # draw bounces
    for (x, y, depth) in bounces_img:
        proj = project_to_court(x, y, frame_w, frame_h, homography)
        if proj is None:
            continue
        mp = _court_to_minimap(proj[0], proj[1], mw, mh)
        col = BOUNCE_COLORS.get(depth, (255, 255, 255))
        cv2.circle(base, mp, 5, col, -1, cv2.LINE_AA)
        cv2.circle(base, mp, 5, (255, 255, 255), 1, cv2.LINE_AA)

    # title (flag fallback)
    exact = homography is not None and homography.get("H") is not None
    title = "Court map" if exact else "Court map ~"
    cv2.putText(base, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    # composite bottom-right: mostly opaque so the court reads clearly
    x0 = frame_w - mw - margin
    y0 = frame_h - mh - margin
    roi = frame[y0:y0 + mh, x0:x0 + mw]
    blended = cv2.addWeighted(base, 0.94, roi, 0.06, 0)
    frame[y0:y0 + mh, x0:x0 + mw] = blended
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + mw + 1, y0 + mh + 1), (255, 255, 255), 2)
    return frame
