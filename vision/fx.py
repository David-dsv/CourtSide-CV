"""
Cinematic visual-effects library for the CourtSide-CV tennis pipeline.

A self-contained, GPU-free rendering toolkit that gives the annotated output a
modern sports-broadcast look (think SwingVision / Hawk-Eye / TV graphics) instead
of flat OpenCV primitives. Pure ``cv2`` + ``numpy`` + ``PIL`` — no extra deps, no
GPU, importable and unit-runnable without a video.

Aesthetic: neon-on-dark. Glowing elements are composited additively (bloom),
panels are frosted glass (glassmorphism), text is crisp TrueType (not Hershey),
and impacts/markers pulse with light. All colors are **BGR** to match the rest of
the pipeline (cv2 convention).

Design rules (see CLAUDE.md "zero-hardcoding"):
  - Spatial magnitudes that should scale with the video are expressed relative to
    frame size (``frame.shape``). UI chrome (panel padding, font sizes) is allowed
    fixed pixels but is scaled from frame height where it matters for legibility.
  - Every glowing primitive routes through :func:`add_bloom`, the foundation.

Color grading helpers operate on a green→yellow→red ramp for "good→fast/hot".

Public API
----------
    add_bloom(frame, glow_layer, sigmas, intensity)
    comet_trail(frame, points, base_color, max_thickness, glow)
    glass_panel(frame, x, y, w, h, tint, alpha, border, radius)
    draw_text(frame, text, org, size, color, font, shadow, anchor)
    shockwave(frame, center, age_frames, max_age, color, max_radius)
    speed_gauge(frame, x, y, value_kmh, max_kmh, label, radius)
    stat_card(frame, x, y, lines, title, width)
    rally_card(frame, x, y, *, rally_index, lines, outcome, width, title)
    placement_heatmap(frame, points, frame_w, frame_h, alpha, sigma, colormap)
    glow_marker(frame, center, color, radius, pulse)
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # scipy is in requirements; degrade gracefully if missing at import time
    from scipy.stats import gaussian_kde
    _HAS_KDE = True
except Exception:  # pragma: no cover
    _HAS_KDE = False


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
# macOS system fonts (verified to exist on the target machine). We keep a
# regular + bold path and fall back to PIL's bundled default if absent so the
# module never hard-crashes on another OS.
_FONT_PATHS = {
    "regular": "/System/Library/Fonts/Supplemental/Arial.ttf",
    "bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
}
# Cache loaded ImageFont objects by (path, size) — truetype loading is not free.
_FONT_CACHE: dict[tuple[str, int], "ImageFont.FreeTypeFont"] = {}


def _get_font(font: str, size: int) -> "ImageFont.ImageFont":
    """Return a cached PIL font for ``font`` ('regular'|'bold'|explicit path) at
    ``size`` px. Falls back to PIL's default bitmap font if the file is missing."""
    path = _FONT_PATHS.get(font, font)
    key = (path, int(size))
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        if os.path.exists(path):
            f = ImageFont.truetype(path, int(size))
        else:  # try as a font-name lookup, else default
            f = ImageFont.truetype(path, int(size))
    except Exception:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


# ---------------------------------------------------------------------------
# Small color / math helpers
# ---------------------------------------------------------------------------
def _clip_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _scale_color(color: Sequence[int], factor: float) -> tuple[int, int, int]:
    """Scale a BGR color brightness by ``factor`` (clamped to 0..255)."""
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


def speed_color(p: float) -> tuple[int, int, int]:
    """Green→yellow→red ramp for a normalized value ``p`` in [0,1] (BGR).

    Used by gauges and heat indicators: low = calm green, high = hot red.
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p < 0.5:  # green -> yellow
        t = p / 0.5
        b, g, r = 60 * (1 - t), 220, 40 + (215 - 40) * t
    else:  # yellow -> red
        t = (p - 0.5) / 0.5
        b, g, r = 30, 220 * (1 - t) + 40 * t, 255
    return (int(b), int(g), int(r))


def _rounded_rect_mask(w: int, h: int, radius: int) -> np.ndarray:
    """Single-channel uint8 mask (0/255) of a rounded rectangle of size w×h."""
    radius = int(max(0, min(radius, min(w, h) // 2)))
    mask = np.zeros((h, w), np.uint8)
    if radius == 0:
        mask[:] = 255
        return mask
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


# ---------------------------------------------------------------------------
# 1. Bloom — the foundation primitive
# ---------------------------------------------------------------------------
def add_bloom(frame: np.ndarray, glow_layer: np.ndarray,
              sigmas: Sequence[int] = (5, 11, 21),
              intensity: float = 1.0) -> np.ndarray:
    """Additively bloom the bright parts of ``glow_layer`` onto ``frame``.

    The classic real-time bloom recipe: bright-pass the scratch layer, then
    **downscale to 0.25** before blurring at each sigma (blurring a quarter-size
    image is ~16× cheaper and indistinguishable once upscaled), upscale back, and
    additively stack every blur onto the frame. This is the primitive every other
    glowing effect routes through.

    Parameters
    ----------
    frame : HxWx3 uint8 BGR — composited in place-ish (a new array is returned).
    glow_layer : HxWx3 uint8 BGR — black canvas with bright shapes drawn on it.
    sigmas : Gaussian sigmas (in *full-res* px) for the multi-scale halo. Larger
        sigmas give a wider, softer glow.
    intensity : overall multiplier on the added light.

    Returns
    -------
    HxWx3 uint8 BGR composited frame.
    """
    if glow_layer is None or not glow_layer.any():
        return frame
    h, w = frame.shape[:2]

    # Bright-pass: keep only the luminous pixels so dark scratch areas don't wash
    # the frame. Soft threshold via a smooth knee.
    gl = glow_layer.astype(np.float32)
    lum = gl.max(axis=2, keepdims=True)
    knee = 40.0
    mask = np.clip((lum - knee) / (255.0 - knee), 0.0, 1.0)
    bright = gl * mask

    small = cv2.resize(bright, (max(1, w // 4), max(1, h // 4)),
                       interpolation=cv2.INTER_LINEAR)

    accum = np.zeros_like(small)
    for s in sigmas:
        ks = max(3, int(s) | 1)  # odd kernel; sigma is at quarter res so /4-ish
        blurred = cv2.GaussianBlur(small, (ks, ks), sigmaX=max(1.0, s / 4.0))
        accum += blurred
    if len(sigmas):
        accum /= len(sigmas)

    glow_full = cv2.resize(accum, (w, h), interpolation=cv2.INTER_LINEAR)
    out = frame.astype(np.float32) + glow_full * float(intensity)
    return _clip_u8(out)


def _bloom_overlay(frame: np.ndarray, glow_layer: np.ndarray,
                   sigmas: Sequence[int] = (5, 11, 21),
                   intensity: float = 1.0, crisp: float = 1.0) -> np.ndarray:
    """Composite a glow layer as **crisp shape + bloom halo**.

    :func:`add_bloom` alone only adds a blurred halo — the sharp shape never
    lands on the frame, so thin neon lines (arcs, trails, rings) come out as a
    dim smudge. This helper first stamps the crisp ``glow_layer`` onto the frame
    (additive, weighted by ``crisp``) and *then* blooms it, so the result is a
    sharp luminous line with a soft halo around it — the look every glow effect
    here wants.
    """
    if glow_layer is None or not glow_layer.any():
        return frame
    base = _clip_u8(frame.astype(np.float32) + glow_layer.astype(np.float32) * crisp)
    out = add_bloom(base, glow_layer, sigmas=sigmas, intensity=intensity)
    frame[:] = out  # write back in place so the effect lands even if the return is dropped
    return frame


# ---------------------------------------------------------------------------
# 2. Comet trail
# ---------------------------------------------------------------------------
def comet_trail(frame: np.ndarray, points: Sequence[Sequence[float]],
                base_color: Sequence[int] = (0, 200, 255),
                max_thickness: int = 6, glow: bool = True) -> np.ndarray:
    """Draw a tapering, alpha-decaying comet trail through ``points``.

    The head (last point) is bright and thick; the tail fades to thin and dim,
    like a comet. Optionally bloomed so the head burns.

    Parameters
    ----------
    points : ordered list of (x, y) recent positions, **head last**.
    base_color : BGR of the head.
    max_thickness : head line thickness in px (tapers to 1 at the tail).
    glow : if True, draw to a scratch layer and route through :func:`add_bloom`.
    """
    pts = [(int(p[0]), int(p[1])) for p in points if p is not None]
    if len(pts) < 2:
        return frame
    n = len(pts)
    glow_layer = np.zeros_like(frame) if glow else None
    target = glow_layer if glow else frame

    # Draw oldest→newest so brighter head segments paint over the tail.
    for i in range(1, n):
        t = i / (n - 1)  # 0 at tail .. 1 at head
        thick = max(1, int(round(max_thickness * (0.25 + 0.75 * t))))
        # brightness eases up toward the head; tail keeps a faint ember
        bright = 0.22 + 0.78 * (t ** 1.4)
        col = _scale_color(base_color, bright)
        cv2.line(target, pts[i - 1], pts[i], col, thick, cv2.LINE_AA)

    # A hot core dot at the head for the "burning" tip.
    cv2.circle(target, pts[-1], max(2, max_thickness // 2 + 1),
               _scale_color(base_color, 1.0), -1, cv2.LINE_AA)
    cv2.circle(target, pts[-1], max(1, max_thickness // 3),
               (255, 255, 255), -1, cv2.LINE_AA)  # white-hot center

    if glow:
        # Crisp trail on the frame + soft halo, so the comet reads sharply.
        return _bloom_overlay(frame, glow_layer, sigmas=(7, 15, 27),
                              intensity=0.9, crisp=1.0)
    return frame


# ---------------------------------------------------------------------------
# 3. Glassmorphism panel
# ---------------------------------------------------------------------------
def glass_panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
                tint: Sequence[int] = (28, 20, 20), alpha: float = 0.45,
                border: Sequence[int] = (255, 255, 255), radius: int = 16) -> np.ndarray:
    """Render a frosted-glass (glassmorphism) panel in place.

    The ROI under the panel is blurred (frosted), a dark tint is blended over it,
    and a soft rounded-rect border is drawn. Rounded corners are masked so the
    panel reads as a single floating pane. The caller draws text on top
    afterwards.

    Note: ``tint`` is BGR; the prompt's default (20,20,28) is a cool near-black —
    here expressed BGR-correctly as a dark slate.
    """
    H, W = frame.shape[:2]
    x = int(np.clip(x, 0, W - 1)); y = int(np.clip(y, 0, H - 1))
    w = int(max(2, min(w, W - x))); h = int(max(2, min(h, H - y)))
    radius = int(min(radius, min(w, h) // 2))

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return frame

    # 1) Frost: heavy blur of what's behind the panel.
    k = max(3, (min(w, h) // 6) | 1)
    k = min(k, 99)
    frosted = cv2.GaussianBlur(roi, (k, k), sigmaX=15)

    # 2) Dark tint + subtle top-to-bottom gradient for depth.
    tint_arr = np.empty_like(frosted, dtype=np.float32)
    tint_arr[:] = np.array(tint, dtype=np.float32)
    grad = np.linspace(1.12, 0.86, h, dtype=np.float32)[:, None, None]
    tint_arr *= grad
    panel = frosted.astype(np.float32) * (1 - alpha) + tint_arr * alpha

    # 3) A faint inner top highlight (glass sheen).
    sheen_h = max(2, h // 6)
    sheen = np.linspace(40, 0, sheen_h, dtype=np.float32)[:, None, None]
    panel[:sheen_h] += sheen

    panel = _clip_u8(panel)

    # 4) Rounded-corner mask: only paint inside the rounded rect.
    mask = _rounded_rect_mask(w, h, radius)
    m3 = (mask[..., None].astype(np.float32) / 255.0)
    blended = roi.astype(np.float32) * (1 - m3) + panel.astype(np.float32) * m3
    frame[y:y + h, x:x + w] = _clip_u8(blended)

    # 5) Soft rounded border (~30% alpha) drawn on a scratch + blended.
    if border is not None:
        bcanvas = frame[y:y + h, x:x + w].copy()
        _draw_rounded_border(bcanvas, 0, 0, w, h, radius, border, thickness=1)
        be = (mask[..., None].astype(np.float32) / 255.0)
        frame[y:y + h, x:x + w] = _clip_u8(
            frame[y:y + h, x:x + w].astype(np.float32) * (1 - 0.30 * be)
            + bcanvas.astype(np.float32) * (0.30 * be)
            + frame[y:y + h, x:x + w].astype(np.float32) * 0.0
        )
        # Re-overlay border cleanly at low alpha (the line itself).
        line_layer = frame[y:y + h, x:x + w].copy()
        _draw_rounded_border(line_layer, 0, 0, w, h, radius, border, thickness=1)
        frame[y:y + h, x:x + w] = cv2.addWeighted(
            frame[y:y + h, x:x + w], 0.7, line_layer, 0.3, 0)
    return frame


def _draw_rounded_border(img, x, y, w, h, radius, color, thickness=1):
    """Draw a rounded-rectangle outline (anti-aliased)."""
    color = tuple(int(c) for c in color)
    r = int(radius)
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h - 1), (x + w - r, y + h - 1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w - 1, y + r), (x + w - 1, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# 4. Crisp TrueType text
# ---------------------------------------------------------------------------
def draw_text(frame: np.ndarray, text: str, org: Sequence[int],
              size: int = 22, color: Sequence[int] = (255, 255, 255),
              font: str = "bold", shadow: bool = True,
              anchor: str = "lt") -> np.ndarray:
    """Draw crisp anti-aliased TrueType text (PIL), not Hershey.

    Parameters
    ----------
    org : reference point (x, y). Interpreted with ``anchor``.
    size : font size in px.
    color : BGR.
    font : 'bold' | 'regular' | explicit truetype path.
    shadow : draw a 2px black drop-shadow for legibility on busy frames.
    anchor : horizontal alignment of ``org`` — 'lt' (left/top, default),
        'center' (center/top), 'rt' (right/top).

    Returns the frame (modified in place via ROI swap).
    """
    if text is None or text == "":
        return frame
    H, W = frame.shape[:2]
    pil_font = _get_font(font, size)

    # Measure text to size a tight ROI (efficient: convert only what we touch).
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        l, t, r, b = tmp.textbbox((0, 0), text, font=pil_font)
        tw, th = r - l, b - t
        ox, oy = l, t
    except Exception:
        tw, th = pil_font.getsize(text) if hasattr(pil_font, "getsize") else (len(text) * size // 2, size)
        ox, oy = 0, 0

    x, y = int(org[0]), int(org[1])
    if anchor == "center":
        x -= tw // 2
    elif anchor == "rt":
        x -= tw

    pad = 4  # room for shadow + AA
    rx0 = max(0, x - pad); ry0 = max(0, y - pad)
    rx1 = min(W, x + tw + pad + 2); ry1 = min(H, y + th + pad + 2)
    if rx1 <= rx0 or ry1 <= ry0:
        return frame

    roi = frame[ry0:ry1, rx0:rx1]
    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Position inside the ROI, compensating the glyph's own bbox offset.
    px, py = x - rx0 - ox, y - ry0 - oy
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    if shadow:
        draw.text((px + 2, py + 2), text, font=pil_font, fill=(0, 0, 0))
    draw.text((px, py), text, font=pil_font, fill=rgb)

    frame[ry0:ry1, rx0:rx1] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def text_size(text: str, size: int, font: str = "bold") -> tuple[int, int]:
    """Return (width, height) in px of ``text`` at ``size`` — for layout."""
    pil_font = _get_font(font, size)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    try:
        l, t, r, b = tmp.textbbox((0, 0), text, font=pil_font)
        return (r - l, b - t)
    except Exception:
        return (len(text) * size // 2, size)


# ---------------------------------------------------------------------------
# 5. Shockwave / impact ring
# ---------------------------------------------------------------------------
def shockwave(frame: np.ndarray, center: Sequence[int], age_frames: int,
              max_age: int, color: Sequence[int] = (0, 220, 255),
              max_radius: int = 70) -> np.ndarray:
    """An expanding, eased impact ring + soft radial flash that fades with age.

    Dresses up a bounce/contact. The ring radius eases out (fast then settling),
    its thickness and alpha decay over the effect lifetime, and an early filled
    flash adds a punch. The ring is bloomed so it glows.

    Parameters
    ----------
    center : (x, y) impact point.
    age_frames : how many frames since the impact (0 = just happened).
    max_age : lifetime in frames; the effect is fully gone at ``age >= max_age``.
    color : BGR.
    max_radius : final ring radius in px.
    """
    if age_frames >= max_age or age_frames < 0:
        return frame
    p = age_frames / float(max_age)            # 0..1 progress
    eased = 1 - (1 - p) ** 3                    # ease-out
    radius = max(1, int(max_radius * eased))
    fade = (1 - p)                              # global fade-out

    cx, cy = int(center[0]), int(center[1])
    glow_layer = np.zeros_like(frame)

    # Early radial flash (filled, soft) — strongest at birth, gone by ~40% life.
    if p < 0.45:
        flash_a = (1 - p / 0.45)
        flash = np.zeros_like(frame)
        cv2.circle(flash, (cx, cy), int(max_radius * 0.55),
                   _scale_color(color, 1.0), -1, cv2.LINE_AA)
        kf = max(3, int(max_radius * 0.6) | 1)
        flash = cv2.GaussianBlur(flash, (kf, kf), sigmaX=max_radius * 0.25)
        frame[:] = _clip_u8(frame.astype(np.float32) + flash.astype(np.float32) * 0.45 * flash_a)

    # Expanding ring on the glow layer (thickness decays with age).
    thick = max(1, int(round(4 * (1 - p) + 1)))
    ring_col = _scale_color(color, 0.6 + 0.4 * fade)
    cv2.circle(glow_layer, (cx, cy), radius, ring_col, thick, cv2.LINE_AA)
    # A second, fainter trailing ring for a richer wave.
    if radius > 8:
        cv2.circle(glow_layer, (cx, cy), int(radius * 0.72),
                   _scale_color(color, 0.35 * fade), max(1, thick - 1), cv2.LINE_AA)

    # Crisp ring on the frame + bloom halo.
    out = _bloom_overlay(frame, glow_layer, sigmas=(5, 13, 23),
                         intensity=0.8 * fade + 0.25, crisp=0.9)
    return out


# ---------------------------------------------------------------------------
# 6. Broadcast speed gauge
# ---------------------------------------------------------------------------
def speed_gauge(frame: np.ndarray, x: int, y: int, value_kmh: float,
                max_kmh: float = 200, label: str = "BALL",
                radius: int = 46) -> np.ndarray:
    """A circular broadcast-style speed readout.

    A 270°-sweep arc fills proportionally to ``value/max`` and is color-graded
    green→yellow→red. The number sits big in the center, the label under it, all
    on a glassmorphism backing.

    Parameters
    ----------
    x, y : top-left of the gauge's bounding box.
    value_kmh : the speed to display.
    max_kmh : the value at which the arc is full.
    radius : arc radius in px (panel sizes from this).
    """
    frame_orig = frame  # rebinding `frame` below must still mutate the caller's buffer
    pad = int(radius * 0.42)
    box = 2 * radius + 2 * pad
    # Glass backing.
    glass_panel(frame, x, y, box, box, tint=(30, 22, 18), alpha=0.5,
                border=(180, 180, 200), radius=int(radius * 0.45))

    cx, cy = x + box // 2, y + box // 2
    r = radius
    start_ang = 135           # sweep from 135° clockwise 270°
    sweep = 270
    p = float(np.clip(value_kmh / max_kmh, 0.0, 1.0))

    arc_th = max(4, r // 6)   # bold, broadcast-weight arc

    # Track (dim background arc) — drawn directly on the frame.
    cv2.ellipse(frame, (cx, cy), (r, r), 0, start_ang, start_ang + sweep,
                (62, 58, 70), arc_th, cv2.LINE_AA)

    # Filled value arc on a glow layer (color-graded green→yellow→red).
    glow_layer = np.zeros_like(frame)
    n_seg = max(2, int(sweep * p / 4))
    for i in range(n_seg):
        a0 = start_ang + (sweep * p) * (i / n_seg)
        a1 = start_ang + (sweep * p) * ((i + 1) / n_seg)
        seg_p = (i / max(1, n_seg - 1)) * p  # color follows position along arc
        col = speed_color(seg_p if p else 0)
        cv2.ellipse(glow_layer, (cx, cy), (r, r), 0, a0, a1 + 1.5,
                    col, arc_th, cv2.LINE_AA)
    # End-cap dot (white-hot leading edge of the fill).
    if p > 0:
        end_ang = math.radians(start_ang + sweep * p)
        ex, ey = int(cx + r * math.cos(end_ang)), int(cy + r * math.sin(end_ang))
        cv2.circle(glow_layer, (ex, ey), max(3, arc_th // 2 + 1),
                   (255, 255, 255), -1, cv2.LINE_AA)
    # Crisp arc on the frame + halo bloom.
    frame = _bloom_overlay(frame, glow_layer, sigmas=(5, 11, 19),
                           intensity=0.7, crisp=1.0)

    # Big number + units + label.
    num = f"{int(round(value_kmh))}"
    num_size = int(r * 0.95)
    draw_text(frame, num, (cx, cy - int(r * 0.42)), size=num_size,
              color=(255, 255, 255), font="bold", anchor="center")
    draw_text(frame, "km/h", (cx, cy + int(r * 0.18)), size=int(r * 0.30),
              color=(190, 200, 210), font="regular", anchor="center")
    draw_text(frame, label.upper(), (cx, cy + int(r * 0.52)), size=int(r * 0.32),
              color=speed_color(p), font="bold", anchor="center")
    if frame is not frame_orig:
        frame_orig[:] = frame  # honor the in-place contract even if caller drops the return
    return frame_orig


# ---------------------------------------------------------------------------
# 7. Stat card
# ---------------------------------------------------------------------------
def stat_card(frame: np.ndarray, x: int, y: int,
              lines: Sequence[Sequence[str]], title: str | None = None,
              width: int = 200) -> np.ndarray:
    """A glassmorphism stat card: optional title + rows of (label, value).

    Clean typography; values are right-aligned against the panel edge — for
    FH/BH counts, rally length, shot quality, etc.

    Parameters
    ----------
    lines : iterable of (label, value) string pairs.
    title : optional header (drawn in an accent color with an underline rule).
    width : panel width in px (height is derived from the row count).
    """
    lines = list(lines)
    fh = frame.shape[0]
    # Type scale derived from frame height for legibility on any resolution.
    title_sz = max(14, int(fh * 0.022))
    row_sz = max(12, int(fh * 0.019))
    pad = max(12, int(width * 0.07))
    row_h = int(row_sz * 1.55)

    header_h = (title_sz + int(row_sz * 0.9)) if title else 0
    h = pad + header_h + row_h * len(lines) + pad
    glass_panel(frame, x, y, width, h, tint=(34, 26, 22), alpha=0.5,
                border=(170, 175, 195), radius=14)

    cy = y + pad
    accent = (255, 190, 90)  # warm cyan/amber accent (BGR)
    if title:
        draw_text(frame, title.upper(), (x + pad, cy), size=title_sz,
                  color=accent, font="bold")
        # underline rule
        ry = cy + title_sz + int(row_sz * 0.25)
        cv2.line(frame, (x + pad, ry), (x + width - pad, ry),
                 (110, 115, 135), 1, cv2.LINE_AA)
        cy += header_h

    for label, value in lines:
        draw_text(frame, str(label), (x + pad, cy), size=row_sz,
                  color=(210, 215, 225), font="regular")
        draw_text(frame, str(value), (x + width - pad, cy), size=row_sz,
                  color=(255, 255, 255), font="bold", anchor="rt")
        cy += row_h
    return frame


# ---------------------------------------------------------------------------
# 7b. Rally card (per-rally HUD with "ÉCHANGE N" badge + outcome pill)
# ---------------------------------------------------------------------------
# Outcome → BGR color mapping (cv2 convention). Keys are the canonical rally
# outcome labels the pipeline emits; unknown / None outcomes draw nothing.
#   winner         → green   (a clean point won)
#   forced_error   → orange  (opponent forced into the error — earned, low-ish)
#   unforced_error → red     (point lost on own mistake)
#   neutral        → gray    (rally still open / no decisive outcome)
_OUTCOME_COLORS: dict[str, tuple[int, int, int]] = {
    "winner": (90, 210, 90),          # green
    "forced_error": (60, 165, 245),   # orange
    "unforced_error": (60, 60, 235),  # red
    "neutral": (150, 150, 150),       # gray
}
# Human-readable labels shown on the pill (kept short; uppercased at draw time).
_OUTCOME_LABELS: dict[str, str] = {
    "winner": "WINNER",
    "forced_error": "FORCED ERR",
    "unforced_error": "UNFORCED ERR",
    "neutral": "EN COURS",
}


def rally_card(frame: np.ndarray, x: int, y: int, *,
               rally_index, lines: Sequence[Sequence[str]],
               outcome: str | None = None,
               width: int | None = None, title: str = "ÉCHANGE") -> np.ndarray:
    """Glassmorphism card for the **current rally**'s live stats (bottom-left HUD).

    Visually a richer sibling of :func:`stat_card`, purpose-built for per-rally
    telemetry that resets every point: a prominent ``"{title} {rally_index}"``
    badge header (e.g. ``"ÉCHANGE 3"``), a body of (label, value) stat rows, and
    an optional colored outcome pill (winner / forced_error / …).

    Parameters
    ----------
    rally_index : int | str — the current rally number, rendered after ``title``
        in the header badge. Stringified, so anything printable works.
    lines : iterable of (label, value) string pairs — the rally's stats
        (e.g. ``[("Frappes", "4"), ("Rebonds", "3"), ("Peak", "98 km/h")]``).
        May be empty (header-only card).
    outcome : str | None — one of ``_OUTCOME_COLORS``'s keys. When given, a
        colored pill with the outcome label is drawn under the stats. Unknown or
        ``None`` → no pill (the card simply omits it).
    width : panel width in px. Defaults to a frame-height-derived width (same
        spirit as :func:`stat_card`) so it reads on any resolution.
    title : header prefix; defaults to ``"ÉCHANGE"``.

    Returns
    -------
    The frame, annotated in place (same convention as :func:`stat_card`).
    """
    lines = list(lines)
    fh = frame.shape[0]
    # Type scale derived from frame height — legible on any resolution.
    badge_sz = max(17, int(fh * 0.027))   # header is bigger/bolder than rows
    row_sz = max(12, int(fh * 0.019))
    pill_sz = max(12, int(fh * 0.018))

    # Width: derive from frame height when not given, with a floor wide enough
    # for a large rally_index + the longest outcome label to never clip.
    if width is None:
        width = max(190, int(fh * 0.26))
    width = int(width)
    pad = max(12, int(width * 0.07))
    row_h = int(row_sz * 1.55)

    # ── Vertical layout budget ──────────────────────────────────────────────
    # header badge block, then an underline rule, then the stat rows, then the
    # optional outcome pill. Each segment contributes a fixed, derived height so
    # the panel size is stable regardless of rally_index magnitude.
    header_h = badge_sz + int(row_sz * 0.95)
    rows_h = row_h * len(lines)
    pill_h = int(pill_sz * 2.1) + int(row_sz * 0.4) if outcome in _OUTCOME_COLORS else 0
    h = pad + header_h + rows_h + pill_h + pad

    # Glass backing — slightly cooler tint than stat_card so the two cards read
    # as a family but the rally card stands out as "live".
    glass_panel(frame, x, y, width, h, tint=(38, 28, 22), alpha=0.52,
                border=(180, 185, 205), radius=14)

    cy = y + pad

    # ── Header badge: "ÉCHANGE N" ───────────────────────────────────────────
    # A small accent tick to the left of the title gives it a "live indicator"
    # feel, then the big bold header text.
    accent = (245, 195, 95)  # warm amber accent (BGR)
    tick_w = max(3, int(width * 0.018))
    tick_h = badge_sz
    cv2.rectangle(frame, (x + pad, cy + int(badge_sz * 0.08)),
                  (x + pad + tick_w, cy + int(badge_sz * 0.08) + tick_h),
                  accent, -1, cv2.LINE_AA)
    header_text = f"{title} {rally_index}"
    draw_text(frame, header_text, (x + pad + tick_w + max(6, pad // 2), cy),
              size=badge_sz, color=(255, 255, 255), font="bold")

    # Underline rule under the header.
    ry = cy + badge_sz + int(row_sz * 0.30)
    cv2.line(frame, (x + pad, ry), (x + width - pad, ry),
             (115, 120, 145), 1, cv2.LINE_AA)
    cy += header_h

    # ── Stat rows ───────────────────────────────────────────────────────────
    for label, value in lines:
        draw_text(frame, str(label), (x + pad, cy), size=row_sz,
                  color=(210, 215, 225), font="regular")
        draw_text(frame, str(value), (x + width - pad, cy), size=row_sz,
                  color=(255, 255, 255), font="bold", anchor="rt")
        cy += row_h

    # ── Outcome pill ────────────────────────────────────────────────────────
    if outcome in _OUTCOME_COLORS:
        col = _OUTCOME_COLORS[outcome]
        label = _OUTCOME_LABELS.get(outcome, str(outcome).upper())
        cy += int(row_sz * 0.4)
        pill_top = cy
        pill_bottom = cy + int(pill_sz * 1.9)
        pill_r = (pill_bottom - pill_top) // 2
        # Rounded pill: a filled rounded-rect, translucent so it sits on glass.
        pill_x0, pill_x1 = x + pad, x + width - pad
        _filled_rounded_rect(frame, pill_x0, pill_top, pill_x1, pill_bottom,
                             pill_r, col, alpha=0.85)
        # A bright dot on the left of the pill, then the outcome label.
        dot_cx = pill_x0 + pill_r
        dot_cy = (pill_top + pill_bottom) // 2
        cv2.circle(frame, (dot_cx, dot_cy), max(2, pill_r // 3),
                   (255, 255, 255), -1, cv2.LINE_AA)
        # Choose text color for contrast against the pill fill.
        txt_col = (255, 255, 255)
        draw_text(frame, label, (dot_cx + pill_r, pill_top + int(pill_sz * 0.18)),
                  size=pill_sz, color=txt_col, font="bold")
    return frame


def _filled_rounded_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                         radius: int, color: Sequence[int],
                         alpha: float = 1.0) -> None:
    """Alpha-blend a filled rounded rectangle (BGR ``color``) onto ``frame``.

    Used by :func:`rally_card`'s outcome pill. Builds a rounded-rect mask, paints
    a solid color layer, and blends it over the existing ROI at ``alpha`` so the
    pill reads as a translucent chip on the glass card (not a flat sticker).
    """
    H, W = frame.shape[:2]
    x0 = int(np.clip(x0, 0, W - 1)); x1 = int(np.clip(x1, 0, W))
    y0 = int(np.clip(y0, 0, H - 1)); y1 = int(np.clip(y1, 0, H))
    w, h = x1 - x0, y1 - y0
    if w <= 1 or h <= 1:
        return
    mask = _rounded_rect_mask(w, h, int(radius))
    roi = frame[y0:y1, x0:x1]
    color_layer = np.empty_like(roi, dtype=np.float32)
    color_layer[:] = np.array(color, dtype=np.float32)
    m3 = (mask[..., None].astype(np.float32) / 255.0) * float(np.clip(alpha, 0, 1))
    frame[y0:y1, x0:x1] = _clip_u8(
        roi.astype(np.float32) * (1 - m3) + color_layer * m3)


# ---------------------------------------------------------------------------
# 8. Placement heatmap
# ---------------------------------------------------------------------------
def placement_heatmap(frame: np.ndarray, points: Sequence[Sequence[float]],
                      frame_w: int, frame_h: int, alpha: float = 0.55,
                      sigma: float = 40, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Kernel-density heatmap of impact / bounce locations (INFERNO).

    Builds a continuous density field from ``points`` (gaussian KDE when ≥3
    points are available, else gaussian-blurred impulses), normalizes it, color-
    maps it with INFERNO, and alpha-blends it onto the frame **only where the
    density is meaningful** so the court underneath stays readable. This is the
    headline "exploitable data" visual.

    Parameters
    ----------
    points : (x, y) impact locations in frame pixels.
    frame_w, frame_h : frame dims (the density grid spans these).
    sigma : spread of each impact's influence in px.
    """
    pts = [(float(p[0]), float(p[1])) for p in points if p is not None]
    if not pts:
        return frame

    density = np.zeros((frame_h, frame_w), np.float32)
    if _HAS_KDE and len(pts) >= 3:
        try:
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            # Evaluate KDE on a coarse grid then upscale (fast, smooth).
            gw, gh = max(32, frame_w // 8), max(32, frame_h // 8)
            xx, yy = np.meshgrid(np.linspace(0, frame_w, gw),
                                 np.linspace(0, frame_h, gh))
            # Bandwidth scaled so each impact spreads ~sigma px → smooth field,
            # not confetti. Floor keeps sparse-point maps from over-tightening.
            bw = max(0.18, sigma / max(frame_w, frame_h) * 6)
            kde = gaussian_kde(np.vstack([xs, ys]), bw_method=bw)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(gh, gw)
            density = cv2.resize(zz.astype(np.float32), (frame_w, frame_h),
                                 interpolation=cv2.INTER_CUBIC)
            density = np.clip(density, 0, None)
        except Exception:
            density = np.zeros((frame_h, frame_w), np.float32)

    if density.max() <= 0:  # fallback: blurred impulses
        for (px, py) in pts:
            ix, iy = int(np.clip(px, 0, frame_w - 1)), int(np.clip(py, 0, frame_h - 1))
            cv2.circle(density, (ix, iy), 1, 1.0, -1)
        k = int(sigma) | 1
        density = cv2.GaussianBlur(density, (k, k), sigmaX=sigma)

    if density.max() <= 0:
        return frame
    norm = density / density.max()
    heat_u8 = _clip_u8(norm * 255)
    colored = cv2.applyColorMap(heat_u8, colormap)

    # Blend only where density is meaningful — smooth alpha ramp from a floor.
    floor = 0.12
    a = np.clip((norm - floor) / (1 - floor), 0, 1) * alpha
    a3 = a[..., None].astype(np.float32)
    out = frame.astype(np.float32) * (1 - a3) + colored.astype(np.float32) * a3

    # A subtle bright core at the hottest cells for a "data" feel.
    hot = (norm > 0.78).astype(np.uint8) * 255
    if hot.any():
        glow = np.zeros_like(frame)
        glow[hot > 0] = (90, 150, 255)
        out = add_bloom(_clip_u8(out), glow, sigmas=(7, 15), intensity=0.5)
    frame[:] = _clip_u8(out)  # in-place so the heatmap lands even if return is dropped
    return frame


# ---------------------------------------------------------------------------
# 9. Glowing marker
# ---------------------------------------------------------------------------
def glow_marker(frame: np.ndarray, center: Sequence[int],
                color: Sequence[int] = (0, 220, 255), radius: int = 8,
                pulse: float = 0.0) -> np.ndarray:
    """A glowing point marker (ball / minimap dot) with optional pulse.

    The drawn radius breathes as ``radius * (1 + 0.3*sin(pulse))`` so callers can
    animate it by passing an increasing phase. Routed through bloom so it emits
    light; a white-hot core keeps it crisp.

    Parameters
    ----------
    center : (x, y).
    color : BGR halo color.
    radius : base radius in px.
    pulse : animation phase (radians). 0 disables the breathing.
    """
    cx, cy = int(center[0]), int(center[1])
    r = max(1, int(round(radius * (1 + 0.3 * math.sin(pulse)))))
    glow_layer = np.zeros_like(frame)
    # Soft outer halo + saturated mid + white core.
    cv2.circle(glow_layer, (cx, cy), int(r * 1.6), _scale_color(color, 0.55), -1, cv2.LINE_AA)
    cv2.circle(glow_layer, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(glow_layer, (cx, cy), max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)
    out = _bloom_overlay(frame, glow_layer, sigmas=(5, 11, 19),
                         intensity=1.0, crisp=1.0)
    # Re-stamp a crisp saturated body + white core on top of the bloom.
    cv2.circle(out, (cx, cy), r, color, -1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)
    return out


__all__ = [
    "add_bloom", "comet_trail", "glass_panel", "draw_text", "text_size",
    "shockwave", "speed_gauge", "stat_card", "rally_card", "placement_heatmap",
    "glow_marker", "speed_color",
]
