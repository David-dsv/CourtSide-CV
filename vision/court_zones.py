"""Derive court zones from a court homography (image px ↔ world meters).

The HF YOLO court-zone detector (Davidsv/CourtSide-Computer-Vision-v1) suffers a
domain shift on broadcast framings — it returns {} on wide broadcast angles even
though the court-keypoint homography (yastrebksv) solves 14/14 with rms ~3.7px.
The zones are a *redundant fallback*: every consumer (px→m scale, bounce depth,
net row, player-validity band, shot placement, overlay) only needs the
*geometric consequences* of the zones, all of which are derivable from the
homography — usually more precisely, because the projection is perspective-correct.

This module projects the 8 ITF court-zone rectangles (defined in world meters)
through H_inv into image space, producing the same `{name: {"box", ...}}` dict the
rest of the pipeline expects (drop-in compatible), plus a `polygon` field for an
optional perspective overlay.

Zero-hardcoding: the only constants are ITF court dimensions (physical facts).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from models.court_calibrator import world_to_img
from models.court_detector import (
    DOUBLES_HALF_W,
    SINGLES_HALF_W,
    HALF_LEN,
    SERVICE_FROM_NET,
)

logger = logging.getLogger(__name__)

# Half-width of the net band (meters). The ITF has no "net zone"; _net_y needs a
# box whose vertical center sits on the net line (world Y=0), so we synthesize a
# thin band straddling it. Pure convention — not a magic frame fraction.
NET_HALF_BAND = 0.5

# World-frame convention (matches court_detector.WORLD_KEYPOINTS):
#   origin = court center, +X = right, +Y = FAR baseline.
# So +Y zones are "top" (far in a broadcast image), -Y zones are "bottom" (near).
D = DOUBLES_HALF_W      # 5.485  (doubles half-width)
S = SINGLES_HALF_W      # 4.115  (singles half-width)
H = HALF_LEN            # 11.885 (half court length, baseline↔net)
SN = SERVICE_FROM_NET   # 6.40   (service line distance from net)

# Each zone: canonical name → 4 world-space corners (Xm, Ym), any order.
# Names match COURT_ZONE_NAMES in models/yolo_detector.py so the pipeline's
# pattern-matching consumers ("dead-zone", "service-box", "court", "net") work
# unchanged.
ZONE_WORLD_QUADS: dict[str, list[tuple[float, float]]] = {
    "court": [(-S, -H), (+S, -H), (+S, +H), (-S, +H)],
    "net": [(-D, -NET_HALF_BAND), (+D, -NET_HALF_BAND),
            (+D, +NET_HALF_BAND), (-D, +NET_HALF_BAND)],
    "top-dead-zone": [(-S, +SN), (+S, +SN), (+S, +H), (-S, +H)],
    "bottom-dead-zone": [(-S, -H), (+S, -H), (+S, -SN), (-S, -SN)],
    "left-service-box": [(-S, -H), (0.0, -H), (0.0, -SN), (-S, -SN)],
    "right-service-box": [(0.0, -H), (+S, -H), (+S, -SN), (0.0, -SN)],
    "left-doubles-alley": [(-D, -H), (-S, -H), (-S, +H), (-D, +H)],
    "right-doubles-alley": [(+S, -H), (+D, -H), (+D, +H), (+S, +H)],
}


def derive_court_zones_from_homography(court_homography: Optional[dict]) -> dict:
    """Project the 8 ITF court zones through H_inv → image-space zones.

    Returns ``{name: {"box": [x1,y1,x2,y2], "polygon": [(x,y)*4],
                       "score": 1.0, "source": "homography"}}``, drop-in
    compatible with the existing ``court_zones`` consumers. Returns ``{}`` if
    ``H_inv`` is missing. Degenerate / far-off-screen projections are skipped
    with a warning (e.g. a grazing oblique angle whose H_inv explodes at the far
    baseline).
    """
    if not court_homography or court_homography.get("H_inv") is None:
        return {}
    H_inv = np.asarray(court_homography["H_inv"], dtype=np.float64)

    # Reference frame size to sanity-check projections. We don't have the frame
    # here, so we bound by the diagonal of the projection itself: skip a zone
    # whose AABB spans >50× the median zone span (a blow-up signature).
    aabbs = []
    polys = []
    for name, quad in ZONE_WORLD_QUADS.items():
        pts = [world_to_img(H_inv, xm, ym) for (xm, ym) in quad]
        if any(not (np.isfinite(x) and np.isfinite(y)) for x, y in pts):
            logger.warning(f"court zone '{name}': non-finite projection, skipped")
            continue
        poly = np.array(pts, dtype=np.float32)
        polys.append((name, poly))

    if not polys:
        return {}

    spans = []
    for _, poly in polys:
        x0, y0, w, h = cv2.boundingRect(poly.reshape(-1, 1, 2))
        spans.append(max(w, h))
    median_span = float(np.median(spans)) if spans else 1.0

    zones = {}
    for name, poly in polys:
        x0, y0, w, h = cv2.boundingRect(poly.reshape(-1, 1, 2))
        span = max(w, h)
        # skip blow-ups (ill-conditioned H_inv near a singularity)
        if median_span > 0 and span > 50.0 * median_span:
            logger.warning(f"court zone '{name}': projection blow-up "
                           f"(span {span:.0f} >> median {median_span:.0f}), skipped")
            continue
        zones[name] = {
            "box": [float(x0), float(y0), float(x0 + w), float(y0 + h)],
            "polygon": [(float(p[0]), float(p[1])) for p in poly.tolist()],
            "score": 1.0,
            "source": "homography",
        }
    return zones


def get_court_zones(frame, court_homography, yolo_detector=None, *,
                    conf_threshold: float = 0.3) -> dict:
    """Prefer homography-derived zones (exact geometry); fall back to YOLO
    detection when the homography is absent or only a coarse fallback.

    Gating: derive whenever an H_inv is present AND confidence is not
    "fallback". This covers broadcast (high) and manually-calibrated oblique
    clips (medium/low); only the no-homography-at-all path uses YOLO.
    """
    conf = (court_homography or {}).get("confidence")
    if court_homography and court_homography.get("H_inv") is not None \
            and conf != "fallback":
        zones = derive_court_zones_from_homography(court_homography)
        if zones:
            return zones
    if yolo_detector is not None and frame is not None:
        return yolo_detector.detect_court_zones(frame, conf_threshold=conf_threshold)
    return {}
