"""
Semi-automatic court homography from a handful of annotated court points.

The pretrained court-keypoint model (models/court_detector.py) only localizes
enough keypoints on standard broadcast angles; on oblique / amateur / court-level
framings (our target use case, e.g. felix.mp4) it fires on <4 points and we lose
the metric scale — which kills the minimap, real speed and depth.

For a STATIC camera (the overwhelmingly common amateur case: a phone on a tripod)
the court homography is constant for the whole clip. So we only need the court's
key points ONCE, in image pixels, and we can solve a single exact homography that
holds for every frame. This module:

  * defines the named court landmarks in ITF world meters (same frame as
    models.court_detector.WORLD_KEYPOINTS: origin at court center, +X right,
    +Y toward the far baseline),
  * builds an image→world homography from >=4 named image points
    (cv2.findHomography), and
  * loads/saves these annotations as a small JSON sidecar keyed by video, so a
    clip is calibrated once and reused (the seed of a future SaaS "draw your
    court" calibration step).

A homography needs >=4 non-collinear correspondences; the 4 court CORNERS
(doubles baseline intersections) are the easiest to click and give an excellent
conditioned solve. More points (service-box corners, net posts) improve accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2

# ITF court geometry (meters) — must match models/court_detector.py
DOUBLES_HALF_W = 5.485     # 10.97 / 2
SINGLES_HALF_W = 4.115     # 8.23 / 2
HALF_LEN = 11.885          # 23.77 / 2  (baseline to net is HALF_LEN)
SERVICE_FROM_NET = 6.40

# Named court landmarks in world meters. Names are stable keys an annotation tool
# (or a person) uses to say "this image pixel is THAT court point".
#   far  = the baseline farther from the camera (+Y)
#   near = the baseline closer to the camera   (-Y)
WORLD_LANDMARKS = {
    # 4 doubles corners (the easy, high-value click targets)
    "far_bl_left":   (-DOUBLES_HALF_W, +HALF_LEN),
    "far_bl_right":  (+DOUBLES_HALF_W, +HALF_LEN),
    "near_bl_left":  (-DOUBLES_HALF_W, -HALF_LEN),
    "near_bl_right": (+DOUBLES_HALF_W, -HALF_LEN),
    # singles sideline × baseline (inner corners)
    "far_sl_left":   (-SINGLES_HALF_W, +HALF_LEN),
    "far_sl_right":  (+SINGLES_HALF_W, +HALF_LEN),
    "near_sl_left":  (-SINGLES_HALF_W, -HALF_LEN),
    "near_sl_right": (+SINGLES_HALF_W, -HALF_LEN),
    # service line × singles sideline
    "far_svc_left":  (-SINGLES_HALF_W, +SERVICE_FROM_NET),
    "far_svc_right": (+SINGLES_HALF_W, +SERVICE_FROM_NET),
    "near_svc_left": (-SINGLES_HALF_W, -SERVICE_FROM_NET),
    "near_svc_right":(+SINGLES_HALF_W, -SERVICE_FROM_NET),
    # center service marks + net posts
    "far_svc_center":  (0.0, +SERVICE_FROM_NET),
    "near_svc_center": (0.0, -SERVICE_FROM_NET),
    "net_left":  (-DOUBLES_HALF_W, 0.0),
    "net_right": (+DOUBLES_HALF_W, 0.0),
}


def homography_from_points(image_points: dict):
    """image_points: {landmark_name: (x_px, y_px)}. Returns a homography dict
    {H, H_inv, confidence, rms_px, n_used, source} mapping image px → world meters.

    Requires >=4 known landmarks. RMS reprojection error (world→image) is reported
    in pixels; <0.4% of the frame diagonal is 'high' confidence.
    """
    img_pts, world_pts, names = [], [], []
    for name, (x, y) in image_points.items():
        if name not in WORLD_LANDMARKS:
            raise ValueError(f"Unknown court landmark '{name}'. "
                             f"Valid: {sorted(WORLD_LANDMARKS)}")
        img_pts.append((float(x), float(y)))
        world_pts.append(WORLD_LANDMARKS[name])
        names.append(name)

    n = len(img_pts)
    if n < 4:
        raise ValueError(f"Need >=4 court points for a homography, got {n}")

    img_arr = np.array(img_pts, dtype=np.float32)
    world_arr = np.array(world_pts, dtype=np.float32)

    H, mask = cv2.findHomography(img_arr, world_arr, method=cv2.RANSAC,
                                 ransacReprojThreshold=0.10)  # 10 cm in court meters
    if H is None:
        raise RuntimeError("cv2.findHomography failed (degenerate / collinear points?)")
    H_inv = np.linalg.inv(H)

    # reprojection error in PIXELS (world → image), the honest accuracy metric
    reproj = cv2.perspectiveTransform(world_arr.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    err = np.linalg.norm(reproj - img_arr, axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))

    # Honest confidence. With exactly 4 points the solve is exact (rms≈0), so rms
    # alone is meaningless — the real risk is OVER-EXTRAPOLATION from points that
    # don't span the court (the grazing-oblique case). We test conditioning by how
    # well the homography maps a reference court rectangle back to a sane image
    # quad: project the 4 ITF doubles corners to image and check they form a
    # convex, non-degenerate, in-frame-ish quad spanning a reasonable area.
    confidence = _assess_conditioning(H_inv, world_pts, img_arr, n, rms)

    # named landmark → image pixel (used by the minimap's geometric fallback to
    # anchor the depth band on the real far/near baseline rows)
    landmark_px = {name: (float(px), float(py))
                   for name, (px, py) in zip(names, img_pts)}

    return {"H": H, "H_inv": H_inv, "rms_px": rms, "n_used": n,
            "landmarks": names, "landmark_px": landmark_px,
            "source": "manual4pt", "confidence": confidence}


def _assess_conditioning(H_inv, world_pts, img_arr, n, rms):
    """Return 'high' | 'medium' | 'low' for how trustworthy the homography is.

    Grazing-oblique 4-point solves are exact at the control points (rms≈0) but
    extrapolate wildly off them. We flag that by measuring how much the control
    points span the court's WIDTH and DEPTH (a solve with no near-half or
    one-sided points can't constrain the far field) and whether projecting the
    full doubles rectangle back to the image yields a convex, sane quad."""
    wp = np.array(world_pts, dtype=np.float32)
    x_span = float(wp[:, 0].max() - wp[:, 0].min())   # court-width coverage (m)
    y_span = float(wp[:, 1].max() - wp[:, 1].min())   # court-depth coverage (m)
    # full doubles rectangle → image; must be a convex non-self-intersecting quad
    corners_world = np.array([[-DOUBLES_HALF_W, +HALF_LEN], [+DOUBLES_HALF_W, +HALF_LEN],
                              [+DOUBLES_HALF_W, -HALF_LEN], [-DOUBLES_HALF_W, -HALF_LEN]],
                             dtype=np.float32)
    proj = cv2.perspectiveTransform(corners_world.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    hull = cv2.convexHull(proj.astype(np.float32))
    convex = len(hull) == 4                            # all 4 corners on the hull → convex quad

    # IMAGE-space conditioning: a GRAZING oblique angle (camera near-parallel to a
    # sideline) crushes the 23.77m-deep court into a thin image band — the depth
    # axis is severely foreshortened, so tiny pixel errors blow up into large
    # meter errors in the far field (the felix case). The tell is the projected
    # court's image ASPECT: a healthy view has real vertical extent; a grazing one
    # has court_h ≪ court_w. We also penalize when the court fills very little of
    # the frame's height (a thin sliver).
    court_w = float(proj[:, 0].max() - proj[:, 0].min())
    court_h = float(proj[:, 1].max() - proj[:, 1].min())
    aspect = court_h / (court_w + 1e-6)                 # depth-extent / width-extent

    well_spread = x_span >= 7.0 and y_span >= 12.0     # ~most of doubles width + both baselines
    some_spread = x_span >= 5.0 and y_span >= 6.0
    # A non-grazing court view projects to an aspect well above ~0.35 (the court is
    # deeper than it is wide in reality; perspective compresses it but not to a sliver).
    if convex and well_spread and aspect >= 0.35 and n >= 4:
        return "high"
    if convex and some_spread and aspect >= 0.18:
        return "medium"
    return "low"


def img_to_world(H, x, y):
    """Image pixel → world meters through H."""
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def world_to_img(H_inv, Xm, Ym):
    """World meters → image pixel through H_inv."""
    p = H_inv @ np.array([Xm, Ym, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# ───────────────────────── annotation sidecar ──────────────────────────────

def _sidecar_path(video_path: str) -> Path:
    """Where a clip's court calibration lives: <video>.court.json next to it,
    falling back to a repo-level calibrations/ dir for read-only assets."""
    vp = Path(video_path)
    return vp.with_suffix(vp.suffix + ".court.json")


def save_calibration(video_path: str, image_points: dict, frame_wh=None):
    """Persist {landmark: [x,y]} for a clip so it's calibrated once."""
    data = {
        "video": str(Path(video_path).name),
        "frame_wh": list(frame_wh) if frame_wh else None,
        "points": {k: [float(v[0]), float(v[1])] for k, v in image_points.items()},
    }
    path = _sidecar_path(video_path)
    path.write_text(json.dumps(data, indent=2))
    return path


def load_calibration(video_path: str):
    """Load a clip's saved court points → homography dict, or None if no sidecar.

    Also checks a repo-level `calibrations/<name>.court.json` so committed test
    fixtures (e.g. felix) work without a sidecar next to the raw video."""
    candidates = [_sidecar_path(video_path)]
    name = Path(video_path).stem
    repo_cal = Path(__file__).resolve().parents[1] / "calibrations" / f"{name}.court.json"
    candidates.append(repo_cal)
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            pts = {k: tuple(v) for k, v in data["points"].items()}
            homo = homography_from_points(pts)
            homo["source"] = f"calibration:{path.name}"
            return homo
    return None
