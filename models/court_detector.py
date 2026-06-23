"""
Tennis court keypoint detector → perspective homography (px ↔ meters).

Wraps the TennisCourtDetector model (yastrebksv/TennisCourtDetector, a TrackNet
variant `BallTrackerNet`) which detects 14 court keypoints (+1 center channel) as
heatmaps from a single frame. We map those 14 keypoints to their real-world ITF
court coordinates (meters) and solve cv2.findHomography to get an image↔court-plane
transform. This replaces the geometrically-wrong scalar px-per-meter scale: under
perspective, a single scalar is wrong everywhere except one depth; a homography is
correct across the whole court.

The model arch is vendored under vendor/court/court_tracknet.py (the repo has no
LICENSE — used here as an internal evaluation/integration; flagged in Known Debt).

World frame: origin at court center, +X toward the right doubles sideline, +Y
toward the far baseline, units = meters. ITF singles/doubles dimensions.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "court"))
from court_tracknet import BallTrackerNet  # noqa: E402

OUT_W, OUT_H = 640, 360
# The model's heatmaps are at 360x640 but its keypoint coordinates are defined in
# a 720x1280 reference space (the repo's postprocess multiplies peaks by scale=2).
HM_SCALE = 2
REF_W, REF_H = OUT_W * HM_SCALE, OUT_H * HM_SCALE  # 1280 x 720
N_KP = 14

# ── ITF court geometry (meters) ─────────────────────────────────────────
DOUBLES_HALF_W = 5.485     # 10.97 / 2
SINGLES_HALF_W = 4.115     # 8.23 / 2
HALF_LEN = 11.885          # 23.77 / 2  (baseline to net)
SERVICE_FROM_NET = 6.40    # service line distance from net

# World coords for the 14 keypoints, IN THE MODEL'S key_points ORDER
# (from court_reference.py: baseline_top, baseline_bottom, left_inner_line,
#  right_inner_line, top_inner_line, bottom_inner_line, middle_line).
# +Y = far side (top in the reference image), -Y = near side.
WORLD_KEYPOINTS = np.array([
    (-DOUBLES_HALF_W, +HALF_LEN),   # 0 baseline_top L (far baseline × left doubles)
    (+DOUBLES_HALF_W, +HALF_LEN),   # 1 baseline_top R
    (-DOUBLES_HALF_W, -HALF_LEN),   # 2 baseline_bottom L (near baseline × left doubles)
    (+DOUBLES_HALF_W, -HALF_LEN),   # 3 baseline_bottom R
    (-SINGLES_HALF_W, +HALF_LEN),   # 4 left_inner_line far (far baseline × left singles)
    (-SINGLES_HALF_W, -HALF_LEN),   # 5 left_inner_line near
    (+SINGLES_HALF_W, +HALF_LEN),   # 6 right_inner_line far
    (+SINGLES_HALF_W, -HALF_LEN),   # 7 right_inner_line near
    (-SINGLES_HALF_W, +SERVICE_FROM_NET),  # 8 top_inner_line L (far service × left singles)
    (+SINGLES_HALF_W, +SERVICE_FROM_NET),  # 9 top_inner_line R
    (-SINGLES_HALF_W, -SERVICE_FROM_NET),  # 10 bottom_inner_line L (near service × left singles)
    (+SINGLES_HALF_W, -SERVICE_FROM_NET),  # 11 bottom_inner_line R
    (0.0, +SERVICE_FROM_NET),       # 12 middle_line far (center service mark, far)
    (0.0, -SERVICE_FROM_NET),       # 13 middle_line near
], dtype=np.float32)


def _peak(heatmap, low_thresh=170):
    """heatmap uint8 (360,640) -> (x,y) of the brightest blob center, or None."""
    if heatmap.max() < low_thresh:
        return None
    _, binar = cv2.threshold(heatmap, low_thresh, 255, cv2.THRESH_BINARY)
    ncc, labels, stats, cent = cv2.connectedComponentsWithStats(binar, connectivity=8)
    if ncc <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cent[best]
    return float(cx), float(cy)


class CourtKeypointDetector:
    def __init__(self, weights_path, device="mps"):
        self.device = device
        self.model = BallTrackerNet(out_channels=15)
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(sd)
        self.model.eval().to(device)
        logger.info(f"Court keypoint detector loaded ({weights_path}, device={device})")

    @torch.no_grad()
    def detect_keypoints(self, frame):
        """frame: native-res BGR. Returns list of 14 (x,y)|None in native px."""
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (OUT_W, OUT_H))
        inp = (img.astype(np.float32) / 255.0)
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(inp)
        # BallTrackerNet returns (B, 15, H*W) logits -> reshape, sigmoid
        pred = torch.sigmoid(out).detach().cpu().numpy()[0]  # (15, H*W) or (15,H,W)
        pred = pred.reshape(15, OUT_H, OUT_W)
        kps = []
        # peak is in 360x640 heatmap space → ×HM_SCALE to 720x1280 ref → ×native/ref
        sx = w0 / REF_W
        sy = h0 / REF_H
        for i in range(N_KP):
            hm = (pred[i] * 255).astype(np.uint8)
            p = _peak(hm)
            if p is not None:
                kps.append((p[0] * HM_SCALE * sx, p[1] * HM_SCALE * sy))
            else:
                kps.append(None)
        return kps

    def compute_homography(self, frame, frame_diag=None):
        """Detect keypoints and solve image→world (meters) homography.

        Returns dict: {H, H_inv, confidence, rms_px, n_used} or confidence='fallback'.
        """
        kps = self.detect_keypoints(frame)
        img_pts, world_pts, idxs = [], [], []
        for i, kp in enumerate(kps):
            if kp is not None:
                img_pts.append(kp)
                world_pts.append(WORLD_KEYPOINTS[i])
                idxs.append(i)
        n = len(img_pts)
        h0, w0 = frame.shape[:2]
        frame_diag = frame_diag or float(np.hypot(w0, h0))

        if n < 4:
            logger.warning(f"Court homography: only {n} keypoints visible (<4) → fallback")
            return {"H": None, "H_inv": None, "confidence": "fallback",
                    "rms_px": None, "n_used": n, "keypoints": kps}

        img_arr = np.array(img_pts, dtype=np.float32)
        world_arr = np.array(world_pts, dtype=np.float32)
        # Robust solve: the keypoint detector occasionally hallucinates a point
        # (e.g. a service-line mark mapped to the wrong corner, 50-190px off),
        # which a vanilla RANSAC at a sub-pixel threshold can't reject. Use LMEDS
        # (no threshold, robust to <=50% outliers), then drop any keypoint whose
        # reprojection error exceeds an absolute floor and re-solve on the clean
        # inliers. rms is computed on the inliers so confidence reflects the real
        # fit, not the hallucinated outliers.
        H, _ = cv2.findHomography(img_arr, world_arr, method=cv2.LMEDS)
        if H is None:
            return {"H": None, "H_inv": None, "confidence": "fallback",
                    "rms_px": None, "n_used": n, "keypoints": kps}
        H_inv = np.linalg.inv(H)
        reproj = cv2.perspectiveTransform(world_arr.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        err = np.linalg.norm(reproj - img_arr, axis=1)
        outlier_floor = max(12.0, frame_diag * 0.008)   # ~1% of diagonal
        inlier = err <= outlier_floor
        n_used = int(inlier.sum())
        if n_used >= 4 and not inlier.all():
            img_in = img_arr[inlier]
            world_in = world_arr[inlier]
            H2, _ = cv2.findHomography(img_in, world_in, method=cv2.LMEDS)
            if H2 is not None:
                H, H_inv = H2, np.linalg.inv(H2)
                reproj = cv2.perspectiveTransform(
                    world_in.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
                err = np.linalg.norm(reproj - img_in, axis=1)
        rms = float(np.sqrt(np.mean(err ** 2)))

        thr = frame_diag * 0.004   # ~ a few px on 1080p
        if rms < thr:
            conf = "high"
        elif rms < 2 * thr:
            conf = "low"
        else:
            conf = "fallback"
            logger.warning(f"Court homography rms={rms:.1f}px > {2*thr:.1f}px → unreliable, fallback")
        logger.info(f"Court homography: {n_used}/{n} keypoints, rms={rms:.1f}px, confidence={conf}")
        return {"H": H, "H_inv": H_inv, "confidence": conf, "rms_px": rms,
                "n_used": n_used, "keypoints": kps}


def img_to_world(H, x, y):
    """Map an image point (px) to world coords (meters) through H."""
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])
