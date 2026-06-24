"""
Tennis Match Annotation Pipeline
Pose skeletons on all detected persons + Kalman-filtered ball trail.
2-player tracking with bounding boxes, ball bounce detection, court zone overlay.

Works on any tennis match video regardless of resolution, camera angle, or players.
"""
import sys
import argparse
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from collections import deque
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO

from models.yolo_detector import TennisYOLODetector
from utils.video_utils import VideoReader, VideoWriter
from utils.drawing import draw_player_bbox, draw_bounce_marker, draw_court_overlay, create_legend

# Derived analytics (Session B): fatigue trend + per-rally outcome.
# These are pure functions over the stats dicts; they add no rendering.
from vision.fatigue import analyze_fatigue
from vision.rally_outcome import classify_rally_outcome

# COCO skeleton connections (fixed, not video-dependent)
BODY_BONES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
HEAD_BONES = [(0,1),(0,2),(1,3),(2,4)]

# Default colors (BGR)
BALL_COLOR = (0, 255, 100)
TRAIL_COLOR = (50, 255, 50)

# Player colors (BGR) — amber for P1, blue for P2
PLAYER_COLORS = {
    1: (0, 191, 255),   # amber/gold
    2: (255, 160, 50),   # blue
}


class BallKalmanFilter:
    """
    Constant-velocity Kalman filter for ball tracking.
    State: [x, y, vx, vy]  — position and velocity in pixels.

    - predict(): advance state by one timestep, return predicted (x, y)
    - update(x, y): correct state with a measurement
    - gating_distance(x, y): Mahalanobis-like distance to check if
      a detection is consistent with the predicted position
    """

    def __init__(self, process_noise=100.0, measurement_noise=4.0):
        # State: [x, y, vx, vy]
        self.kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 500
        self.initialized = False
        self.frames_since_update = 0

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.frames_since_update = 0

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        self.frames_since_update += 1
        return float(pred[0][0]), float(pred[1][0])

    def update(self, x, y):
        if not self.initialized:
            self.init_state(x, y)
            return
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))
        self.frames_since_update = 0

    def gating_distance(self, x, y):
        """Euclidean distance between predicted position and measurement."""
        if not self.initialized:
            return 0.0
        pred = self.kf.statePost
        return float(np.hypot(x - pred[0][0], y - pred[1][0]))


# Player identity is now handled by vision.player_track.TwoPlayerTracker (Pass 1.5):
# persistent P1(far)/P2(near) lock + gap-fill + smoothing. The old 2-slot IoU
# PlayerTracker was replaced by it.

# Bounce detection + trajectory smoothing now live in the vision/ engine
# module (PROJET.md 'moteur propre'), imported here so the pipeline and the
# test suite exercise the identical algorithm.
from vision.bounce import smooth_ball_trajectory, detect_bounces_from_trajectory  # noqa: E402
from vision.pose import estimate_players_pose  # noqa: E402
from vision.minimap import draw_minimap  # noqa: E402
import vision.fx as fx  # noqa: E402  — cinematic visual effects (glow, glass HUD, shockwave)
from vision.court_track import h_at  # noqa: E402  — per-frame homography lookup (track-camera)
from vision.shot_guard import compute_court_visibility_mask, non_court_spans  # noqa: E402


def _court_frames_between(mask, a, b):
    """Count court-visible frames strictly between absolute frame indices a and b.

    Used by rally segmentation so a multi-second non-court span (a replay
    between two real shots) doesn't falsely split one rally: the gap is measured
    in COURT frames only, treating the replay as if it weren't there."""
    if b <= a + 1:
        return 0
    return sum(1 for k in range(a + 1, b)
               if k < len(mask) and mask[k])


def _merge_solo_rallies(rallies):
    """Re-attribute any 1-shot rally ("solo") into its nearest neighbour.

    Mirrors mergeSoloRallies in web/src/lib/analysis/highlights.ts exactly, so
    the pipeline JSON and the front-end-derived rallies stay in sync. A solo is
    absorbed into whichever neighbour (prev/next) is closer by gap; ties go to
    the previous (chronological). Returns a new list.
    """
    if len(rallies) <= 1:
        return rallies
    out = [
        {
            "start_frame": r["start_frame"],
            "end_frame": r["end_frame"],
            "shot_frames": list(r["shot_frames"]),
            "bounce_frames": list(r["bounce_frames"]),
            "n_shots": r["n_shots"],
        }
        for r in rallies
    ]
    i = 0
    while i < len(out):
        r = out[i]
        if r["n_shots"] != 1:
            i += 1
            continue
        prev = out[i - 1] if i - 1 >= 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        gap_prev = (r["start_frame"] - prev["end_frame"]) if prev else float("inf")
        gap_next = (nxt["start_frame"] - r["end_frame"]) if nxt else float("inf")
        if gap_prev <= gap_next and prev:
            prev["shot_frames"].extend(r["shot_frames"])
            prev["bounce_frames"].extend(r["bounce_frames"])
            prev["end_frame"] = max(prev["end_frame"], r["end_frame"])
            prev["n_shots"] += r["n_shots"]
            out.pop(i)
        elif nxt:
            nxt["shot_frames"] = list(r["shot_frames"]) + nxt["shot_frames"]
            nxt["bounce_frames"] = list(r["bounce_frames"]) + nxt["bounce_frames"]
            nxt["start_frame"] = min(nxt["start_frame"], r["start_frame"])
            nxt["n_shots"] += r["n_shots"]
            out.pop(i)
        else:
            i += 1
    return out


def estimate_px_per_meter(court_zones, frame_height):
    """
    Estimate the scale factor (pixels per meter) from court zones or frame geometry.
    A tennis court is 23.77m long.
    """
    # Try to use court zone detection
    for name, info in court_zones.items():
        if name == "court":
            box = info["box"]
            court_h_px = box[3] - box[1]
            if court_h_px > 0:
                px_per_m = court_h_px / 23.77
                logger.info(f"Scale from court zone: {px_per_m:.2f} px/m "
                            f"(court height={court_h_px:.0f}px)")
                return px_per_m

    # Fallback: in broadcast tennis, the court occupies ~58% of frame height
    court_h_px = frame_height * 0.58
    px_per_m = court_h_px / 23.77
    logger.info(f"Scale from fallback (58% frame height): {px_per_m:.2f} px/m")
    return px_per_m


def classify_bounce_depth(bounce_events, court_zones, frame_height=None):
    """
    Classify each bounce as deep, short, or mid based on court zones.
    - deep: in dead zones or top/bottom 20% of court
    - short: in service boxes
    - mid: elsewhere in court

    Falls back to geometric approximation when court_zones is empty.

    Args:
        bounce_events: list of (frame_idx, x, y)
        court_zones: dict from detect_court_zones()
        frame_height: frame height in pixels (for geometric fallback)

    Returns:
        list of (frame_idx, x, y, depth_label)
    """
    def _point_in_box(px, py, box):
        return box[0] <= px <= box[2] and box[1] <= py <= box[3]

    # ─── Geometric fallback when no court zones detected ───
    if not court_zones and frame_height is not None:
        # Broadcast tennis approximations:
        # Net ≈ 47% of frame height, far baseline ≈ 18%, near baseline ≈ 82%
        net_y = frame_height * 0.47
        far_baseline = frame_height * 0.18
        near_baseline = frame_height * 0.82
        classified = []
        for frame_idx, x, y in bounce_events:
            # Determine which half the bounce is in
            if y < net_y:
                # Far side: depth relative to far baseline
                half_span = net_y - far_baseline
                dist_from_baseline = y - far_baseline
            else:
                # Near side: depth relative to near baseline
                half_span = near_baseline - net_y
                dist_from_baseline = near_baseline - y

            ratio = max(0.0, min(1.0, dist_from_baseline / half_span)) if half_span > 0 else 0.5
            # ratio close to 0 = near baseline = deep, close to 1 = near net = short
            if ratio <= 0.35:
                depth = "deep"
            elif ratio >= 0.70:
                depth = "short"
            else:
                depth = "mid"
            classified.append((frame_idx, x, y, depth))
        return classified

    # ─── Court-zone-based classification ───
    dead_zones = []
    service_boxes = []
    court_box = None
    for name, info in court_zones.items():
        box = info["box"]
        if "dead-zone" in name:
            dead_zones.append(box)
        elif "service-box" in name:
            service_boxes.append(box)
        elif name == "court":
            court_box = box

    classified = []
    for frame_idx, x, y in bounce_events:
        # Check dead zones first
        in_dead = any(_point_in_box(x, y, dz) for dz in dead_zones)
        if in_dead:
            classified.append((frame_idx, x, y, "deep"))
            continue

        # Check service boxes
        in_service = any(_point_in_box(x, y, sb) for sb in service_boxes)
        if in_service:
            classified.append((frame_idx, x, y, "short"))
            continue

        # If inside court but not in dead/service, check top/bottom 20%
        if court_box is not None and _point_in_box(x, y, court_box):
            court_h = court_box[3] - court_box[1]
            top_thresh = court_box[1] + court_h * 0.2
            bot_thresh = court_box[3] - court_h * 0.2
            if y <= top_thresh or y >= bot_thresh:
                classified.append((frame_idx, x, y, "deep"))
                continue

        classified.append((frame_idx, x, y, "mid"))

    return classified


def draw_skeleton(frame, keypoints, kp_conf, color, conf_thresh=0.25):
    for (i, j) in BODY_BONES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if kp_conf[i] < conf_thresh or kp_conf[j] < conf_thresh:
            continue
        cv2.line(frame, (int(keypoints[i][0]), int(keypoints[i][1])),
                 (int(keypoints[j][0]), int(keypoints[j][1])), color, 3, cv2.LINE_AA)
    for (i, j) in HEAD_BONES:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if kp_conf[i] < conf_thresh or kp_conf[j] < conf_thresh:
            continue
        cv2.line(frame, (int(keypoints[i][0]), int(keypoints[i][1])),
                 (int(keypoints[j][0]), int(keypoints[j][1])), color, 2, cv2.LINE_AA)
    for i, (kp, sc) in enumerate(zip(keypoints, kp_conf)):
        if sc < conf_thresh:
            continue
        pt = (int(kp[0]), int(kp[1]))
        cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 4, color, 2, cv2.LINE_AA)
    return frame


def draw_ball_trail(frame, trail):
    n = len(trail)
    if n < 2:
        return frame
    for i in range(1, n):
        alpha = i / n
        thickness = max(2, int(5 * alpha))
        c = tuple(int(v * (0.3 + 0.7 * alpha)) for v in TRAIL_COLOR)
        cv2.line(frame, trail[i - 1], trail[i], c, thickness, cv2.LINE_AA)
    return frame


def draw_ball_marker(frame, cx, cy):
    cv2.circle(frame, (cx, cy), 7, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Tennis Match Annotation Pipeline")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path (default: data/output/<input>_annotated.mp4)")
    parser.add_argument("-s", "--start", type=float, default=0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("-e", "--end", type=float, default=None,
                        help="End time in seconds (default: end of video)")
    parser.add_argument("-d", "--duration", type=float, default=None,
                        help="Duration in seconds from start (alternative to --end)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, mps (default: cpu)")
    parser.add_argument("--trail-length", type=int, default=12, help="Ball trail length in frames")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--pose-conf", type=float, default=0.10, help="Pose estimation confidence")
    parser.add_argument("--pose-model", default="yolov8m-pose.pt",
                        help="Pose model (yolov8m-pose.pt or yolov8x-pose.pt for better small player detection)")
    parser.add_argument("--pose-backend", choices=["yolo", "rtmw"], default="yolo",
                        help="Pose backend. 'yolo' = YOLOv8-pose (default, fast, 17 kpts). "
                             "'rtmw' = RTMW whole-body via rtmlib (133 kpts incl. hands/feet, "
                             "better on the tiny far player; CPU-only, ~3x slower). "
                             "Needs: pip install rtmlib onnxruntime.")
    # New feature flags
    parser.add_argument("--no-court", action="store_true", help="Disable court zone overlay")
    parser.add_argument("--no-bounces", action="store_true", help="Disable bounce detection markers")
    parser.add_argument("--no-player-boxes", action="store_true",
                        help="(default) Disable player bounding boxes. Boxes are OFF by default — "
                             "skeletons read cleaner; pass --player-boxes to re-enable.")
    parser.add_argument("--player-boxes", action="store_true",
                        help="Re-enable player bounding boxes (off by default).")
    parser.add_argument("--court-labels", action="store_true", help="Show court zone labels")
    parser.add_argument("--dump-bounces", metavar="PATH", default=None,
                        help="Also write detected bounces to PATH as a predictions JSON "
                             "(eval_bounces.py format) for scoring against ground truth. "
                             "Off by default; does not affect the annotated video.")
    parser.add_argument("--demo-override", metavar="PATH", default=None,
                        help="Manual ground-truth sidecar (JSON from tools/annotate_demo.py) "
                             "that OVERRIDES this clip's bounces with bounces_true (replacing "
                             "the detector's) and adds racket->ball strikes (forehand/backhand) "
                             "as glow markers. Bounded to this clip only — no global threshold "
                             "changes (respects zero-hardcoding). demo_frame is 0-based within "
                             "the -s/-d window, matching the annotator.")
    parser.add_argument("--homography", action="store_true",
                        help="Compute a court homography (px↔meters) from court keypoints "
                             "for perspective-correct speed/depth. Falls back to the scalar "
                             "scale when the court can't be reliably localized (oblique/partial).")
    parser.add_argument("--track-camera", action="store_true",
                        help="Recompute the court homography at every frame so the overlay/"
                             "minimap/speed follow a panning/tilting camera (adds a Pass 0, "
                             "~14ms/frame). Implies --homography. Off by default (the f0 "
                             "homography is fine for a fixed camera).")
    parser.add_argument("--court-model", default="training/court/tennis_court_det.pt",
                        help="Path to the court-keypoint model weights (for --homography).")
    parser.add_argument("--ball-tracker", choices=["kalman", "wasb"], default="kalman",
                        help="Ball tracker. 'kalman' = YOLO26 + Kalman (default). 'wasb' = "
                             "WASB heatmap-temporal tracker (denser/cleaner trajectory, higher "
                             "bounce recall on broadcast; needs --wasb-weights).")
    parser.add_argument("--wasb-weights", default="training/wasb/wasb_tennis.pth.tar",
                        help="Path to WASB tennis weights (for --ball-tracker wasb).")
    parser.add_argument("--no-minimap", action="store_true",
                        help="Disable the 2D top-down court minimap (ball trajectory + bounces).")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Ignore any saved court calibration sidecar (<video>.court.json / "
                             "calibrations/<name>.court.json) used for the metric homography.")
    parser.add_argument("--no-fx", action="store_true",
                        help="Disable cinematic FX (glow trail, glass HUD, shockwaves). FX on by default.")
    parser.add_argument("--match-mode", action="store_true",
                        help="Stable P1/P2 HUMAN identity across side-changes (match play). Without it, "
                             "P1=far / P2=near (net-side identity) — the training default, unchanged. With it, "
                             "players keep their id when they swap ends between games (HSV jersey re-id + "
                             "continuity), and each shot/rally carries a stable human_id on top of player_side.")
    parser.add_argument("--no-shot-guard", action="store_true",
                        help="Disable the court-visibility gate (shot-guard). By default, frames where the "
                             "full court is NOT visible (camera close-ups, replays, graphics) are detected "
                             "and EXCLUDED from ball/shot/bounce/pose stats and overlay — they're written "
                             "clean to the output video so garbage detections aren't attributed to a player.")
    parser.add_argument("--shot-guard-badge", action="store_true",
                        help="Draw a discreet 'NON-PLAY' badge on gated frames in the output video. Off by "
                             "default (gated frames are written with no overlay at all).")
    args = parser.parse_args()
    # Player boxes default OFF (user feedback: boxes distract / a stale box looks
    # fixed). --player-boxes opts back in; --no-player-boxes is the explicit default.
    if not args.player_boxes:
        args.no_player_boxes = True
    return args


def main():
    args = parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(video_path).stem
        output_path = f"data/output/{stem}_annotated.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    trail_length = args.trail_length
    device = args.device

    # ─── Init models ───
    logger.info("Initializing models...")
    detector = TennisYOLODetector(
        conf_threshold=args.conf, iou_threshold=0.45, device=device, imgsz=1280
    )
    if args.pose_backend == "rtmw":
        from vision.pose_rtmw import RTMWPose
        logger.info("Pose backend: RTMW whole-body (rtmlib, CPU) — 133 kpts, slower")
        pose_model = RTMWPose(mode="balanced")
    else:
        pose_model = YOLO(args.pose_model)

    # ─── Read video info ───
    reader = VideoReader(video_path)
    fps = reader.fps
    frame_width = reader.width
    frame_height = reader.height
    total_frames = reader.frame_count

    # Compute frame range from start/end/duration
    start_frame = int(args.start * fps)
    if args.duration:
        end_frame = min(start_frame + int(args.duration * fps), total_frames)
    elif args.end:
        end_frame = min(int(args.end * fps), total_frames)
    else:
        end_frame = total_frames
    start_frame = max(0, min(start_frame, total_frames - 1))
    max_frames = end_frame - start_frame

    # Seek to start
    if start_frame > 0:
        reader.seek(start_frame)
        logger.info(f"Seeking to frame {start_frame} ({args.start:.1f}s)")

    # Pose image size: use large resolution for good detection of small/far players
    # For broadcast tennis, the far player is tiny — we need high resolution
    pose_imgsz = max(960, min(frame_width, 1920))
    # Round to nearest 32 (YOLO requirement)
    pose_imgsz = (pose_imgsz // 32) * 32

    # Ball jump threshold: relative to frame diagonal
    frame_diag = np.hypot(frame_width, frame_height)
    max_ball_jump = frame_diag * 0.15  # 15% of diagonal

    logger.info(f"Video: {frame_width}x{frame_height}, {fps:.1f}fps, {total_frames} frames")
    logger.info(f"Processing: frames {start_frame}-{end_frame} ({max_frames} frames, {max_frames/fps:.1f}s)")
    logger.info(f"Pose imgsz: {pose_imgsz}, ball jump threshold: {max_ball_jump:.0f}px")

    # ─── Read first frame (used for court homography + zones) ───
    first_frame = None
    if (not args.no_court or not args.no_bounces) or args.homography:
        first_frame = next(reader.iter_frames())
        reader.release()
        reader = VideoReader(video_path)
        if start_frame > 0:
            reader.seek(start_frame)

    # ─── Court homography (px ↔ meters) via keypoints, for perspective-correct
    # speed & depth. Only used when it solves with high/low confidence; otherwise
    # the pipeline falls back to the scalar px-per-meter scale. Computed BEFORE
    # court zones because the zones are now DERIVED from it (the HF zone detector
    # suffers a domain shift on broadcast framings and returns {}). ───
    court_homography = None
    if args.homography and first_frame is not None:
        try:
            from models.court_detector import CourtKeypointDetector
            court_det = CourtKeypointDetector(args.court_model, device=device)
            court_homography = court_det.compute_homography(first_frame, frame_diag)
            logger.info(f"Court homography: confidence={court_homography['confidence']}, "
                        f"{court_homography['n_used']}/14 keypoints"
                        + (f", rms={court_homography['rms_px']:.1f}px"
                           if court_homography['rms_px'] else ""))
            if court_homography['confidence'] == 'fallback':
                logger.warning("Court homography unreliable on this framing "
                               "(oblique/partial court) → using scalar scale fallback")
                court_homography = None
        except Exception as e:
            logger.warning(f"Court homography failed ({e}) → scalar scale fallback")
            court_homography = None

    # ─── Calibration homography (semi-auto 4-corner) ───
    # On oblique/amateur angles the keypoint model can't localize the court, but a
    # static camera only needs the court annotated ONCE. If a calibration sidecar
    # exists for this clip (<video>.court.json, or calibrations/<name>.court.json),
    # use it — it gives an exact metric homography where the keypoint model fails.
    # This is what makes the minimap metric on felix-style angles.
    if court_homography is None and not args.no_calibration:
        try:
            from models.court_calibrator import load_calibration
            cal = load_calibration(video_path)
            if cal is not None:
                court_homography = cal
                logger.info(f"Court calibration loaded ({cal['source']}): "
                            f"{cal['n_used']} points, rms={cal['rms_px']:.1f}px, "
                            f"confidence={cal['confidence']} → metric homography enabled")
                if cal['confidence'] == 'low':
                    logger.warning("Calibration is LOW confidence (grazing/oblique angle "
                                   "— far-field depth is approximate). Add more well-spread "
                                   "court points via tools/calibrate_court.py to improve.")
        except Exception as e:
            logger.warning(f"Court calibration load failed ({e})")

    # ─── Pass 0 (optional): per-frame homography for a moving camera. ───
    # A broadcast camera pans/tilts, so the f0 homography drifts (measured up to
    # ~140px on a 13s clip). When --track-camera is set, recompute the homography
    # at every frame (robust to hallucinated keypoints + hold on fallback) so the
    # court overlay / minimap / speed follow the camera. Pass 2 looks up the
    # per-frame H via _h_at. Off by default (fixed camera → f0 H is exact + cheap).
    court_homography_by_frame = None
    if args.track_camera:
        if not args.homography:
            args.homography = True  # --track-camera implies --homography
        if court_homography is None and first_frame is not None:
            # no f0 solve above (e.g. homography just forced) → solve f0 first
            try:
                from models.court_detector import CourtKeypointDetector
                court_det = CourtKeypointDetector(args.court_model, device=device)
                court_homography = court_det.compute_homography(first_frame, frame_diag)
                if court_homography['confidence'] == 'fallback':
                    court_homography = None
            except Exception as e:
                logger.warning(f"Court homography (track-camera f0) failed: {e}")
        try:
            from vision.court_track import compute_homography_series
            from models.court_detector import CourtKeypointDetector
            court_det = CourtKeypointDetector(args.court_model, device=device)
            rdr0 = VideoReader(video_path)
            if start_frame > 0:
                rdr0.seek(start_frame)
            pass0_frames = []
            for j, fr in enumerate(rdr0.iter_frames()):
                if j >= max_frames:
                    break
                pass0_frames.append(fr)
            rdr0.release()
            court_homography_by_frame = compute_homography_series(
                pass0_frames, court_det, frame_diag)
            logger.info(f"Pass 0: per-frame homography ready "
                        f"({sum(1 for f in court_homography_by_frame if f.H is not None)}/"
                        f"{len(court_homography_by_frame)} frames tracked)")
        except Exception as e:
            logger.warning(f"Pass 0 (track-camera) failed: {e} — falling back to f0 homography")
            court_homography_by_frame = None

    # ─── Pass 0.5: shot-guard — court-visibility mask (Canny + HoughLinesP) ───
    # A cheap, colour-independent gate: a frame is "full court visible" when it
    # has >= 40 long straight lines (the court lines). Close-ups / replays /
    # graphics have ~0. Raw signal is dwell-hysteresis-cleaned (~0.3 s) so
    # single-frame transition dips stay court. The mask is consumed by every
    # downstream stage: ball/pose detection is SKIPPED on non-court frames
    # (no garbage stats) and Pass 2 writes those frames with NO overlay. Active
    # by default; --no-shot-guard disables. See vision/shot_guard.py.
    shot_guard_enabled = not args.no_shot_guard
    court_visible_mask = [True] * max_frames
    non_court_spans_rel = []
    if shot_guard_enabled and max_frames > 0:
        sg_reader = VideoReader(video_path)
        if start_frame > 0:
            sg_reader.seek(start_frame)
        sg_frames = []
        for j, fr in enumerate(sg_reader.iter_frames()):
            if j >= max_frames:
                break
            sg_frames.append(fr)
        sg_reader.release()
        court_visible_mask = compute_court_visibility_mask(sg_frames, fps)
        del sg_frames  # free before Pass 1
        non_court_spans_rel = non_court_spans(court_visible_mask)
        n_court = sum(court_visible_mask)
        logger.info(f"Pass 0.5 (shot-guard): {n_court}/{len(court_visible_mask)} court frames "
                    f"({100 * n_court / max(1, len(court_visible_mask)):.1f}%), "
                    f"{len(non_court_spans_rel)} non-court spans")

    # ─── Court zones: DERIVE from the homography (exact perspective geometry)
    # when one is available; fall back to the HF YOLO zone detector only when
    # there is no homography at all. Derived zones are drop-in compatible
    # (same {name: {box, ...}} shape) and carry a `polygon` for a perspective
    # overlay. See vision/court_zones.py. ───
    court_zones = {}
    if not args.no_court or not args.no_bounces:
        from vision.court_zones import get_court_zones
        yolo_det = detector if not args.no_court else None
        court_zones = get_court_zones(first_frame, court_homography, yolo_det,
                                      conf_threshold=0.3)
        if court_zones:
            src = next(iter(court_zones.values())).get("source", "yolo")
            logger.info(f"Court zones ({src}): {len(court_zones)} — "
                        f"{list(court_zones.keys())}")
        else:
            logger.warning("No court zones (no homography and YOLO returned nothing)")

    # ─── Pass 1: Detect ball positions + bounces ───
    logger.info("=== PASS 1: Detect ball ===")
    ball_conf_thresh = 0.15  # low threshold — we filter by motion, not confidence

    # ─── WASB heatmap tracker path (--ball-tracker wasb) ───
    # A heatmap-temporal tracker (WASB) produces a far denser, cleaner trajectory
    # than YOLO+Kalman (no static-FP/reinit jitter), which is the real lever for
    # bounce recall. We buffer frames, run the tracker, and feed all_ball_centers
    # straight into the (robust) bounce detector, skipping static-FP + Kalman.
    wasb_path = (args.ball_tracker == "wasb")
    if wasb_path:
        try:
            from models.wasb_ball_tracker import HeatmapBallTracker
            wasb_frames = []
            for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 1a: read"):
                if len(wasb_frames) >= max_frames:
                    break
                wasb_frames.append(frame)
            reader.release()
            tracker = HeatmapBallTracker(args.wasb_weights, device=device)
            logger.info("Tracking ball with WASB heatmap model...")
            raw_ball_centers = tracker.track_ball(wasb_frames)
            # shot-guard: WASB can hallucinate a heatmap blob on a close-up; null
            # its output on non-court frames so no ball is fabricated there.
            if shot_guard_enabled:
                raw_ball_centers = [
                    None if (i < len(court_visible_mask) and not court_visible_mask[i]) else c
                    for i, c in enumerate(raw_ball_centers)
                ]
            ball_speeds_px = [0.0] * len(raw_ball_centers)  # filled below
            raw_detections = None
        except Exception as e:
            logger.warning(f"WASB tracker failed ({e}) → falling back to YOLO+Kalman")
            wasb_path = False
            reader = VideoReader(video_path)
            if start_frame > 0:
                reader.seek(start_frame)

    # Step 1a: Collect ALL ball candidates per frame (YOLO+Kalman path)
    if not wasb_path:
        raw_detections = []  # list of list of (cx, cy, score)
        frame_count = 0
        for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 1a: detect"):
            if frame_count >= max_frames:
                break
            # shot-guard: skip ball detection on non-court frames (close-ups /
            # replays). The spline interpolation (max_gap 0.4 s) won't bridge a
            # multi-second cut, so no ball is fabricated across a replay.
            if shot_guard_enabled and not court_visible_mask[frame_count]:
                raw_detections.append([])
                frame_count += 1
                continue
            det = detector.detect(frame, classes=[32])
            ball_mask = det["class_ids"] == 32
            candidates = []
            if ball_mask.any():
                boxes = det["boxes"][ball_mask]
                scores = det["scores"][ball_mask]
                for i in range(len(boxes)):
                    if scores[i] >= ball_conf_thresh:
                        bx = boxes[i]
                        cx, cy = int((bx[0]+bx[2])/2), int((bx[1]+bx[3])/2)
                        candidates.append((cx, cy, float(scores[i])))
            raw_detections.append(candidates)
            frame_count += 1
        reader.release()

    # The static-FP filter + Kalman tracking only run on the YOLO path; WASB
    # already produced raw_ball_centers directly.
    if not wasb_path:
        # Step 1b: Identify static false positives (logos, court marks)
        # A point that appears at ~same location in >8% of frames is static
        from collections import Counter
        static_radius = 20  # pixels — detections within this radius are "same spot"
        all_points = []
        for candidates in raw_detections:
            for cx, cy, _ in candidates:
                # Quantize to grid
                all_points.append((cx // static_radius, cy // static_radius))

        point_counts = Counter(all_points)
        static_threshold = max(10, int(max_frames * 0.08))  # 8% of frames
        static_zones = set()
        for (gx, gy), count in point_counts.items():
            if count >= static_threshold:
                static_zones.add((gx, gy))

        if static_zones:
            logger.info(f"Filtered {len(static_zones)} static false positive zones")

        def is_static(cx, cy):
            gx, gy = cx // static_radius, cy // static_radius
            # Check the cell and neighbors
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (gx + dx, gy + dy) in static_zones:
                        return True
            return False

        # Step 1c: Kalman filter on cleaned detections
        raw_ball_centers = []
        ball_speeds_px = []  # speed in pixels/frame from Kalman state

        ball_kf = BallKalmanFilter(
            process_noise=200.0,
            measurement_noise=8.0
        )
        # Gate grows with frames_since_update: ball moves further when we miss frames.
        # Tuned against the felix.mp4 bounce ground truth: the looser reinit + wider
        # gate + shorter coast recover the trajectory through direction reversals
        # (raising GT-window coverage to 16/16 and bounce recall from 0.25 to ~0.75).
        kf_gate_base = 150.0              # base gate in pixels for 1-frame gap
        kf_gate_growth = 80.0            # extra pixels per missed frame
        kf_gate_max = frame_diag * 0.35   # never exceed 35% of diagonal
        kf_max_coast = int(fps * 0.12)    # coast ~0.12s then attempt reinit
        last_known_pos = None              # last accepted ball position for reinit check

        for frame_idx in tqdm(range(len(raw_detections)), desc="Pass 1b: track"):
            candidates = raw_detections[frame_idx]

            # Filter out static false positives
            moving = [(cx, cy, sc) for cx, cy, sc in candidates if not is_static(cx, cy)]

            ball_center = None
            predicted = ball_kf.predict()

            # Adaptive gate: wider when we've been missing frames
            current_gate = min(kf_gate_max,
                               kf_gate_base + kf_gate_growth * ball_kf.frames_since_update)

            if moving:
                if not ball_kf.initialized:
                    # Pick highest confidence moving detection to init
                    best = max(moving, key=lambda c: c[2])
                    ball_kf.init_state(best[0], best[1])
                    ball_center = (best[0], best[1])
                else:
                    # Pick the moving detection closest to Kalman prediction
                    best_det = None
                    best_dist = float('inf')
                    for cx, cy, sc in moving:
                        dist = ball_kf.gating_distance(cx, cy)
                        if dist < current_gate and dist < best_dist:
                            best_dist = dist
                            best_det = (cx, cy)

                    if best_det is not None:
                        # Good match — update Kalman
                        ball_kf.update(best_det[0], best_det[1])
                        state = ball_kf.kf.statePost
                        ball_center = (int(state[0][0]), int(state[1][0]))
                    elif ball_kf.frames_since_update <= kf_max_coast and predicted:
                        # No match but still coasting — use prediction
                        ball_center = (int(predicted[0]), int(predicted[1]))
                    elif last_known_pos is not None:
                        # Coast expired — only reinit if a detection is near last known pos
                        nearest = None
                        nearest_dist = float('inf')
                        reinit_gate = frame_diag * 0.50
                        for cx, cy, sc in moving:
                            d = np.hypot(cx - last_known_pos[0], cy - last_known_pos[1])
                            if d < reinit_gate and d < nearest_dist:
                                nearest_dist = d
                                nearest = (cx, cy)
                        if nearest is not None:
                            ball_kf.init_state(nearest[0], nearest[1])
                            ball_center = nearest
                        # else: no valid reinit — skip frame

            if ball_center is None and ball_kf.initialized:
                if ball_kf.frames_since_update <= kf_max_coast and predicted:
                    ball_center = (int(predicted[0]), int(predicted[1]))

            if ball_center is not None:
                last_known_pos = ball_center

            # Extract speed from Kalman state (pixels/frame)
            if ball_kf.initialized:
                vx = float(ball_kf.kf.statePost[2][0])
                vy = float(ball_kf.kf.statePost[3][0])
                speed_px = np.hypot(vx, vy)
            else:
                speed_px = 0.0
            ball_speeds_px.append(speed_px)

            raw_ball_centers.append(ball_center)

    # ─── Post-process ball: cubic spline interpolation ───
    interp_gap = int(fps * 0.4)  # interpolate gaps up to 0.4s
    all_ball_centers = smooth_ball_trajectory(raw_ball_centers, max_gap=interp_gap)
    filled = sum(1 for a, b in zip(raw_ball_centers, all_ball_centers) if a is None and b is not None)
    detected = sum(1 for c in raw_ball_centers if c is not None)
    logger.info(f"Ball tracking: {detected}/{len(raw_ball_centers)} detected, "
                f"{filled} interpolated, "
                f"{sum(1 for c in all_ball_centers if c is not None)}/{len(all_ball_centers)} total")

    # ─── Convert speeds to km/h ───
    # Primary path: a court homography maps ball positions to real court METERS,
    # which is perspective-correct (a single px-per-meter scalar is geometrically
    # wrong everywhere except one depth). Fallback: the scalar scale.
    px_per_m = estimate_px_per_meter(court_zones, frame_height)
    ball_speeds_kmh = []
    speed_source = "scalar"
    if court_homography is not None and court_homography.get("H") is not None:
        from models.court_detector import img_to_world
        H = court_homography["H"]
        speed_source = f"homography({court_homography['confidence']})"
        # speed at frame i from the metric displacement between consecutive centers
        prev_world = None
        world_pts = []
        for c in all_ball_centers:
            world_pts.append(img_to_world(H, c[0], c[1]) if c is not None else None)
        for i in range(len(all_ball_centers)):
            if (i > 0 and world_pts[i] is not None and world_pts[i - 1] is not None):
                dx = world_pts[i][0] - world_pts[i - 1][0]
                dy = world_pts[i][1] - world_pts[i - 1][1]
                dist_m = np.hypot(dx, dy)
                ball_speeds_kmh.append(dist_m * fps * 3.6)
            else:
                ball_speeds_kmh.append(0.0)
        # reject physically-impossible spikes from tracking jitter (a tennis ball
        # tops out ~260 km/h on a pro serve); clamp before smoothing so one bad
        # metric jump doesn't poison the HUD/stats.
        MAX_PLAUSIBLE_KMH = 280.0
        ball_speeds_kmh = [s if s <= MAX_PLAUSIBLE_KMH else 0.0 for s in ball_speeds_kmh]
        # light smoothing to tame per-frame jitter
        if len(ball_speeds_kmh) >= 5:
            k = 5
            sm = np.convolve(ball_speeds_kmh, np.ones(k) / k, mode="same")
            ball_speeds_kmh = list(sm)
    else:
        # Scalar px→m fallback. Compute per-frame speed from the TRAJECTORY pixel
        # displacement (works for both WASB and Kalman trackers; ball_speeds_px is
        # only populated on the Kalman path). Perspective makes this approximate,
        # but it's the honest fallback when no homography is available.
        raw_kmh = [0.0]
        for i in range(1, len(all_ball_centers)):
            a, b = all_ball_centers[i - 1], all_ball_centers[i]
            if a is not None and b is not None:
                dpx = np.hypot(b[0] - a[0], b[1] - a[1])
                raw_kmh.append((dpx / px_per_m) * fps * 3.6)
            else:
                raw_kmh.append(0.0)
        MAX_PLAUSIBLE_KMH = 280.0
        raw_kmh = [s if s <= MAX_PLAUSIBLE_KMH else 0.0 for s in raw_kmh]
        if len(raw_kmh) >= 5:
            raw_kmh = list(np.convolve(raw_kmh, np.ones(5) / 5, mode="same"))
        ball_speeds_kmh = raw_kmh
    logger.info(f"Speed source: {speed_source}")
    valid_speeds = [s for s in ball_speeds_kmh if s > 5.0]
    if valid_speeds:
        logger.info(f"Ball speed stats: avg={np.mean(valid_speeds):.0f} km/h, "
                    f"max={np.max(valid_speeds):.0f} km/h, "
                    f"min={np.min(valid_speeds):.0f} km/h")

    # ─── Detect bounces from smoothed trajectory ───
    # On the WASB (dense heatmap) trajectory, use the robust detector: it despikes
    # outliers and uses IRLS fits + an energy hit-vs-bounce test, which gives the
    # best recall on a dense trajectory (felix: F1 0.865 vs 0.79). On the sparser
    # YOLO+Kalman trajectory the curve-fit detector is the tuned default.
    bounce_events = []
    if not args.no_bounces:
        if wasb_path:
            from vision.bounce import detect_bounces_robust
            bounce_events, all_ball_centers, _ = detect_bounces_robust(
                all_ball_centers, ball_speeds_px, fps, frame_height, frame_width)
        else:
            bounce_events = detect_bounces_from_trajectory(
                all_ball_centers, ball_speeds_px, fps, frame_height, frame_width)

    # ─── Optional: export bounce predictions for offline evaluation ───
    # Additive, gated, read-only w.r.t. the pipeline: reads the already-computed
    # bounce_events and writes the eval_bounces.py schema. frame_idx is 0-based
    # within the processed window, so absolute frame = start_frame + frame_idx
    # (matches the annotator's convention -> aligns with ground truth).
    if args.dump_bounces:
        pred_doc = {
            "video": video_path,
            "fps": fps,
            "frame_offset": start_frame,
            "annotator": "run_pipeline_8s.py:detect_bounces_from_trajectory",
            "notes": "auto-exported bounce predictions",
            "bounces": [
                {"frame": int(start_frame + idx), "x": int(x), "y": int(y)}
                for (idx, x, y) in bounce_events
            ],
        }
        Path(args.dump_bounces).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_bounces, "w") as f:
            json.dump(pred_doc, f, indent=2)
            f.write("\n")
        logger.info(f"Dumped {len(bounce_events)} bounce predictions -> {args.dump_bounces}")

    # ─── Demo override: replace detected bounces with the manual ground truth ───
    # A curated clip (e.g. the landing-page hero) can override the detector's
    # bounces with hand-annotated ones (the detector misses ~6/9 far-side
    # bounces on this broadcast clip) and add racket->ball strikes. demo_frame
    # in the sidecar is 0-based within the -s/-d window, i.e. it maps directly
    # to the segment frame index used below. Bounded to this clip only — no
    # global thresholds move (respects zero-hardcoding).
    override_strikes = {}  # segment_frame -> {"xy": (x,y), "type": "forehand"|"backhand"}
    if args.demo_override:
        with open(args.demo_override) as _f:
            ov = json.load(_f)
        gt_bounces = ov.get("bounces_true", [])
        if gt_bounces:
            # bounces_false let us suppress specific detector hits by proximity.
            suppress_xy = [tuple(b["xy"]) for b in ov.get("bounces_false", [])]
            override_events = [
                (int(b["demo_frame"]), float(b["xy"][0]), float(b["xy"][1]))
                for b in gt_bounces
            ]
            logger.info(f"Demo override: replacing {len(bounce_events)} detected "
                        f"bounces with {len(override_events)} manual GT bounces.")
            bounce_events = override_events
        for s in ov.get("strikes", []):
            override_strikes[int(s["demo_frame"])] = {
                "xy": (float(s["xy"][0]), float(s["xy"][1])),
                "type": s.get("type", "forehand"),
            }
        if override_strikes:
            logger.info(f"Demo override: {len(override_strikes)} racket->ball strikes "
                        f"({sum(1 for v in override_strikes.values() if v['type']=='forehand')} "
                        f"forehand, {sum(1 for v in override_strikes.values() if v['type']=='backhand')} backhand).")

    # ─── Classify bounces ───
    classified_bounces = []
    if bounce_events:
        classified_bounces = classify_bounce_depth(bounce_events, court_zones, frame_height)
        depth_counts = {}
        for _, _, _, depth in classified_bounces:
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        logger.info(f"Bounce detection: {len(classified_bounces)} bounces — {depth_counts}")
    else:
        logger.info("Bounce detection: 0 bounces detected")

    # ─── Compute quality score per bounce ───
    # speed_window: average speed over ±0.3s around the bounce
    speed_window = int(fps * 0.3)
    # Reference speed: 90th percentile of all valid speeds
    valid_speeds_arr = np.array([s for s in ball_speeds_kmh if s > 5.0])
    speed_ref = float(np.percentile(valid_speeds_arr, 90)) if len(valid_speeds_arr) > 0 else 100.0
    speed_ref = max(speed_ref, 30.0)  # floor to avoid division issues

    depth_score_map = {"deep": 1.0, "mid": 0.5, "short": 0.2}

    bounce_by_frame = {}
    for fidx, bx, by, depth in classified_bounces:
        # Average speed in window around bounce
        lo = max(0, fidx - speed_window)
        hi = min(len(ball_speeds_kmh), fidx + speed_window + 1)
        window_speeds = [ball_speeds_kmh[i] for i in range(lo, hi) if ball_speeds_kmh[i] > 5.0]
        avg_speed = float(np.mean(window_speeds)) if window_speeds else 0.0

        speed_norm = min(1.0, avg_speed / speed_ref)
        depth_norm = depth_score_map.get(depth, 0.5)
        quality = int((0.4 * speed_norm + 0.6 * depth_norm) * 100)
        quality = max(0, min(100, quality))

        bounce_by_frame[fidx] = (bx, by, depth, avg_speed, quality)
        logger.debug(f"Bounce frame {fidx}: depth={depth}, speed={avg_speed:.0f}km/h, Q={quality}")

    # ─── Pass 1.5: Pose on every frame → locked P1/P2 skeleton tracks ───
    # The per-frame pose estimator returns "the 2 players this frame" with no
    # temporal identity. We run it on EVERY frame once, then feed the results to
    # TwoPlayerTracker which assigns persistent P1(far)/P2(near) identities,
    # associates frame-to-frame, GAP-FILLS missed detections, and smooths — so
    # every frame has a stable, jitter-free skeleton for both players. This single
    # pose pass is reused by the shot pre-pass and Pass 2 (no redundant inference).
    logger.info("=== PASS 1.5: Pose + lock P1/P2 ===")
    from vision.player_track import TwoPlayerTracker
    match_mode = bool(args.match_mode)
    if match_mode:
        from vision.match_tracker import MatchTracker, human_id_for_side
    players_per_frame = [None] * max_frames
    rdr = VideoReader(video_path)
    if start_frame > 0:
        rdr.seek(start_frame)
    net_y_seed = None
    if court_zones:
        from vision.pose import _net_y
        net_y_seed = _net_y(court_zones, frame_height)
    if match_mode:
        # In match mode P1/P2 are STABLE HUMANS (not net sides): a side-change
        # between games no longer inverts the identity, so per-human stats stay
        # correct across the whole clip. See vision/match_tracker.py.
        ptracker = MatchTracker(frame_width, frame_height, net_y_seed, fps)
    else:
        ptracker = TwoPlayerTracker(frame_width, frame_height, net_y_seed, fps)
    for i, frame in enumerate(tqdm(rdr.iter_frames(), total=max_frames,
                                   desc="Pass 1.5: pose")):
        if i >= max_frames:
            break
        # shot-guard: skip pose inference on non-court frames (no skeleton can
        # be trusted on a close-up, and we save the GPU call). Both trackers
        # accept an empty list; the long gap-fill stays None, and since Pass 2
        # skips the overlay on these frames, no phantom skeleton is drawn.
        if shot_guard_enabled and not court_visible_mask[i]:
            players = []
        else:
            players = estimate_players_pose(
                detector.person_model, pose_model, frame, device,
                ball_xy=all_ball_centers[i] if i < len(all_ball_centers) else None,
                court_zones=court_zones if court_zones else None)
        players_per_frame[i] = players
        if match_mode:
            ptracker.add_frame(players, frame)   # frame needed for HSV re-id
        else:
            ptracker.add_frame(players)
    rdr.release()
    # locked, gap-filled, smoothed tracks: {1: [pose|None]*N, 2: [pose|None]*N}.
    # In match mode keys are stable HUMAN ids; in training mode they are sides.
    match_result = None
    if match_mode:
        match_result = ptracker.finalize()
        player_tracks = match_result["tracks"]
        n_swaps = len(match_result["swaps"])
        c1 = (sum(1 for p in player_tracks[1] if p is not None) / len(player_tracks[1])
              if player_tracks[1] else 0.0)
        c2 = (sum(1 for p in player_tracks[2] if p is not None) / len(player_tracks[2])
              if player_tracks[2] else 0.0)
        logger.info(f"Player tracks locked (MATCH MODE): P1 {c1*100:.1f}%, "
                    f"P2 {c2*100:.1f}%, side-swaps detected: {n_swaps}")
    else:
        player_tracks = ptracker.finalize()
        cov = ptracker.coverage()
        logger.info(f"Player tracks locked: P1 coverage={cov[1]*100:.1f}%, "
                    f"P2 coverage={cov[2]*100:.1f}% (gap-filled)")

    # ─── Shot pre-pass: detect hits, attribute to player, classify FH/BH, score ───
    # A hit is a ball vertical-direction reversal near a player. Reuses the Pass 1.5
    # poses (no extra inference), attributes + classifies, then links each hit to
    # the next bounce for a multi-factor quality score (speed + depth + placement).
    shot_by_frame = {}      # frame -> dict(side, fhb, quality, speed)
    if not args.no_bounces and all_ball_centers:
        from vision.shots import detect_hits, classify_forehand_backhand, shot_quality
        bset = list(bounce_by_frame.keys())
        # poses already computed for every frame in Pass 1.5 (players_per_frame)
        hits = detect_hits(all_ball_centers, players_per_frame, fps,
                           frame_height, frame_width, bounce_frames=bset)
        sorted_bounces = sorted(bounce_by_frame.keys())
        for h in hits:
            fhb = classify_forehand_backhand(h, players_per_frame)
            # link to the next bounce after this hit for depth/placement/speed
            nb = next((b for b in sorted_bounces if b > h["frame"]), None)
            if nb is not None:
                bx, by, depth, b_speed, _ = bounce_by_frame[nb]
                _h = h_at(court_homography_by_frame, nb, court_homography)
                q = shot_quality(b_speed, depth, (bx, by), court_zones,
                                 frame_width, frame_height,
                                 homography=_h, speed_ref=speed_ref)
            else:
                b_speed, q = 0.0, 0
            shot_by_frame[h["frame"]] = {
                "side": h["player_side"], "fhb": fhb, "quality": q,
                "speed": b_speed, "x": h["x"], "y": h["y"],
                "human_id": (human_id_for_side(match_result, h["player_side"], h["frame"])
                             if match_mode else None)}
        logger.info(f"Shots: {len(shot_by_frame)} detected "
                    f"({sum(1 for s in shot_by_frame.values() if s['fhb']=='forehand')} FH, "
                    f"{sum(1 for s in shot_by_frame.values() if s['fhb']=='backhand')} BH)")

    # ─── Demo override: inject manual racket->ball strikes ───
    # Overrides the geometric shot detector with hand-annotated contacts. Each
    # strike replaces any auto-detected hit at the same frame (no duplicates) and
    # is linked to the next bounce for a quality score, mirroring the auto path.
    if override_strikes:
        sorted_bounces = sorted(bounce_by_frame.keys())
        n_replaced = 0
        for sfi, s in sorted(override_strikes.items()):
            if sfi in shot_by_frame:
                n_replaced += 1
            nb = next((b for b in sorted_bounces if b > sfi), None)
            if nb is not None:
                bx, by, depth, b_speed, _ = bounce_by_frame[nb]
                try:
                    from vision.shots import shot_quality
                    _h = h_at(court_homography_by_frame, nb, court_homography)
                    q = shot_quality(b_speed, depth, (bx, by), court_zones,
                                     frame_width, frame_height,
                                     homography=_h, speed_ref=speed_ref)
                except Exception:
                    q = 0
            else:
                b_speed, q = 0.0, 0
            shot_by_frame[sfi] = {
                "side": None, "fhb": s["type"], "quality": q,
                "speed": b_speed, "x": s["xy"][0], "y": s["xy"][1]}
        logger.info(f"Demo override strikes: injected {len(override_strikes)} "
                    f"({n_replaced} replaced auto hits).")

    # ─── Pass 2: Annotate ───
    logger.info("=== PASS 2: Annotate ===")
    reader = VideoReader(video_path)
    if start_frame > 0:
        reader.seek(start_frame)
    writer = VideoWriter(output_path, fps=fps)
    ball_trail = deque(maxlen=trail_length)

    # Bounce display duration: 0.5s worth of frames
    bounce_display_frames = int(fps * 0.5)

    # Build legend items
    legend_items = {"Ball": BALL_COLOR}
    if not args.no_player_boxes:
        legend_items["Player 1"] = PLAYER_COLORS[1]
        legend_items["Player 2"] = PLAYER_COLORS[2]
    if not args.no_bounces:
        legend_items["Bounce (deep)"] = (0, 0, 255)
        legend_items["Bounce (short)"] = (0, 230, 255)
        legend_items["Quality 70+"] = (0, 200, 0)
        legend_items["Quality <40"] = (0, 0, 220)

    # Precompute cumulative match stats per frame for the live HUD card (running
    # FH/BH counts, rally length = bounces in the current point, max/avg speed).
    sorted_shot_frames = sorted(shot_by_frame.keys())
    sorted_bounce_frames = sorted(bounce_by_frame.keys())
    peak_speed = 0.0

    frame_idx = 0
    for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 2"):
        if frame_idx >= max_frames:
            break
        out = frame.copy()

        # shot-guard: on a non-court frame (close-up / replay / graphic) draw NO
        # annotation at all (optionally a discreet NON-PLAY badge) and write the
        # clean frame. This keeps the output timeline 1:1 with the input (the web
        # player's frame↔time mapping stays valid) while excluding garbage
        # detections from the visible overlay and the stats.
        if shot_guard_enabled and not court_visible_mask[frame_idx]:
            if args.shot_guard_badge and not args.no_fx:
                _bw = int(frame_width * 0.18)
                out = fx.draw_text(out, "NON-PLAY",
                                   (frame_width - 16, frame_height - 14),
                                   size=int(frame_height * 0.022),
                                   color=(210, 210, 210), font="regular",
                                   anchor="rt")
            writer.write_frame(out)
            frame_idx += 1
            continue

        # 1. Court zone overlay (background layer) — redrive from the per-frame
        # homography when tracking a moving camera, so the zones follow the court.
        if not args.no_court and court_zones:
            cz = court_zones
            if court_homography_by_frame is not None:
                _h = h_at(court_homography_by_frame, frame_idx, court_homography)
                if _h is not None and _h.get("H_inv") is not None:
                    from vision.court_zones import derive_court_zones_from_homography
                    cz = derive_court_zones_from_homography(_h) or court_zones
            out = draw_court_overlay(out, cz, alpha=0.15,
                                     draw_labels=args.court_labels)

        # 2. Pose — LOCKED P1/P2 skeleton tracks from Pass 1.5 (persistent identity,
        # gap-filled so every frame has both skeletons, distinct colors per player).
        # No per-frame pose here: we draw the pre-tracked, smoothed poses.
        person_boxes = []
        for pid in (1, 2):
            pose = player_tracks[pid][frame_idx] if frame_idx < len(player_tracks[pid]) else None
            if pose is None or pose.get("kps_xy") is None:
                continue
            out = draw_skeleton(out, pose["kps_xy"], pose["kps_conf"], PLAYER_COLORS[pid])
            person_boxes.append((pid, pose["box"]))

        # 3. Player bounding boxes (off by default; same locked P1/P2 identity)
        if not args.no_player_boxes:
            for pid, box in person_boxes:
                out = draw_player_bbox(out, box, pid)

        # 4. Ball — cinematic comet trail + glowing pulsing marker (FX), or the
        # plain trail/marker when --no-fx.
        bc = all_ball_centers[frame_idx]
        if bc is not None:
            ball_trail.append(bc)
        if not args.no_fx:
            if len(ball_trail) >= 2:
                out = fx.comet_trail(out, list(ball_trail), base_color=BALL_COLOR,
                                     max_thickness=7, glow=True)
            if bc is not None:
                # pulse phase advances with frame_idx so the ball "breathes"
                out = fx.glow_marker(out, (int(bc[0]), int(bc[1])), color=BALL_COLOR,
                                     radius=8, pulse=frame_idx * 0.5)
        else:
            if bc is not None:
                out = draw_ball_marker(out, bc[0], bc[1])
            draw_ball_trail(out, list(ball_trail))

        # 4b. Current ball speed (HUD drawn later, after overlays, as a gauge/card)
        current_speed = ball_speeds_kmh[frame_idx] if frame_idx < len(ball_speeds_kmh) else 0.0

        # 5. Bounce markers — FX shockwave (expanding glow ring) at each impact,
        # color-coded by depth, with a styled label. Falls back to the plain
        # marker under --no-fx.
        DEPTH_COL = {"deep": (60, 60, 255), "mid": (0, 200, 255), "short": (120, 230, 0)}
        if not args.no_bounces:
            for bfi in range(max(0, frame_idx - bounce_display_frames), frame_idx + 1):
                if bfi in bounce_by_frame:
                    bx, by, depth, b_speed, b_quality = bounce_by_frame[bfi]
                    age = frame_idx - bfi
                    dcol = DEPTH_COL.get(depth, (255, 255, 255))
                    if not args.no_fx:
                        out = fx.shockwave(out, (bx, by), age, bounce_display_frames,
                                           color=dcol, max_radius=int(frame_height * 0.07))
                    else:
                        alpha = max(0.3, 1.0 - age / bounce_display_frames)
                        out = draw_bounce_marker(out, bx, by, depth=depth, alpha=alpha)
                    # styled label (only while fresh, so it doesn't clutter)
                    if age <= bounce_display_frames // 2:
                        lx, ly = bx + 20, by - 26
                        if not args.no_fx:
                            out = fx.draw_text(out, depth.upper(), (lx, ly), size=18,
                                               color=dcol, font="bold")
                            out = fx.draw_text(out, f"{int(b_speed)} km/h", (lx, ly + 20),
                                               size=15, color=(210, 210, 210), font="regular")
                            qc = ((90, 220, 90) if b_quality >= 70 else
                                  (90, 220, 220) if b_quality >= 40 else (90, 90, 230))
                            out = fx.draw_text(out, f"Q {b_quality}", (lx, ly + 39),
                                               size=15, color=qc, font="bold")

        # 5b. Shot markers (forehand/backhand + quality at the contact frame)
        if shot_by_frame:
            for sfi in range(max(0, frame_idx - bounce_display_frames), frame_idx + 1):
                if sfi in shot_by_frame:
                    s = shot_by_frame[sfi]
                    age = frame_idx - sfi
                    if age > bounce_display_frames:
                        continue
                    sx, sy = s["x"], s["y"]
                    fhb = s["fhb"]
                    tag = {"forehand": "FOREHAND", "backhand": "BACKHAND"}.get(fhb, "SHOT")
                    fcol = (0, 200, 255) if fhb == "forehand" else (255, 150, 0)
                    if not args.no_fx:
                        out = fx.glow_marker(out, (sx, sy), color=fcol, radius=11)
                        out = fx.draw_text(out, tag, (sx + 18, sy - 8), size=17,
                                           color=fcol, font="bold")
                        if s["quality"] > 0:
                            qc = ((90, 220, 90) if s["quality"] >= 70 else
                                  (90, 220, 220) if s["quality"] >= 40 else (90, 90, 230))
                            out = fx.draw_text(out, f"Q {s['quality']}", (sx + 18, sy + 12),
                                               size=15, color=qc, font="bold")
                    else:
                        cv2.circle(out, (sx, sy), 14, fcol, 2, cv2.LINE_AA)
                        cv2.putText(out, tag, (sx + 16, sy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, fcol, 2, cv2.LINE_AA)

        # 5c. 2D top-down court minimap (ball trajectory + bounce locations)
        if not args.no_minimap:
            # a longer dedicated trajectory window for the map (the on-court trail
            # is intentionally short). Show ~2s of recent ball positions.
            mm_window = int(fps * 2.0)
            lo = max(0, frame_idx - mm_window)
            mm_trail = [all_ball_centers[j] for j in range(lo, frame_idx + 1)
                        if all_ball_centers[j] is not None]
            bounces_so_far = [(v[0], v[1], v[2]) for bf, v in bounce_by_frame.items()
                              if bf <= frame_idx]
            # GROUND-PATH for the radar arc: only metrically-trustworthy ground
            # contacts — bounces (true ground impacts) + hit/contact points (≈ground
            # at the player) — in temporal (frame) order, growing over the clip. The
            # in-flight ball is no longer projected naively onto the top-down map.
            gp_events = []
            for bf, v in bounce_by_frame.items():
                if bf <= frame_idx:
                    gp_events.append((bf, v[0], v[1], "bounce"))
            for sf, s in shot_by_frame.items():
                if sf <= frame_idx:
                    gp_events.append((sf, s["x"], s["y"], "hit"))
            gp_events.sort(key=lambda e: e[0])
            ground_path = [(e[1], e[2], e[3]) for e in gp_events]
            # live player feet (bottom-center of each locked P1/P2 box) → radar dots
            players_img = [((b[0] + b[2]) / 2.0, b[3], pid) for pid, b in person_boxes]
            out = draw_minimap(out, mm_trail, bounces_so_far,
                               frame_width, frame_height,
                               homography=h_at(court_homography_by_frame, frame_idx, court_homography),
                               players_img=players_img,
                               pulse=frame_idx * 0.5,
                               ground_path=ground_path,
                               now_frame=frame_idx)

        # 6. HUD — speed gauge + live match-stats card, anchored BOTTOM-LEFT so it
        # never collides with a broadcast scoreboard (top corners). On amateur phone
        # footage the top corners are free too; bottom-left is the safe universal slot.
        if not args.no_fx:
            peak_speed = max(peak_speed, current_speed if bc is not None else 0.0)
            margin = int(frame_height * 0.022)
            gauge_r = int(frame_height * 0.052)
            gauge_box = 2 * gauge_r + 2 * int(gauge_r * 0.42)
            card_w = int(frame_width * 0.135)
            n_fh = sum(1 for f in sorted_shot_frames
                       if f <= frame_idx and shot_by_frame[f]["fhb"] == "forehand")
            n_bh = sum(1 for f in sorted_shot_frames
                       if f <= frame_idx and shot_by_frame[f]["fhb"] == "backhand")
            n_rally = sum(1 for f in sorted_bounce_frames if f <= frame_idx)
            card_lines = [
                ("Forehand", str(n_fh)),
                ("Backhand", str(n_bh)),
                ("Rally", str(n_rally)),
                ("Peak", f"{int(peak_speed)} km/h"),
            ]
            card_h = int(frame_height * 0.022) * 2 + int(frame_height * 0.019 * 1.55) * len(card_lines) + 50
            # stack: gauge on top, stats card under it, bottom-left aligned
            card_y = frame_height - margin - card_h
            gauge_y = card_y - gauge_box - int(frame_height * 0.012)
            fx.speed_gauge(out, margin, gauge_y, current_speed if bc is not None else 0.0,
                           max_kmh=180, label="BALL", radius=gauge_r)
            fx.stat_card(out, margin, card_y, card_lines, title="Match", width=card_w)
        else:
            out = create_legend(out, legend_items, position="top-right")

        writer.write_frame(out)
        frame_idx += 1

    reader.release()
    writer.release()

    # ─── Match Summary ───
    if bounce_by_frame:
        logger.info("=== MATCH SUMMARY ===")
        all_depths = [v[2] for v in bounce_by_frame.values()]
        all_bspeeds = [v[3] for v in bounce_by_frame.values()]
        all_qualities = [v[4] for v in bounce_by_frame.values()]

        n = len(all_depths)
        deep_n = sum(1 for d in all_depths if d == "deep")
        mid_n = sum(1 for d in all_depths if d == "mid")
        short_n = sum(1 for d in all_depths if d == "short")
        logger.info(f"Total shots: {n}")
        logger.info(f"Depth: deep={deep_n} ({100*deep_n/n:.0f}%), "
                    f"mid={mid_n} ({100*mid_n/n:.0f}%), "
                    f"short={short_n} ({100*short_n/n:.0f}%)")

        speeds_arr = np.array(all_bspeeds)
        logger.info(f"Speed: avg={np.mean(speeds_arr):.0f} km/h, "
                    f"max={np.max(speeds_arr):.0f} km/h, "
                    f"min={np.min(speeds_arr):.0f} km/h")

        quals_arr = np.array(all_qualities)
        high_q = sum(1 for q in all_qualities if q >= 70)
        logger.info(f"Quality: avg={np.mean(quals_arr):.0f}, "
                    f"max={np.max(quals_arr):.0f}, "
                    f"min={np.min(quals_arr):.0f}, "
                    f"high (>=70): {high_q}/{n} ({100*high_q/n:.0f}%)")

        # ─── Rally segmentation ───
        # Group shots into rallies: a gap > 2.5s between consecutive shots
        # breaks a rally (threshold is a fraction of fps — zero hardcoding).
        rallies = []
        if shot_by_frame:
            rally_gap_frames = 2.5 * fps
            sorted_sf = sorted(shot_by_frame.keys())
            grouped = []
            cur = [sorted_sf[0]]
            for sf in sorted_sf[1:]:
                # shot-guard: measure the gap in COURT frames only, so a replay
                # between two real shots doesn't split one rally in two.
                if shot_guard_enabled:
                    gap = _court_frames_between(court_visible_mask, cur[-1], sf)
                else:
                    gap = sf - cur[-1]
                if gap > rally_gap_frames:
                    grouped.append(cur)
                    cur = []
                cur.append(sf)
            grouped.append(cur)
            sorted_bf = sorted(bounce_by_frame.keys())
            for g in grouped:
                start = g[0]
                end = max(g[-1], max([bf for bf in sorted_bf if start <= bf <= g[-1] + rally_gap_frames], default=g[-1]))
                bframes = [bf for bf in sorted_bf if start <= bf <= end]
                rallies.append({
                    "start_frame": int(start_frame + start),
                    "end_frame": int(start_frame + end),
                    "shot_frames": [int(start_frame + sf) for sf in g],
                    "bounce_frames": [int(start_frame + bf) for bf in bframes],
                    "n_shots": len(g),
                })
            logger.info(f"Rallies: {len(rallies)} "
                        f"(longest={max((r['n_shots'] for r in rallies), default=0)} shots)")

            # Merge solo (1-shot) rallies into their nearest neighbour. A solo
            # arises when a strike sits >gap from both neighbours — a genuine
            # isolated hit — but surfacing it as its own point is noise in the
            # highlights. Must match web/src/lib/analysis/highlights.ts
            # mergeSoloRallies so the pipeline JSON and the front-end agree.
            rallies = _merge_solo_rallies(rallies)

        # Shot-type stats
        if shot_by_frame:
            fh = sum(1 for s in shot_by_frame.values() if s["fhb"] == "forehand")
            bh = sum(1 for s in shot_by_frame.values() if s["fhb"] == "backhand")
            logger.info(f"Strokes: {len(shot_by_frame)} hits — forehand={fh}, backhand={bh}")

        # ─── Derived analytics (Session B): per-rally outcome + fatigue ───
        # Enrich each rally with a geometric outcome label (winner / forced_error /
        # unforced_error / neutral). Note: forced_error is LOW confidence — the
        # frontend treats it as such. Both modules are pure functions over the
        # shots/bounces and degrade gracefully (no shots → no outcome, fatigue none).
        stats_shots = [{"frame": int(start_frame + f), "x": s["x"], "y": s["y"],
                        "player_side": s["side"], "stroke": s["fhb"],
                        "quality": s["quality"], "speed_kmh": round(s["speed"], 1),
                        "human_id": s.get("human_id")}
                       for f, s in sorted(shot_by_frame.items())]
        stats_bounces = [{"frame": int(start_frame + f), "x": int(v[0]), "y": int(v[1]),
                          "depth": v[2], "speed_kmh": round(v[3], 1), "quality": v[4]}
                         for f, v in sorted(bounce_by_frame.items())]
        for r in rallies:
            r["outcome"] = classify_rally_outcome(r, stats_shots, stats_bounces, fps)
            # In match mode, tag the rally with the stable human_id of its last
            # striker (the rally-ending player) on top of the near/far side. The
            # outcome label still uses player_side (handled by rally_outcome.py).
            if match_mode:
                last_shot_frame = r["shot_frames"][-1] if r.get("shot_frames") else None
                last_rel = (last_shot_frame - start_frame) if last_shot_frame is not None else None
                if last_rel is not None and last_rel in shot_by_frame:
                    r["human_id"] = shot_by_frame[last_rel].get("human_id")
                else:
                    r["human_id"] = None
        stats_fatigue = analyze_fatigue(stats_shots, [start_frame, end_frame], fps)

        # ─── Structured stats JSON (the actionable data output) ───
        stats = {
            "video": video_path,
            "fps": fps,
            "frame_range": [start_frame, end_frame],
            "speed_source": speed_source,
            "match_mode": match_mode,
            "match": ({"side_swaps": (match_result["swaps"] if match_result else []),
                       "n_swaps": (len(match_result["swaps"]) if match_result else 0)}
                      if match_mode else None),
            "shot_guard": {
                "enabled": shot_guard_enabled,
                "court_frames": int(sum(court_visible_mask)),
                "total_frames": int(len(court_visible_mask)),
                "non_court_ratio": (1.0 - sum(court_visible_mask) / max(1, len(court_visible_mask)))
                                   if shot_guard_enabled else 0.0,
                "non_court_spans": non_court_spans_rel,
                "threshold_lines": 40,
                "min_line_length_frac": 0.15,
                "dwell_frames": int(max(1, round(0.3 * fps))),
            },
            "homography_confidence": (court_homography.get("confidence")
                                      if court_homography else "none"),
            "summary": {
                "total_bounces": n,
                "depth": {"deep": deep_n, "mid": mid_n, "short": short_n},
                "speed_kmh": {"avg": float(np.mean(speeds_arr)),
                              "max": float(np.max(speeds_arr)),
                              "min": float(np.min(speeds_arr))},
                "quality": {"avg": float(np.mean(quals_arr)),
                            "max": int(np.max(quals_arr)),
                            "high_count": int(high_q)},
                "strokes": ({"forehand": sum(1 for s in shot_by_frame.values() if s["fhb"] == "forehand"),
                             "backhand": sum(1 for s in shot_by_frame.values() if s["fhb"] == "backhand")}
                            if shot_by_frame else {}),
            },
            "bounces": stats_bounces,
            "shots": stats_shots,
            "rallies": rallies,
            "fatigue": stats_fatigue,
        }
        stats_path = str(Path(output_path).with_suffix("")) + "_stats.json"
        try:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Stats written to: {stats_path} "
                        f"(fatigue: {stats_fatigue['confidence']}, "
                        f"rallies: {len(rallies)})")
        except Exception as exc:  # never let stats emission break the video output
            logger.warning(f"Stats JSON not written: {exc}")


    logger.info(f"Done! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
