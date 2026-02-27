"""
Tennis Match Annotation Pipeline
Pose skeletons on all detected persons + Kalman-filtered ball trail.
2-player tracking with bounding boxes, ball bounce detection, court zone overlay.

Works on any tennis match video regardless of resolution, camera angle, or players.
"""
import sys
import argparse
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
    - check_bounce(): detect vertical velocity sign change (bounce)
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
        # Bounce detection state
        self.prev_vy = 0.0
        self.frames_since_last_bounce = 999

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.frames_since_update = 0
        self.prev_vy = 0.0
        self.frames_since_last_bounce = 999

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

    def check_bounce(self, min_vy_threshold=2.0, min_frames_between=10,
                     frame_height=None, min_y_ratio=0.45):
        """
        Check if a bounce occurred by detecting vertical velocity sign change.
        A bounce is: prev_vy > threshold (going down) and current vy < -threshold (going up),
        AND the ball must be in the lower portion of the frame (below min_y_ratio).
        This avoids false bounces when a player hits the ball mid-air.

        Returns (x, y) of bounce position or None.
        """
        if not self.initialized:
            return None

        self.frames_since_last_bounce += 1
        vy = float(self.kf.statePost[3][0])

        bounce = None
        if (self.prev_vy > min_vy_threshold and
                vy < -min_vy_threshold and
                self.frames_since_last_bounce >= min_frames_between):
            x = float(self.kf.statePost[0][0])
            y = float(self.kf.statePost[1][0])
            # Only count as bounce if ball is in lower portion of frame
            if frame_height is None or y >= frame_height * min_y_ratio:
                bounce = (int(x), int(y))
                self.frames_since_last_bounce = 0

        self.prev_vy = vy
        return bounce


class PlayerTracker:
    """
    Simple 2-slot IoU-based tracker for tennis players.
    Maintains up to 2 player slots with consistent IDs.
    """

    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.slots = {}  # {player_id: box}
        self.next_id = 1

    @staticmethod
    def _iou(box_a, box_b):
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    def update(self, detected_boxes):
        """
        Match detected boxes to existing slots via IoU (greedy).
        Returns list of (player_id, box) for matched/new tracks.
        """
        if len(detected_boxes) == 0:
            return list(self.slots.items())

        # Compute IoU between all slots and detections
        slot_ids = list(self.slots.keys())
        pairs = []
        for sid in slot_ids:
            for di, dbox in enumerate(detected_boxes):
                iou = self._iou(self.slots[sid], dbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, sid, di))
        pairs.sort(reverse=True)

        matched_slots = set()
        matched_dets = set()
        for iou, sid, di in pairs:
            if sid in matched_slots or di in matched_dets:
                continue
            self.slots[sid] = list(detected_boxes[di])
            matched_slots.add(sid)
            matched_dets.add(di)

        # Assign unmatched detections to new slots (max 2 total)
        for di, dbox in enumerate(detected_boxes):
            if di in matched_dets:
                continue
            if len(self.slots) < 2:
                pid = self.next_id
                self.next_id += 1
                self.slots[pid] = list(dbox)

        return list(self.slots.items())


def smooth_ball_trajectory(raw_centers, max_gap=10):
    """
    Post-process raw ball centers using cubic spline interpolation.

    Args:
        raw_centers: list of (x,y) or None for each frame
        max_gap: maximum number of consecutive missing frames to interpolate

    Returns:
        list of (x,y) or None — same length, with gaps filled
    """
    n = len(raw_centers)
    if n < 4:
        return raw_centers

    # Collect known positions
    known_idx = []
    known_x = []
    known_y = []
    for i, c in enumerate(raw_centers):
        if c is not None:
            known_idx.append(i)
            known_x.append(c[0])
            known_y.append(c[1])

    if len(known_idx) < 4:
        return raw_centers

    # Build cubic splines
    t = np.array(known_idx, dtype=float)
    cs_x = CubicSpline(t, known_x, bc_type='natural')
    cs_y = CubicSpline(t, known_y, bc_type='natural')

    result = list(raw_centers)  # copy
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        # Find the gap: consecutive None frames
        gap_start = i
        while i < n and result[i] is None:
            i += 1
        gap_end = i  # exclusive

        gap_len = gap_end - gap_start
        if gap_len > max_gap:
            continue  # gap too long, leave as None

        # Only interpolate if we have known points on both sides
        if gap_start == 0 or gap_end >= n:
            continue
        if raw_centers[gap_start - 1] is None or (gap_end < n and raw_centers[gap_end] is None):
            continue

        # Interpolate
        for j in range(gap_start, gap_end):
            result[j] = (int(cs_x(j)), int(cs_y(j)))

    return result


def detect_bounces_from_trajectory(ball_centers, ball_speeds_px, fps, frame_height):
    """
    Detect ball bounces from the smoothed trajectory.

    A bounce is a local maximum of y (lowest point of the arc in image coords)
    where the ball is on the court surface (lower portion of frame) and
    horizontal direction is preserved (ruling out player hits).

    Args:
        ball_centers: list of (x,y) or None per frame (smoothed trajectory)
        ball_speeds_px: list of speed in px/frame per frame
        fps: video frame rate
        frame_height: frame height in pixels

    Returns:
        list of (frame_idx, x, y)
    """
    if len(ball_centers) < 5:
        return []

    # Parameters derived from fps/frame size
    min_gap = int(fps * 0.4)          # minimum frames between bounces
    half_window = max(3, int(fps * 0.15))  # window to check local max (~150ms)
    # Court surface zone: bounces happen in the lower portion of the frame
    # but not at the very bottom (that's below the court)
    min_y_ratio = 0.35   # ball must be below 35% of frame (excludes sky/stands)
    max_y_ratio = 0.92   # ball must be above 92% (excludes below-court area)
    speed_window = max(3, int(fps * 0.2))  # window to measure speed change

    bounces = []
    last_bounce_frame = -min_gap

    for i in range(half_window, len(ball_centers) - half_window):
        if ball_centers[i] is None:
            continue

        cx, cy = ball_centers[i]
        y_ratio = cy / frame_height

        # Must be on the court surface area
        if y_ratio < min_y_ratio or y_ratio > max_y_ratio:
            continue

        # Check if this is a local maximum of y (= lowest point physically)
        is_local_max = True
        for j in range(i - half_window, i + half_window + 1):
            if j == i:
                continue
            if ball_centers[j] is None:
                continue
            if ball_centers[j][1] > cy:
                is_local_max = False
                break

        if not is_local_max:
            continue

        # Minimum gap between bounces
        if (i - last_bounce_frame) < min_gap:
            continue

        # Check that ball was coming down before and going up after
        # (y increasing then decreasing in image coords)
        before_y = []
        for j in range(max(0, i - half_window * 2), i):
            if ball_centers[j] is not None:
                before_y.append(ball_centers[j][1])
        after_y = []
        for j in range(i + 1, min(len(ball_centers), i + half_window * 2 + 1)):
            if ball_centers[j] is not None:
                after_y.append(ball_centers[j][1])

        if len(before_y) < 2 or len(after_y) < 2:
            continue

        # Ball should be descending before (y increasing) and ascending after (y decreasing)
        coming_down = before_y[-1] > before_y[0]  # y increases = ball descending
        going_up = after_y[-1] < after_y[0]        # y decreases = ball ascending

        if not (coming_down and going_up):
            continue

        # Filter out player hits: check horizontal direction
        # At a bounce, horizontal direction is preserved
        # At a hit, horizontal direction reverses
        before_x = []
        for j in range(max(0, i - half_window), i):
            if ball_centers[j] is not None:
                before_x.append(ball_centers[j][0])
        after_x = []
        for j in range(i + 1, min(len(ball_centers), i + half_window + 1)):
            if ball_centers[j] is not None:
                after_x.append(ball_centers[j][0])

        if len(before_x) >= 2 and len(after_x) >= 2:
            vx_before = before_x[-1] - before_x[0]
            vx_after = after_x[-1] - after_x[0]
            # If horizontal direction reverses with significant magnitude, it's a hit
            if (vx_before * vx_after) < 0 and abs(vx_before) > 2 and abs(vx_after) > 2:
                # Also check speed spike — hits cause sharp acceleration
                speed_before = np.mean([ball_speeds_px[j]
                                        for j in range(max(0, i - speed_window), i)
                                        if ball_speeds_px[j] > 0]) if i > 0 else 0
                speed_after = np.mean([ball_speeds_px[j]
                                       for j in range(i, min(len(ball_speeds_px), i + speed_window))
                                       if ball_speeds_px[j] > 0]) if i < len(ball_speeds_px) else 0
                speed_ratio = speed_after / max(speed_before, 0.1)
                if speed_ratio > 1.5:
                    logger.debug(f"Rejected hit at frame {i}: vx reversal + speed_ratio={speed_ratio:.1f}")
                    continue

        bounces.append((i, cx, cy))
        last_bounce_frame = i
        logger.debug(f"Bounce detected at frame {i}: ({cx}, {cy}), y_ratio={y_ratio:.2f}")

    logger.info(f"Trajectory bounce detection: {len(bounces)} bounces found")
    return bounces


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
    # New feature flags
    parser.add_argument("--no-court", action="store_true", help="Disable court zone overlay")
    parser.add_argument("--no-bounces", action="store_true", help="Disable bounce detection markers")
    parser.add_argument("--no-player-boxes", action="store_true", help="Disable player bounding boxes")
    parser.add_argument("--court-labels", action="store_true", help="Show court zone labels")
    return parser.parse_args()


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

    # ─── Court zone detection (on first frame) ───
    court_zones = {}
    if not args.no_court or not args.no_bounces:
        # Read first frame for court detection
        first_frame = next(reader.iter_frames())
        reader.release()
        reader = VideoReader(video_path)
        if start_frame > 0:
            reader.seek(start_frame)

        logger.info("Detecting court zones on first frame...")
        court_zones = detector.detect_court_zones(first_frame, conf_threshold=0.3)
        if court_zones:
            logger.info(f"Detected {len(court_zones)} court zones: {list(court_zones.keys())}")
        else:
            logger.warning("No court zones detected")

    # ─── Pass 1: Detect ball positions + bounces ───
    logger.info("=== PASS 1: Detect ball ===")
    ball_conf_thresh = 0.15  # low threshold — we filter by motion, not confidence

    # Step 1a: Collect ALL ball candidates per frame
    raw_detections = []  # list of list of (cx, cy, score)
    frame_count = 0
    for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 1a: detect"):
        if frame_count >= max_frames:
            break
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
        process_noise=50.0,
        measurement_noise=8.0
    )
    # Gate grows with frames_since_update: ball moves further when we miss frames
    kf_gate_base = 80.0               # base gate in pixels for 1-frame gap
    kf_gate_growth = 40.0             # extra pixels per missed frame
    kf_gate_max = frame_diag * 0.20   # never exceed 20% of diagonal
    kf_max_coast = int(fps * 0.5)     # coast 0.5s — prefer prediction over wild reinit
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
                    reinit_gate = frame_diag * 0.15
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

    # ─── Estimate scale and convert speeds to km/h ───
    px_per_m = estimate_px_per_meter(court_zones, frame_height)
    ball_speeds_kmh = []
    for spx in ball_speeds_px:
        speed_m_per_s = (spx / px_per_m) * fps
        speed_kmh = speed_m_per_s * 3.6
        ball_speeds_kmh.append(speed_kmh)
    valid_speeds = [s for s in ball_speeds_kmh if s > 5.0]
    if valid_speeds:
        logger.info(f"Ball speed stats: avg={np.mean(valid_speeds):.0f} km/h, "
                    f"max={np.max(valid_speeds):.0f} km/h, "
                    f"min={np.min(valid_speeds):.0f} km/h")

    # ─── Detect bounces from smoothed trajectory ───
    bounce_events = []
    if not args.no_bounces:
        bounce_events = detect_bounces_from_trajectory(
            all_ball_centers, ball_speeds_px, fps, frame_height)

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

    # ─── Pass 2: Annotate ───
    logger.info("=== PASS 2: Annotate ===")
    reader = VideoReader(video_path)
    if start_frame > 0:
        reader.seek(start_frame)
    writer = VideoWriter(output_path, fps=fps)
    ball_trail = deque(maxlen=trail_length)
    skeleton_color = (0, 220, 255)  # single color for all skeletons
    player_tracker = PlayerTracker(iou_threshold=0.3)

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

    frame_idx = 0
    for frame in tqdm(reader.iter_frames(), total=max_frames, desc="Pass 2"):
        if frame_idx >= max_frames:
            break
        out = frame.copy()

        # 1. Court zone overlay (background layer)
        if not args.no_court and court_zones:
            out = draw_court_overlay(out, court_zones, alpha=0.15,
                                     draw_labels=args.court_labels)

        # 2. Pose — detect persons, pick the 2 most likely players, draw skeletons + boxes
        pose_res = pose_model(frame, conf=args.pose_conf, device=device,
                              imgsz=pose_imgsz, verbose=False)

        person_boxes = []
        pose_player_boxes = []  # boxes from pose model (may have skeletons)

        if (pose_res[0].keypoints is not None and
                pose_res[0].boxes is not None and len(pose_res[0].boxes) > 0):
            pose_kps = pose_res[0].keypoints
            pboxes = pose_res[0].boxes.xyxy.cpu().numpy()

            # Score each person to find the 2 actual players:
            # In broadcast tennis, players are in the central horizontal band.
            # Ball boys, umpire, spectators are on the edges.
            cx_frame = frame_width / 2
            scores = []
            for i in range(len(pboxes)):
                bx = pboxes[i]
                box_cx = (bx[0] + bx[2]) / 2
                box_w = bx[2] - bx[0]
                box_h = bx[3] - bx[1]
                area = box_w * box_h

                x_ratio = box_cx / frame_width

                # Hard reject: outside the central 20%-80% horizontal band
                if x_ratio < 0.20 or x_ratio > 0.80:
                    scores.append(-1.0)
                    continue

                # Horizontal centrality (0-1, 1=center)
                h_dist = abs(box_cx - cx_frame) / cx_frame
                h_score = max(0.0, 1.0 - h_dist)

                # Area score — players are usually the largest persons on court
                area_norm = area / (frame_width * frame_height)

                # Combined: centrality dominates, area is secondary
                score = h_score * 0.7 + area_norm * 30.0
                scores.append(score)

            scores = np.array(scores)
            # Take top 2 by score, reject anyone with score <= 0 (outside central band)
            valid_idx = np.where(scores > 0)[0]
            if len(valid_idx) > 0:
                sorted_valid = valid_idx[np.argsort(-scores[valid_idx])]
                top2_idx = sorted_valid[:2]
                for pi in top2_idx:
                    kps = pose_kps.xy[pi].cpu().numpy()
                    kp_c = pose_kps.conf[pi].cpu().numpy()
                    out = draw_skeleton(out, kps, kp_c, skeleton_color)
                    pose_player_boxes.append(pboxes[pi])

        # Fallback: if pose model found <2 players, use generic YOLO person detector
        # The far player is often too small for pose estimation but detectable by YOLO
        if len(pose_player_boxes) < 2:
            det = detector.detect(frame, classes=[0])  # class 0 = person
            person_mask = det["class_ids"] == 0
            if person_mask.any():
                yolo_pboxes = det["boxes"][person_mask]
                cx_frame = frame_width / 2
                for i in range(len(yolo_pboxes)):
                    bx = yolo_pboxes[i]
                    box_cx = (bx[0] + bx[2]) / 2
                    x_ratio = box_cx / frame_width
                    if x_ratio < 0.20 or x_ratio > 0.80:
                        continue
                    # Check this box doesn't overlap with already-found pose players
                    overlaps = False
                    for pb in pose_player_boxes:
                        iou = PlayerTracker._iou(bx, pb)
                        if iou > 0.3:
                            overlaps = True
                            break
                    if not overlaps:
                        pose_player_boxes.append(bx)
                        if len(pose_player_boxes) >= 2:
                            break

        person_boxes = list(pose_player_boxes)

        # 3. Player bounding boxes (same top 2 persons)
        if not args.no_player_boxes and person_boxes:
            tracked = player_tracker.update(person_boxes)
            for pid, box in tracked:
                out = draw_player_bbox(out, box, pid)

        # 4. Ball (already Kalman-filtered + interpolated)
        bc = all_ball_centers[frame_idx]
        if bc is not None:
            ball_trail.append(bc)
            out = draw_ball_marker(out, bc[0], bc[1])
        draw_ball_trail(out, list(ball_trail))

        # 4b. Speed HUD (top-left)
        current_speed = ball_speeds_kmh[frame_idx] if frame_idx < len(ball_speeds_kmh) else 0.0
        if bc is not None and current_speed > 5.0:
            speed_text = f"Ball: {int(current_speed)} km/h"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(speed_text, font, font_scale, thickness)
            hud_x, hud_y = 15, 35
            # Semi-transparent background
            overlay = out.copy()
            cv2.rectangle(overlay, (hud_x - 5, hud_y - th - 5),
                          (hud_x + tw + 5, hud_y + baseline + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
            cv2.putText(out, speed_text, (hud_x, hud_y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        # 5. Bounce markers (persist for bounce_display_frames, fade over time)
        if not args.no_bounces:
            for bfi in range(max(0, frame_idx - bounce_display_frames), frame_idx + 1):
                if bfi in bounce_by_frame:
                    bx, by, depth, b_speed, b_quality = bounce_by_frame[bfi]
                    age = frame_idx - bfi
                    alpha = max(0.3, 1.0 - age / bounce_display_frames)
                    out = draw_bounce_marker(out, bx, by, depth=depth, alpha=alpha)
                    # Draw multi-line label next to marker
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    lx, ly = bx + 18, by - 10
                    # Line 1: depth
                    cv2.putText(out, depth.upper(), (lx, ly), font, 0.5,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    # Line 2: speed
                    cv2.putText(out, f"{int(b_speed)} km/h", (lx, ly + 18), font, 0.45,
                                (200, 200, 200), 1, cv2.LINE_AA)
                    # Line 3: quality (colored)
                    if b_quality >= 70:
                        q_color = (0, 200, 0)     # green
                    elif b_quality >= 40:
                        q_color = (0, 220, 220)    # yellow
                    else:
                        q_color = (0, 0, 220)      # red
                    cv2.putText(out, f"Q:{b_quality}", (lx, ly + 34), font, 0.45,
                                q_color, 2, cv2.LINE_AA)

        # 6. Legend
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

    logger.info(f"Done! Video saved to: {output_path}")


if __name__ == "__main__":
    main()
