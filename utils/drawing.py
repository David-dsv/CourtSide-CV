"""
Visualization and Drawing Utilities
Draw bounding boxes, keypoints, labels, and tracking information
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import colorsys


def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for visualization

    Args:
        num_colors: Number of colors to generate

    Returns:
        List of BGR color tuples
    """
    num_colors = max(num_colors, 1)
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        bgr = tuple(int(c * 255) for c in reversed(rgb))
        colors.append(bgr)
    return colors


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    label: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on image

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        label: Optional text label
        color: Box color (BGR)
        thickness: Line thickness
        font_scale: Font scale for label
        font_thickness: Font thickness

    Returns:
        Image with drawn box
    """
    x1, y1, x2, y2 = bbox.astype(int)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Draw label if provided
    if label:
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness
        )

        # Draw filled rectangle as background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return image


def draw_bboxes(
    image: np.ndarray,
    bboxes: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw multiple bounding boxes

    Args:
        image: Input image
        bboxes: Array of bounding boxes [N, 4]
        labels: Optional list of labels
        colors: Optional list of colors
        thickness: Line thickness
        font_scale: Font scale

    Returns:
        Image with drawn boxes
    """
    if colors is None:
        colors = [(0, 255, 0)] * len(bboxes)

    if labels is None:
        labels = [None] * len(bboxes)

    for bbox, label, color in zip(bboxes, labels, colors):
        image = draw_bbox(image, bbox, label, color, thickness, font_scale)

    return image


def draw_tracks(
    image: np.ndarray,
    tracks: np.ndarray,
    show_id: bool = True,
    show_score: bool = True,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw tracked objects

    Args:
        image: Input image
        tracks: Array of tracks [N, 6+] where columns are [x1, y1, x2, y2, track_id, score, ...]
        show_id: Show track ID
        show_score: Show confidence score
        colors: Optional list of colors (indexed by track_id)
        thickness: Line thickness
        font_scale: Font scale

    Returns:
        Image with drawn tracks
    """
    if len(tracks) == 0:
        return image

    # Generate colors if not provided
    if colors is None:
        max_id = int(tracks[:, 4].max()) if len(tracks) > 0 else 10
        colors = generate_colors(max_id + 1)

    for track in tracks:
        x1, y1, x2, y2, track_id, score = track[:6]
        track_id = int(track_id)

        # Get color for this track
        color = colors[track_id % len(colors)]

        # Build label
        label_parts = []
        if show_id:
            label_parts.append(f"ID:{track_id}")
        if show_score:
            label_parts.append(f"{score:.2f}")

        label = " ".join(label_parts) if label_parts else None

        # Draw
        image = draw_bbox(
            image,
            np.array([x1, y1, x2, y2]),
            label,
            color,
            thickness,
            font_scale
        )

    return image


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    radius: int = 4,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw keypoints on image

    Args:
        image: Input image
        keypoints: Keypoints array [N, 2] (x, y)
        scores: Optional confidence scores [N]
        threshold: Confidence threshold to draw
        radius: Circle radius
        color: Point color

    Returns:
        Image with drawn keypoints
    """
    for i, (x, y) in enumerate(keypoints):
        # Check score threshold
        if scores is not None and scores[i] < threshold:
            continue

        x, y = int(x), int(y)

        # Draw circle
        cv2.circle(image, (x, y), radius, color, -1)

        # Draw outline
        cv2.circle(image, (x, y), radius + 1, (255, 255, 255), 1)

    return image


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    threshold: float = 0.3,
    line_thickness: int = 2,
    point_radius: int = 4,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw pose skeleton on image

    Args:
        image: Input image
        keypoints: Keypoints array [N, 2]
        scores: Optional confidence scores [N]
        skeleton: List of (start_idx, end_idx) connections
        threshold: Confidence threshold
        line_thickness: Line thickness
        point_radius: Point radius
        color: Drawing color

    Returns:
        Image with drawn skeleton
    """
    # Default COCO skeleton if not provided
    if skeleton is None:
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

    # Draw connections
    for start_idx, end_idx in skeleton:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        # Check scores
        if scores is not None:
            if scores[start_idx] < threshold or scores[end_idx] < threshold:
                continue

        start_point = tuple(keypoints[start_idx].astype(int))
        end_point = tuple(keypoints[end_idx].astype(int))

        cv2.line(image, start_point, end_point, color, line_thickness, cv2.LINE_AA)

    # Draw keypoints
    image = draw_keypoints(image, keypoints, scores, threshold, point_radius, color)

    return image


def draw_action_label(
    image: np.ndarray,
    action: str,
    confidence: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Draw action label on image

    Args:
        image: Input image
        action: Action name
        confidence: Confidence score
        position: Text position (x, y)
        font_scale: Font scale
        thickness: Font thickness
        bg_color: Background color
        text_color: Text color

    Returns:
        Image with action label
    """
    text = f"{action}: {confidence:.2%}"

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    x, y = position

    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        bg_color,
        -1
    )

    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return image


def draw_trajectory(
    image: np.ndarray,
    trajectory: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    max_length: int = 30
) -> np.ndarray:
    """
    Draw object trajectory

    Args:
        image: Input image
        trajectory: List of (x, y) points
        color: Line color
        thickness: Line thickness
        max_length: Maximum trajectory length to display

    Returns:
        Image with trajectory
    """
    if len(trajectory) < 2:
        return image

    # Limit trajectory length
    trajectory = trajectory[-max_length:]

    # Draw lines
    for i in range(len(trajectory) - 1):
        pt1 = tuple(map(int, trajectory[i]))
        pt2 = tuple(map(int, trajectory[i + 1]))

        # Fade older points
        alpha = (i + 1) / len(trajectory)
        line_color = tuple(int(c * alpha) for c in color)

        cv2.line(image, pt1, pt2, line_color, thickness, cv2.LINE_AA)

    return image


def create_legend(
    image: np.ndarray,
    items: Dict[str, Tuple[int, int, int]],
    position: str = "top-right",
    font_scale: float = 0.5,
    thickness: int = 1,
    margin: int = 10
) -> np.ndarray:
    """
    Add legend to image

    Args:
        image: Input image
        items: Dictionary of {label: color}
        position: Legend position (top-right, top-left, bottom-right, bottom-left)
        font_scale: Font scale
        thickness: Font thickness
        margin: Margin from edge

    Returns:
        Image with legend
    """
    if len(items) == 0:
        return image

    # Calculate legend size
    max_text_width = 0
    total_height = 0
    line_height = 25

    for label in items.keys():
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )
        max_text_width = max(max_text_width, text_width)
        total_height += line_height

    legend_width = max_text_width + 40
    legend_height = total_height + 10

    # Determine position
    h, w = image.shape[:2]

    if position == "top-right":
        x = w - legend_width - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    elif position == "bottom-right":
        x = w - legend_width - margin
        y = h - legend_height - margin
    else:  # bottom-left
        x = margin
        y = h - legend_height - margin

    # Draw background
    cv2.rectangle(
        image,
        (x, y),
        (x + legend_width, y + legend_height),
        (0, 0, 0),
        -1
    )
    cv2.rectangle(
        image,
        (x, y),
        (x + legend_width, y + legend_height),
        (255, 255, 255),
        1
    )

    # Draw items
    current_y = y + 20
    for label, color in items.items():
        # Draw color box
        cv2.rectangle(
            image,
            (x + 5, current_y - 10),
            (x + 20, current_y + 5),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x + 25, current_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        current_y += line_height

    return image


def draw_player_bbox(
    image: np.ndarray,
    box: np.ndarray,
    player_id: int,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box with 'Player N' label.
    Default colors: amber for P1, blue for P2.
    """
    PLAYER_COLORS = {
        1: (0, 191, 255),   # amber/gold (BGR)
        2: (255, 160, 50),   # blue (BGR)
    }
    if color is None:
        color = PLAYER_COLORS.get(player_id, (0, 255, 0))

    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label = f"Player {player_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    cv2.rectangle(image, (x1, y1 - th - baseline - 5), (x1 + tw + 4, y1), color, -1)
    cv2.putText(image, label, (x1 + 2, y1 - baseline - 2), font, font_scale,
                (255, 255, 255), font_thick, cv2.LINE_AA)
    return image


def draw_bounce_marker(
    image: np.ndarray,
    x: int,
    y: int,
    depth: str = "mid",
    radius: int = 14,
    thickness: int = 2,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Draw a circle + X marker at bounce location.
    Color by depth: red=deep, orange=mid, yellow=short.
    alpha (0-1) fades the marker over time.
    """
    DEPTH_COLORS = {
        "deep": (0, 0, 255),     # red
        "mid": (0, 140, 255),    # orange
        "short": (0, 230, 255),  # yellow
    }
    base_color = DEPTH_COLORS.get(depth, (0, 140, 255))
    color = tuple(int(c * alpha) for c in base_color)

    cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)
    # X marker inside circle
    d = int(radius * 0.5)
    cv2.line(image, (x - d, y - d), (x + d, y + d), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x - d, y + d), (x + d, y - d), color, thickness, cv2.LINE_AA)

    return image


def draw_court_overlay(
    image: np.ndarray,
    court_zones: Dict[str, Dict],
    alpha: float = 0.15,
    draw_labels: bool = False
) -> np.ndarray:
    """
    Draw semi-transparent colored rectangles for each court zone.
    court_zones: {zone_name: {'box': [x1,y1,x2,y2], 'score': float}}
    """
    ZONE_COLORS = {
        "bottom-dead-zone": (80, 80, 200),
        "court": (80, 180, 80),
        "left-doubles-alley": (200, 150, 50),
        "left-service-box": (200, 200, 50),
        "net": (100, 100, 255),
        "right-doubles-alley": (50, 150, 200),
        "right-service-box": (50, 200, 200),
        "top-dead-zone": (200, 80, 80),
    }

    overlay = image.copy()
    for zone_name, info in court_zones.items():
        box = info["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        color = ZONE_COLORS.get(zone_name, (128, 128, 128))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw solid border lines and optional labels
    for zone_name, info in court_zones.items():
        box = info["box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        color = ZONE_COLORS.get(zone_name, (128, 128, 128))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if draw_labels:
            label = zone_name.replace("-", " ").title()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, label, (x1 + 4, y1 + 16), font, 0.4, color, 1, cv2.LINE_AA)

    return image


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image

    Args:
        image: Input image [H, W, 3]
        heatmap: Heatmap [H, W] or [H, W, 1]
        alpha: Overlay transparency
        colormap: OpenCV colormap

    Returns:
        Image with heatmap overlay
    """
    # Normalize heatmap to 0-255
    heatmap = heatmap.squeeze()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Resize to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)

    # Blend
    result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return result
