"""
Video Annotation Module
Annotate video with detections, tracks, and actions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from utils.video_utils import VideoReader, VideoWriter
from utils.drawing import (
    draw_bboxes,
    draw_tracks,
    draw_keypoints,
    draw_skeleton,
    draw_action_label,
    create_legend
)
import subprocess
import tempfile
import shutil


def _draw_player_stats(frame, action_detections, tracking_results, current_frame):
    """
    Draw player statistics on frame

    Args:
        frame: Current frame
        action_detections: All action detections
        tracking_results: All tracking results
        current_frame: Current frame ID

    Returns:
        Annotated frame
    """
    if not action_detections:
        return frame

    # Count actions per player
    player_stats = {}

    for action in action_detections:
        if action["frame_id"] <= current_frame:  # Only count past actions
            track_id = action.get("track_id", 0)
            action_type = action["action"]

            if track_id not in player_stats:
                player_stats[track_id] = {
                    "forehand": 0,
                    "backhand": 0,
                    "serve": 0,
                    "volley": 0,
                    "slice": 0,
                    "smash": 0
                }

            if action_type in player_stats[track_id]:
                player_stats[track_id][action_type] += 1

    # Draw stats box
    y_pos = 120  # Start below legend
    x_pos = frame.shape[1] - 200  # Right side

    # Noms et couleurs des joueurs
    player_names = {
        4: "Alcaraz",  # Track 4
        5: "Sinner"    # Track 5
    }
    player_colors = {
        4: (0, 255, 255),    # Jaune pour Alcaraz
        5: (60, 20, 220),    # Bordeaux pour Sinner
    }

    for track_id, stats in player_stats.items():
        # Get player color
        player_color = player_colors.get(track_id, (255, 255, 255))
        player_name = player_names.get(track_id, f"Player {track_id}")

        # Draw background box
        cv2.rectangle(frame,
                     (x_pos - 10, y_pos - 5),
                     (x_pos + 180, y_pos + 80),
                     (0, 0, 0), -1)
        cv2.rectangle(frame,
                     (x_pos - 10, y_pos - 5),
                     (x_pos + 180, y_pos + 80),
                     player_color, 2)

        # Player title with name and color
        cv2.putText(frame, player_name,
                   (x_pos, y_pos + 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, player_color, 2, cv2.LINE_AA)

        # Stats
        y_offset = 25
        for action_name in ["forehand", "backhand", "serve"]:
            count = stats.get(action_name, 0)
            if count > 0:
                text = f"{action_name}: {count}"
                cv2.putText(frame, text,
                           (x_pos, y_pos + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (200, 200, 200), 1, cv2.LINE_AA)
                y_offset += 15

        y_pos += 90  # Space for next player

    return frame


def _copy_audio_track(source_video: str, target_video: str):
    """
    Copy audio track from source video to target video using ffmpeg

    Args:
        source_video: Original video with audio
        target_video: Annotated video without audio
    """
    try:
        # Create temporary file for video with audio
        temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

        # FFmpeg command to merge video (from target) and audio (from source)
        cmd = [
            'ffmpeg',
            '-i', target_video,        # Video without audio
            '-i', source_video,        # Original video with audio
            '-map', '0:v:0',           # Take video from first input
            '-map', '1:a:0?',          # Take audio from second input (? = optional)
            '-c:v', 'copy',            # Copy video without re-encoding
            '-c:a', 'aac',             # Encode audio as AAC
            '-b:a', '192k',            # Audio bitrate
            '-shortest',               # Match shortest stream duration
            '-async', '1',             # Audio sync method
            '-vsync', '1',             # Video sync method
            '-y',                      # Overwrite output
            temp_output
        ]

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Replace original file with the one that has audio
        shutil.move(temp_output, target_video)

        logger.info("Audio track copied successfully")

    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not copy audio track: {e.stderr}")
        logger.info("Video saved without audio")
    except Exception as e:
        logger.warning(f"Error copying audio: {e}")
        logger.info("Video saved without audio")


def annotate_with_detections(
    video_path: str,
    detections_list: List[Dict],
    output_path: str,
    show_labels: bool = True,
    show_confidence: bool = True
):
    """
    Annotate video with object detections

    Args:
        video_path: Input video path
        detections_list: List of detection results per frame
        output_path: Output video path
        show_labels: Show class labels
        show_confidence: Show confidence scores
    """
    logger.info(f"Annotating video with detections: {video_path}")

    with VideoReader(video_path) as reader:
        with VideoWriter(output_path, fps=reader.fps) as writer:
            for frame_id, frame in enumerate(tqdm(reader.iter_frames(), total=len(reader))):
                if frame_id >= len(detections_list):
                    break

                det_result = detections_list[frame_id]
                detections = det_result["detections"]

                # Draw detections
                if len(detections["boxes"]) > 0:
                    boxes = np.array(detections["boxes"])
                    labels = None

                    if show_labels:
                        labels = []
                        for i, class_name in enumerate(detections["class_names"]):
                            if show_confidence:
                                score = detections["scores"][i]
                                labels.append(f"{class_name} {score:.2f}")
                            else:
                                labels.append(class_name)

                    frame = draw_bboxes(frame, boxes, labels=labels)

                writer.write_frame(frame)

    logger.info(f"Annotated video saved to {output_path}")


def annotate_with_tracks(
    video_path: str,
    tracking_results: List[Dict],
    output_path: str,
    show_id: bool = True,
    show_score: bool = True
):
    """
    Annotate video with tracking results

    Args:
        video_path: Input video path
        tracking_results: List of tracking results per frame
        output_path: Output video path
        show_id: Show track IDs
        show_score: Show confidence scores
    """
    logger.info(f"Annotating video with tracks: {video_path}")

    with VideoReader(video_path) as reader:
        with VideoWriter(output_path, fps=reader.fps) as writer:
            for frame_id, frame in enumerate(tqdm(reader.iter_frames(), total=len(reader))):
                if frame_id >= len(tracking_results):
                    break

                track_result = tracking_results[frame_id]
                tracks_data = track_result["tracks"]

                # Convert to array format for drawing
                if len(tracks_data["boxes"]) > 0:
                    tracks = []
                    for i in range(len(tracks_data["boxes"])):
                        box = tracks_data["boxes"][i]
                        track_id = tracks_data["track_ids"][i]
                        score = tracks_data["scores"][i]
                        class_id = tracks_data["class_ids"][i]

                        tracks.append([
                            box[0], box[1], box[2], box[3],
                            track_id, score, class_id
                        ])

                    tracks = np.array(tracks)
                    frame = draw_tracks(frame, tracks, show_id=show_id, show_score=show_score)

                writer.write_frame(frame)

    logger.info(f"Annotated video saved to {output_path}")


def annotate_with_actions(
    video_path: str,
    action_detections: List[Dict],
    output_path: str,
    tracking_results: Optional[List[Dict]] = None,
    show_tracks: bool = True
):
    """
    Annotate video with action detections

    Args:
        video_path: Input video path
        action_detections: List of action detection results
        output_path: Output video path
        tracking_results: Optional tracking results to overlay
        show_tracks: Show tracking boxes
    """
    logger.info(f"Annotating video with actions: {video_path}")

    # Create frame-to-action mapping
    frame_actions = {}
    for det in action_detections:
        frame_id = det["frame_id"]
        if frame_id not in frame_actions:
            frame_actions[frame_id] = []
        frame_actions[frame_id].append(det)

    with VideoReader(video_path) as reader:
        with VideoWriter(output_path, fps=reader.fps) as writer:
            for frame_id, frame in enumerate(tqdm(reader.iter_frames(), total=len(reader))):
                # Draw tracks if available
                if show_tracks and tracking_results and frame_id < len(tracking_results):
                    track_result = tracking_results[frame_id]
                    tracks_data = track_result["tracks"]

                    if len(tracks_data["boxes"]) > 0:
                        tracks = []
                        for i in range(len(tracks_data["boxes"])):
                            box = tracks_data["boxes"][i]
                            track_id = tracks_data["track_ids"][i]
                            score = tracks_data["scores"][i]
                            class_id = tracks_data["class_ids"][i]
                            tracks.append([box[0], box[1], box[2], box[3], track_id, score, class_id])

                        tracks = np.array(tracks)
                        frame = draw_tracks(frame, tracks)

                # Draw actions
                if frame_id in frame_actions:
                    y_offset = 30
                    for action_det in frame_actions[frame_id]:
                        action = action_det["action"]
                        confidence = action_det["confidence"]

                        frame = draw_action_label(
                            frame,
                            action,
                            confidence,
                            position=(10, y_offset)
                        )

                        y_offset += 40

                writer.write_frame(frame)

    logger.info(f"Annotated video saved to {output_path}")


def annotate_with_pose(
    video_path: str,
    pose_results: List[Dict],
    output_path: str,
    show_keypoints: bool = True,
    show_skeleton: bool = True,
    conf_threshold: float = 0.3
):
    """
    Annotate video with pose estimation

    Args:
        video_path: Input video path
        pose_results: List of pose results per frame
        output_path: Output video path
        show_keypoints: Show keypoints
        show_skeleton: Show skeleton connections
        conf_threshold: Confidence threshold for keypoints
    """
    logger.info(f"Annotating video with pose: {video_path}")

    with VideoReader(video_path) as reader:
        with VideoWriter(output_path, fps=reader.fps) as writer:
            for frame_id, frame in enumerate(tqdm(reader.iter_frames(), total=len(reader))):
                if frame_id >= len(pose_results):
                    break

                pose_result = pose_results[frame_id]

                # Draw pose
                if "keypoints" in pose_result:
                    keypoints = np.array(pose_result["keypoints"])
                    scores = np.array(pose_result["scores"])

                    if show_skeleton:
                        frame = draw_skeleton(
                            frame,
                            keypoints,
                            scores,
                            threshold=conf_threshold
                        )
                    elif show_keypoints:
                        frame = draw_keypoints(
                            frame,
                            keypoints,
                            scores,
                            threshold=conf_threshold
                        )

                writer.write_frame(frame)

    logger.info(f"Annotated video saved to {output_path}")


def annotate_comprehensive(
    video_path: str,
    output_path: str,
    detections_list: Optional[List[Dict]] = None,
    tracking_results: Optional[List[Dict]] = None,
    action_detections: Optional[List[Dict]] = None,
    pose_results: Optional[List[Dict]] = None,
    show_detections: bool = False,
    show_tracks: bool = True,
    show_actions: bool = True,
    show_pose: bool = False,
    show_legend: bool = True
):
    """
    Comprehensive video annotation with all available information

    Args:
        video_path: Input video path
        output_path: Output video path
        detections_list: Detection results
        tracking_results: Tracking results
        action_detections: Action detection results
        pose_results: Pose estimation results
        show_detections: Show object detections
        show_tracks: Show tracking
        show_actions: Show action labels
        show_pose: Show pose skeleton
        show_legend: Show legend
    """
    logger.info(f"Creating comprehensive annotation: {video_path}")

    # Create frame-to-action mapping
    frame_actions = {}
    if action_detections:
        for det in action_detections:
            frame_id = det["frame_id"]
            if frame_id not in frame_actions:
                frame_actions[frame_id] = []
            frame_actions[frame_id].append(det)

    with VideoReader(video_path) as reader:
        with VideoWriter(output_path, fps=reader.fps) as writer:
            for frame_id, frame in enumerate(tqdm(reader.iter_frames(), total=len(reader))):
                # Draw detections
                if show_detections and detections_list and frame_id < len(detections_list):
                    det_result = detections_list[frame_id]
                    detections = det_result["detections"]

                    if len(detections["boxes"]) > 0:
                        boxes = np.array(detections["boxes"])
                        labels = [
                            f"{name} {score:.2f}"
                            for name, score in zip(detections["class_names"], detections["scores"])
                        ]
                        frame = draw_bboxes(frame, boxes, labels=labels)

                # Draw tracks
                if show_tracks and tracking_results and frame_id < len(tracking_results):
                    track_result = tracking_results[frame_id]
                    tracks_data = track_result["tracks"]

                    if len(tracks_data["boxes"]) > 0:
                        tracks = []
                        for i in range(len(tracks_data["boxes"])):
                            box = tracks_data["boxes"][i]
                            track_id = tracks_data["track_ids"][i]
                            score = tracks_data["scores"][i]
                            class_id = tracks_data["class_ids"][i]
                            tracks.append([box[0], box[1], box[2], box[3], track_id, score, class_id])

                        tracks = np.array(tracks)

                        # Couleurs personnalisées pour les joueurs
                        # Track 4 et 5 sont les deux joueurs principaux
                        custom_colors = {
                            4: (0, 255, 255),    # Jaune pour Alcaraz (track 4)
                            5: (60, 20, 220),    # Bordeaux/Rouge foncé pour Sinner (track 5)
                        }

                        # Générer des couleurs pour les autres tracks
                        max_id = int(tracks[:, 4].max()) if len(tracks) > 0 else 10
                        colors_list = generate_colors(max_id + 1)

                        # Remplacer avec les couleurs personnalisées
                        for track_id, color in custom_colors.items():
                            if track_id < len(colors_list):
                                colors_list[track_id] = color

                        frame = draw_tracks(frame, tracks, colors=colors_list)

                # Draw pose
                if show_pose and pose_results and frame_id < len(pose_results):
                    pose_result = pose_results[frame_id]

                    if "keypoints" in pose_result:
                        keypoints = np.array(pose_result["keypoints"])
                        scores = np.array(pose_result["scores"])
                        frame = draw_skeleton(frame, keypoints, scores, threshold=0.3)

                # Draw actions
                if show_actions and frame_id in frame_actions:
                    y_offset = 30
                    for action_det in frame_actions[frame_id]:
                        action = action_det["action"]
                        confidence = action_det["confidence"]
                        frame = draw_action_label(frame, action, confidence, position=(10, y_offset))
                        y_offset += 40

                # Draw legend
                if show_legend:
                    legend_items = {
                        "Player": (0, 255, 0),
                        "Ball": (255, 255, 0)
                    }
                    frame = create_legend(frame, legend_items, position="top-right")

                    # Draw player statistics
                    frame = _draw_player_stats(frame, action_detections, tracking_results, frame_id)

                writer.write_frame(frame)

    logger.info(f"Comprehensive annotation saved to {output_path}")

    # Copy audio from original video
    _copy_audio_track(video_path, output_path)
