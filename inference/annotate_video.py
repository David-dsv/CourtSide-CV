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
                        frame = draw_tracks(frame, tracks)

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
                        "Ball": (255, 255, 0),
                        "Racket": (255, 0, 255)
                    }
                    frame = create_legend(frame, legend_items, position="top-right")

                writer.write_frame(frame)

    logger.info(f"Comprehensive annotation saved to {output_path}")
