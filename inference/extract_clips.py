"""
Clip Extraction Module
Extract video clips centered on tracked objects
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger

from utils.video_utils import VideoReader, VideoWriter


def extract_player_clips(
    video_path: str,
    tracking_results: List[Dict],
    output_dir: str,
    track_ids: Optional[List[int]] = None,
    margin: int = 50,
    min_size: Tuple[int, int] = (128, 128)
) -> List[str]:
    """
    Extract clips for each tracked player

    Args:
        video_path: Input video path
        tracking_results: Tracking results
        output_dir: Output directory for clips
        track_ids: Optional list of track IDs to extract (None = all)
        margin: Margin around bounding box in pixels
        min_size: Minimum clip size (width, height)

    Returns:
        List of output clip paths
    """
    logger.info(f"Extracting player clips from {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all unique track IDs
    all_track_ids = set()
    for frame_result in tracking_results:
        all_track_ids.update(frame_result["tracks"]["track_ids"])

    # Filter track IDs if specified
    if track_ids is not None:
        all_track_ids = set(track_ids) & all_track_ids

    logger.info(f"Extracting clips for {len(all_track_ids)} tracks")

    output_paths = []

    with VideoReader(video_path) as reader:
        # Extract clips for each track
        for track_id in sorted(all_track_ids):
            track_id = int(track_id)
            output_path = output_dir / f"track_{track_id:04d}.mp4"

            # Get all frames with this track
            track_frames = []

            for frame_result in tracking_results:
                tracks = frame_result["tracks"]

                if track_id not in tracks["track_ids"]:
                    continue

                # Find track index
                idx = tracks["track_ids"].index(track_id)
                bbox = tracks["boxes"][idx]

                track_frames.append({
                    "frame_id": frame_result["frame_id"],
                    "bbox": bbox
                })

            if len(track_frames) == 0:
                continue

            # Extract and crop frames
            with VideoWriter(str(output_path), fps=reader.fps) as writer:
                for track_frame in track_frames:
                    # Read frame
                    reader.seek(track_frame["frame_id"])
                    ret, frame = reader.read_frame()

                    if not ret:
                        continue

                    # Crop around bbox
                    cropped = crop_bbox(frame, track_frame["bbox"], margin, min_size)

                    if cropped is not None:
                        writer.write_frame(cropped)

            output_paths.append(str(output_path))

    logger.info(f"Extracted {len(output_paths)} player clips")

    return output_paths


def extract_action_clips(
    video_path: str,
    action_detections: List[Dict],
    output_dir: str,
    margin_frames: int = 16
) -> List[str]:
    """
    Extract clips around detected actions

    Args:
        video_path: Input video path
        action_detections: List of action detection results
        output_dir: Output directory
        margin_frames: Frames before and after action

    Returns:
        List of output clip paths
    """
    logger.info(f"Extracting action clips from {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    with VideoReader(video_path) as reader:
        for i, action_det in enumerate(action_detections):
            frame_id = action_det["frame_id"]
            action = action_det["action"]

            # Calculate clip range
            start_frame = max(0, frame_id - margin_frames)
            end_frame = min(len(reader) - 1, frame_id + margin_frames)

            # Output path
            output_path = output_dir / f"action_{i:04d}_{action}_frame_{frame_id}.mp4"

            # Extract clip
            reader.seek(start_frame)

            with VideoWriter(str(output_path), fps=reader.fps) as writer:
                for _ in range(end_frame - start_frame):
                    ret, frame = reader.read_frame()
                    if not ret:
                        break
                    writer.write_frame(frame)

            output_paths.append(str(output_path))

    logger.info(f"Extracted {len(output_paths)} action clips")

    return output_paths


def crop_bbox(
    frame: np.ndarray,
    bbox: List[float],
    margin: int = 50,
    min_size: Tuple[int, int] = (128, 128)
) -> Optional[np.ndarray]:
    """
    Crop frame around bounding box with margin

    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Margin in pixels
        min_size: Minimum output size

    Returns:
        Cropped frame or None if invalid
    """
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    # Add margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    # Check minimum size
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_size[0] or crop_h < min_size[1]:
        return None

    # Crop
    cropped = frame[y1:y2, x1:x2]

    return cropped


def extract_temporal_clips(
    frames: List[np.ndarray],
    clip_length: int,
    clip_stride: int,
    output_dir: Optional[str] = None
) -> List[np.ndarray]:
    """
    Extract overlapping temporal clips from frames

    Args:
        frames: List of frames
        clip_length: Frames per clip
        clip_stride: Stride between clips
        output_dir: Optional directory to save clips as videos

    Returns:
        List of clips (each clip is list of frames)
    """
    clips = []

    for start_idx in range(0, len(frames) - clip_length + 1, clip_stride):
        end_idx = start_idx + clip_length
        clip = frames[start_idx:end_idx]
        clips.append(clip)

    # Save to disk if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, clip in enumerate(clips):
            output_path = output_dir / f"clip_{i:04d}.mp4"
            with VideoWriter(str(output_path), fps=30) as writer:
                for frame in clip:
                    writer.write_frame(frame)

    logger.info(f"Extracted {len(clips)} temporal clips")

    return clips
