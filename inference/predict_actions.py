"""
Action Prediction Module
Predict tennis actions from video clips
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from models.action_classifier import TennisActionClassifier
from utils.preprocessing import (
    extract_clip_frames,
    preprocess_for_x3d,
    preprocess_for_slowfast,
    create_sliding_windows
)
from utils.video_utils import VideoReader
from utils.file_utils import save_json, create_results_dict


def predict_actions_from_clips(
    clip_frames_list: List[List[np.ndarray]],
    model: TennisActionClassifier,
    device: str = "cuda",
    batch_size: int = 8
) -> List[Dict]:
    """
    Predict actions from list of clips

    Args:
        clip_frames_list: List of clips (each clip is list of frames)
        model: Action classifier model
        device: Device to run on
        batch_size: Batch size for inference

    Returns:
        List of prediction results
    """
    logger.info(f"Predicting actions for {len(clip_frames_list)} clips")

    results = []

    # Process in batches
    for batch_start in tqdm(range(0, len(clip_frames_list), batch_size)):
        batch_clips = clip_frames_list[batch_start:batch_start + batch_size]

        # Preprocess batch
        batch_tensors = []

        for clip_frames in batch_clips:
            clip_array = np.array(clip_frames)

            # Preprocess based on model type
            if model.model_type == "x3d":
                tensor = preprocess_for_x3d(clip_array)
            elif model.model_type == "slowfast":
                tensor = preprocess_for_slowfast(clip_array)
            else:
                raise ValueError(f"Unknown model type: {model.model_type}")

            batch_tensors.append(tensor)

        # Stack tensors
        if model.model_type == "x3d":
            batch_input = torch.cat(batch_tensors, dim=0).to(device)
            predictions = model.predict_with_names(batch_input)
        elif model.model_type == "slowfast":
            # SlowFast requires special handling
            slow_pathway = torch.cat([t[0] for t in batch_tensors], dim=0).to(device)
            fast_pathway = torch.cat([t[1] for t in batch_tensors], dim=0).to(device)
            batch_input = [slow_pathway, fast_pathway]
            predictions = model.predict_with_names(batch_input)

        # Store results
        for pred in predictions:
            results.append(pred)

    return results


def predict_actions_from_video(
    video_path: str,
    model: TennisActionClassifier,
    clip_length: int = 32,
    clip_stride: int = 8,
    conf_threshold: float = 0.6,
    device: str = "cuda",
    batch_size: int = 8,
    output_json: Optional[str] = None
) -> List[Dict]:
    """
    Predict actions from entire video using sliding window

    Args:
        video_path: Input video path
        model: Action classifier model
        clip_length: Frames per clip
        clip_stride: Stride between clips
        conf_threshold: Confidence threshold
        device: Device to run on
        batch_size: Batch size
        output_json: Optional output JSON path

    Returns:
        List of action detection results
    """
    logger.info(f"Predicting actions from video: {video_path}")

    # Load video frames
    with VideoReader(video_path) as reader:
        fps = reader.fps
        all_frames = list(reader.iter_frames())

    # Create sliding windows
    windows = create_sliding_windows(len(all_frames), clip_length, clip_stride)

    logger.info(f"Processing {len(windows)} sliding windows")

    # Extract clips
    clips = []
    window_info = []

    for start, end in windows:
        clip = all_frames[start:end]
        clips.append(clip)
        window_info.append({
            "start_frame": start,
            "end_frame": end,
            "center_frame": (start + end) // 2
        })

    # Predict actions
    predictions = predict_actions_from_clips(clips, model, device, batch_size)

    # Format results
    results = []

    for pred, info in zip(predictions, window_info):
        if pred["confidence"] >= conf_threshold:
            result = create_results_dict(
                frame_id=info["center_frame"],
                timestamp=info["center_frame"] / fps,
                action=pred["class_name"],
                confidence=pred["confidence"],
                window_start=info["start_frame"],
                window_end=info["end_frame"]
            )
            results.append(result)

    # Save to JSON
    if output_json:
        save_json(results, output_json)

    logger.info(f"Detected {len(results)} actions")

    return results


def predict_actions_from_tracked_clips(
    video_path: str,
    tracking_results: List[Dict],
    model: TennisActionClassifier,
    clip_length: int = 32,
    temporal_window: int = 64,
    conf_threshold: float = 0.6,
    device: str = "cuda",
    track_id: Optional[int] = None
) -> List[Dict]:
    """
    Predict actions for tracked player

    Args:
        video_path: Input video path
        tracking_results: Tracking results
        model: Action classifier model
        clip_length: Frames per clip
        temporal_window: Temporal window to consider
        conf_threshold: Confidence threshold
        device: Device
        track_id: Optional specific track ID to analyze

    Returns:
        List of action detections
    """
    logger.info("Predicting actions for tracked players")

    # Load video
    with VideoReader(video_path) as reader:
        fps = reader.fps
        all_frames = list(reader.iter_frames())

    # Get track trajectories
    from inference.track import get_track_trajectories

    trajectories = get_track_trajectories(tracking_results)

    # Filter by track ID if specified
    if track_id is not None:
        trajectories = {track_id: trajectories.get(track_id, [])}

    all_results = []

    # Process each track
    for tid, trajectory in trajectories.items():
        logger.info(f"Processing track {tid} ({len(trajectory)} detections)")

        # Extract frames for this track
        track_frames = []
        frame_ids = []

        for det in trajectory:
            frame_id = det["frame_id"]
            if frame_id < len(all_frames):
                frame = all_frames[frame_id]

                # Optionally crop to bbox
                # bbox = det["box"]
                # frame = crop_bbox(frame, bbox)

                track_frames.append(frame)
                frame_ids.append(frame_id)

        if len(track_frames) < clip_length:
            logger.warning(f"Track {tid} has insufficient frames ({len(track_frames)})")
            continue

        # Create sliding windows
        windows = create_sliding_windows(len(track_frames), clip_length, temporal_window // 2)

        # Extract clips
        clips = []
        clip_info = []

        for start, end in windows:
            clip = track_frames[start:end]
            clips.append(clip)

            clip_info.append({
                "start_frame": frame_ids[start],
                "end_frame": frame_ids[end - 1] if end > 0 else frame_ids[0],
                "center_frame": frame_ids[(start + end) // 2] if (start + end) // 2 < len(frame_ids) else frame_ids[-1],
                "track_id": tid
            })

        # Predict
        predictions = predict_actions_from_clips(clips, model, device, batch_size=8)

        # Format results
        for pred, info in zip(predictions, clip_info):
            if pred["confidence"] >= conf_threshold:
                result = create_results_dict(
                    frame_id=info["center_frame"],
                    timestamp=info["center_frame"] / fps,
                    action=pred["class_name"],
                    confidence=pred["confidence"],
                    track_id=info["track_id"],
                    window_start=info["start_frame"],
                    window_end=info["end_frame"]
                )
                all_results.append(result)

    logger.info(f"Detected {len(all_results)} actions across all tracks")

    return all_results


def non_maximum_suppression_temporal(
    detections: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Apply NMS to temporal action detections

    Args:
        detections: List of action detections
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    keep = []

    while len(detections) > 0:
        # Keep highest confidence
        best = detections.pop(0)
        keep.append(best)

        # Remove overlapping detections
        filtered = []

        for det in detections:
            # Calculate temporal IoU
            start1 = best.get("window_start", best["frame_id"])
            end1 = best.get("window_end", best["frame_id"])
            start2 = det.get("window_start", det["frame_id"])
            end2 = det.get("window_end", det["frame_id"])

            intersection = max(0, min(end1, end2) - max(start1, start2))
            union = max(end1, end2) - min(start1, start2)
            iou = intersection / (union + 1e-6)

            # Keep if IoU is below threshold or different action
            if iou < iou_threshold or det["action"] != best["action"]:
                filtered.append(det)

        detections = filtered

    logger.info(f"NMS: kept {len(keep)} detections")

    return keep
