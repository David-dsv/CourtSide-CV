"""
Tracking Module
Track detected objects across frames
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from models.tracker import OCSort
from utils.video_utils import VideoReader, VideoWriter
from utils.drawing import draw_tracks
from utils.file_utils import save_json


def track_video(
    detections_list: List[Dict],
    tracker: OCSort,
    video_path: Optional[str] = None,
    output_video: Optional[str] = None,
    output_json: Optional[str] = None,
    visualize: bool = True
) -> List[Dict]:
    """
    Track objects across video using detections

    Args:
        detections_list: List of detection results per frame
        tracker: Tracker instance
        video_path: Optional input video path for visualization
        output_video: Optional output video path
        output_json: Optional output JSON path
        visualize: Draw tracks on frames

    Returns:
        List of tracking results per frame
    """
    logger.info("Running tracking on detections")

    tracking_results = []

    # Setup video reader/writer if visualization needed
    reader = None
    writer = None

    if visualize and video_path and output_video:
        reader = VideoReader(video_path)
        writer = VideoWriter(
            output_video,
            fps=reader.fps,
            frame_size=(reader.width, reader.height)
        )

    # Process each frame
    for frame_id, det_result in enumerate(tqdm(detections_list)):
        detections = det_result["detections"]

        # Convert to arrays
        boxes = np.array(detections["boxes"])
        scores = np.array(detections["scores"])
        class_ids = np.array(detections["class_ids"])

        # Update tracker
        tracks = tracker.update(boxes, scores, class_ids)

        # Store results - handle different output formats
        if len(tracks) > 0:
            # Check number of columns
            num_cols = tracks.shape[1]

            if num_cols >= 7:
                # Full format: [x1, y1, x2, y2, track_id, score, class_id]
                track_data = {
                    "boxes": tracks[:, :4].tolist(),
                    "track_ids": tracks[:, 4].tolist(),
                    "scores": tracks[:, 5].tolist(),
                    "class_ids": tracks[:, 6].tolist()
                }
            elif num_cols >= 6:
                # Without class_id: [x1, y1, x2, y2, track_id, score]
                track_data = {
                    "boxes": tracks[:, :4].tolist(),
                    "track_ids": tracks[:, 4].tolist(),
                    "scores": tracks[:, 5].tolist(),
                    "class_ids": [0] * len(tracks)  # Default class
                }
            elif num_cols >= 5:
                # Minimal: [x1, y1, x2, y2, track_id]
                track_data = {
                    "boxes": tracks[:, :4].tolist(),
                    "track_ids": tracks[:, 4].tolist(),
                    "scores": [1.0] * len(tracks),  # Default score
                    "class_ids": [0] * len(tracks)  # Default class
                }
            else:
                # Only boxes
                track_data = {
                    "boxes": tracks[:, :4].tolist() if num_cols >= 4 else [],
                    "track_ids": list(range(len(tracks))),
                    "scores": [1.0] * len(tracks),
                    "class_ids": [0] * len(tracks)
                }
        else:
            track_data = {
                "boxes": [],
                "track_ids": [],
                "scores": [],
                "class_ids": []
            }

        frame_result = {
            "frame_id": frame_id,
            "timestamp": det_result["timestamp"],
            "tracks": track_data
        }
        tracking_results.append(frame_result)

        # Visualize
        if visualize and reader and writer:
            ret, frame = reader.read_frame()
            if ret:
                frame_vis = draw_tracks(frame.copy(), tracks)
                writer.write_frame(frame_vis)

    # Release resources
    if reader:
        reader.release()
    if writer:
        writer.release()

    # Save JSON
    if output_json:
        save_json(tracking_results, output_json)

    logger.info(f"Tracking completed: {len(tracking_results)} frames processed")

    return tracking_results


def get_track_by_id(
    tracking_results: List[Dict],
    track_id: int
) -> List[Dict]:
    """
    Extract all detections for a specific track ID

    Args:
        tracking_results: List of tracking results
        track_id: Target track ID

    Returns:
        List of track instances
    """
    track_instances = []

    for frame_result in tracking_results:
        tracks = frame_result["tracks"]

        if len(tracks["track_ids"]) == 0:
            continue

        # Find track
        for i, tid in enumerate(tracks["track_ids"]):
            if tid == track_id:
                track_instances.append({
                    "frame_id": frame_result["frame_id"],
                    "timestamp": frame_result["timestamp"],
                    "box": tracks["boxes"][i],
                    "score": tracks["scores"][i],
                    "class_id": tracks["class_ids"][i]
                })

    return track_instances


def get_track_trajectories(
    tracking_results: List[Dict]
) -> Dict[int, List[Dict]]:
    """
    Get all track trajectories

    Args:
        tracking_results: List of tracking results

    Returns:
        Dictionary mapping track_id to list of instances
    """
    trajectories = {}

    for frame_result in tracking_results:
        tracks = frame_result["tracks"]

        for i, track_id in enumerate(tracks["track_ids"]):
            track_id = int(track_id)

            if track_id not in trajectories:
                trajectories[track_id] = []

            trajectories[track_id].append({
                "frame_id": frame_result["frame_id"],
                "timestamp": frame_result["timestamp"],
                "box": tracks["boxes"][i],
                "score": tracks["scores"][i],
                "class_id": tracks["class_ids"][i]
            })

    return trajectories
