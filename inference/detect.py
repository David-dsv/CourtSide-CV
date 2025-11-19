"""
Detection Module
Run YOLO detection on video frames
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
from loguru import logger

from models.yolo_detector import TennisYOLODetector
from utils.video_utils import VideoReader, VideoWriter
from utils.drawing import draw_bboxes
from utils.file_utils import save_json


def detect_video(
    video_path: str,
    detector: TennisYOLODetector,
    output_video: Optional[str] = None,
    output_json: Optional[str] = None,
    visualize: bool = True,
    classes_filter: Optional[List[int]] = None
) -> List[Dict]:
    """
    Run detection on entire video

    Args:
        video_path: Input video path
        detector: YOLO detector
        output_video: Optional output video path
        output_json: Optional output JSON path
        visualize: Draw detections on frames
        classes_filter: Filter by class IDs

    Returns:
        List of detection results per frame
    """
    logger.info(f"Running detection on {video_path}")

    results = []

    with VideoReader(video_path) as reader:
        # Setup video writer if needed
        writer = None
        if output_video:
            writer = VideoWriter(
                output_video,
                fps=reader.fps,
                frame_size=(reader.width, reader.height)
            )

        # Process frames
        for frame_id, frame in enumerate(tqdm(reader.iter_frames(), total=len(reader))):
            # Detect
            detections = detector.detect(frame, classes=classes_filter)

            # Store results
            frame_result = {
                "frame_id": frame_id,
                "timestamp": frame_id / reader.fps,
                "detections": {
                    "boxes": detections["boxes"].tolist(),
                    "scores": detections["scores"].tolist(),
                    "class_ids": detections["class_ids"].tolist(),
                    "class_names": detections["class_names"]
                }
            }
            results.append(frame_result)

            # Visualize
            if visualize and writer:
                labels = [
                    f"{name} {score:.2f}"
                    for name, score in zip(detections["class_names"], detections["scores"])
                ]
                frame_vis = draw_bboxes(
                    frame.copy(),
                    detections["boxes"],
                    labels=labels
                )
                writer.write_frame(frame_vis)

        # Release writer
        if writer:
            writer.release()

    # Save JSON
    if output_json:
        save_json(results, output_json)

    logger.info(f"Detection completed: {len(results)} frames processed")

    return results


def detect_batch(
    frames: List[np.ndarray],
    detector: TennisYOLODetector,
    classes_filter: Optional[List[int]] = None
) -> List[Dict]:
    """
    Run detection on batch of frames

    Args:
        frames: List of frames
        detector: YOLO detector
        classes_filter: Filter by class IDs

    Returns:
        List of detection results
    """
    detections_list = detector.detect_batch(frames, classes=classes_filter)
    return detections_list
