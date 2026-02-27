"""
Tracker Module - ByteTrack wrapper using roboflow/trackers 2.2.0
Provides a consistent numpy-array interface for multi-object tracking.
"""

import numpy as np
from trackers import ByteTrackTracker
import supervision as sv
from loguru import logger


class ByteTrackWrapper:
    """
    Wrapper around trackers.ByteTrackTracker that provides a numpy-array
    interface compatible with the existing pipeline.

    update() accepts raw numpy arrays (boxes, scores, class_ids) and returns
    a numpy array of shape [N, 7]: [x1, y1, x2, y2, track_id, score, class_id]
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6
    ):
        self.tracker = ByteTrackTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
            high_conf_det_threshold=high_conf_det_threshold
        )
        logger.info(
            f"ByteTrackWrapper initialized: activation={track_activation_threshold}, "
            f"buffer={lost_track_buffer}, high_conf={high_conf_det_threshold}, fps={frame_rate}"
        )

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray
    ) -> np.ndarray:
        """
        Update tracker with new detections for one frame.

        Args:
            boxes: Detection boxes [N, 4] as [x1, y1, x2, y2]
            scores: Detection confidence scores [N]
            class_ids: Detection class IDs [N]

        Returns:
            Tracked objects as numpy array [M, 7]:
                [x1, y1, x2, y2, track_id, score, class_id]
            Returns empty array with shape (0, 7) if no tracks.
        """
        if len(boxes) == 0:
            return np.empty((0, 7))

        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.array(scores, dtype=np.float32).flatten(),
            class_id=np.array(class_ids, dtype=int).flatten()
        )

        tracked = self.tracker.update(detections)

        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            return np.empty((0, 7))

        result = np.column_stack([
            tracked.xyxy,
            tracked.tracker_id.astype(np.float64),
            tracked.confidence.astype(np.float64),
            tracked.class_id.astype(np.float64)
        ])

        return result

    def reset(self):
        """Reset tracker state (e.g. between videos)."""
        self.tracker.reset()


# Backward-compatible alias
OCSort = ByteTrackWrapper
