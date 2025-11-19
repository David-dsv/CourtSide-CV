"""
Object Tracker using OCSORT algorithm
Multi-object tracking optimized for tennis players
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from loguru import logger


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes in image space
    State: [x, y, s, r, vx, vy, vs]
    x, y: center coordinates
    s: scale (area)
    r: aspect ratio (width/height)
    vx, vy, vs: velocities
    """
    count = 0

    def __init__(self, bbox: np.ndarray, score: float = 1.0, class_id: int = 0):
        """
        Initialize tracker with bounding box

        Args:
            bbox: [x1, y1, x2, y2]
            score: Detection confidence
            class_id: Object class ID
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Process noise
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = score
        self.class_id = class_id

    def update(self, bbox: np.ndarray, score: float = 1.0):
        """
        Update tracker with new detection

        Args:
            bbox: [x1, y1, x2, y2]
            score: Detection confidence
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        self.score = score

    def predict(self) -> np.ndarray:
        """
        Predict next state

        Returns:
            Predicted bbox [x1, y1, x2, y2]
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        bbox = self._convert_x_to_bbox(self.kf.x)
        self.history.append(bbox)
        return bbox

    def get_state(self) -> np.ndarray:
        """
        Get current bounding box estimate

        Returns:
            Current bbox [x1, y1, x2, y2]
        """
        bbox = self._convert_x_to_bbox(self.kf.x)
        # Ensure bbox is always returned as 1D array
        return bbox.flatten()

    @staticmethod
    def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        Convert [x1, y1, x2, y2] to [x, y, s, r]

        Args:
            bbox: [x1, y1, x2, y2]

        Returns:
            [x, y, s, r] where x,y is center, s is scale, r is aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
        """
        Convert [x, y, s, r] to [x1, y1, x2, y2]

        Args:
            x: State [x, y, s, r, ...]

        Returns:
            bbox [x1, y1, x2, y2]
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / (w + 1e-6)
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0
        ]).flatten()


class OCSort:
    """
    Observation-Centric SORT tracker
    Improved tracking for occlusion scenarios
    """

    def __init__(
        self,
        det_thresh: float = 0.6,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False
    ):
        """
        Initialize OCSORT tracker

        Args:
            det_thresh: Detection confidence threshold
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
            delta_t: Time steps for velocity estimation
            asso_func: Association function (iou or giou)
            inertia: Weight of previous velocity in prediction
            use_byte: Use BYTE association strategy
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = asso_func
        self.inertia = inertia
        self.use_byte = use_byte

        KalmanBoxTracker.count = 0

        logger.info(f"OCSORT tracker initialized with max_age={max_age}, min_hits={min_hits}")

    def update(
        self,
        detections: np.ndarray,
        scores: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections

        Args:
            detections: nx4 array of detections [x1, y1, x2, y2]
            scores: n array of detection scores
            class_ids: n array of class IDs

        Returns:
            nx6 array of tracked objects [x1, y1, x2, y2, track_id, score]
        """
        self.frame_count += 1

        if scores is None:
            scores = np.ones(len(detections))
        if class_ids is None:
            class_ids = np.zeros(len(detections), dtype=int)

        # Get predictions from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t in range(len(self.trackers)):
            # Get prediction from tracker
            pos = self.trackers[t].predict()
            
            # FIX: Ensure pos is properly formatted as array
            # Handle the case where predict() might return different formats
            if isinstance(pos, (list, tuple)):
                pos = pos[0] if len(pos) > 0 else pos
            
            # Convert to numpy array and flatten
            pos = np.array(pos).flatten()
            
            # Check if we have a valid prediction with at least 4 values
            if pos.size < 4:
                to_del.append(t)
                continue
            
            # Check for NaN values
            if np.any(np.isnan(pos[:4])):
                to_del.append(t)
                continue
            
            # Store the tracker prediction
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]

        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Filter detections by threshold
        dets = []
        indices = []
        for i, (det, score) in enumerate(zip(detections, scores)):
            if score >= self.det_thresh:
                dets.append(det)
                indices.append(i)

        dets = np.array(dets) if len(dets) > 0 else np.empty((0, 4))
        scores_filtered = scores[indices] if len(indices) > 0 else np.array([])
        class_ids_filtered = class_ids[indices] if len(indices) > 0 else np.array([])

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], scores_filtered[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], scores_filtered[i], class_ids_filtered[i])
            self.trackers.append(trk)

        # Return confirmed tracks
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue

            # Get the current state
            d = trk.get_state()
            
            # Ensure d is a 1D numpy array
            d = np.array(d).flatten()

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Ensure d has at least 4 elements before using it
                if d.size >= 4:
                    # Create track data with bbox, track_id, score, and class_id
                    track_data = np.concatenate([
                        d[:4],  # Only use first 4 elements (x1, y1, x2, y2)
                        np.array([trk.id + 1, trk.score, trk.class_id])
                    ])
                    ret.append(track_data.reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def _associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
        iou_threshold: float = 0.3
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Assign detections to tracked objects using Hungarian algorithm

        Args:
            detections: nx4 array of detections
            trackers: mx4 array of tracker predictions
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (matched_indices, unmatched_detections, unmatched_trackers)
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), list(range(len(detections))), []

        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), [], list(range(len(trackers)))

        # Compute IoU matrix
        iou_matrix = self._iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self._linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, unmatched_detections, unmatched_trackers

    @staticmethod
    def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """
        Compute IoU between two sets of boxes

        Args:
            bb_test: nx4 array [x1, y1, x2, y2]
            bb_gt: mx4 array [x1, y1, x2, y2]

        Returns:
            nxm IoU matrix
        """
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h

        area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

        iou = wh / (area_test + area_gt - wh + 1e-6)
        return iou

    @staticmethod
    def _linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
        """
        Solve linear assignment problem

        Args:
            cost_matrix: Cost matrix

        Returns:
            Matched indices
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.column_stack((row_ind, col_ind))

    def reset(self):
        """Reset tracker state"""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
        logger.info("Tracker reset")