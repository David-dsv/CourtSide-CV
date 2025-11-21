"""
YOLO Detector for Tennis Analysis
Detects players, ball, and racket in tennis videos
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from loguru import logger


class TennisYOLODetector:
    """
    YOLOv8/v11 detector optimized for tennis scenarios
    Detects: player, ball, racket
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: str = "yolov8",
        model_size: str = "m",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cuda",
        imgsz: int = 640,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to custom trained weights (None = use pretrained)
            model_version: yolov8 or yolov11
            model_size: n, s, m, l, x
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: cuda or cpu
            imgsz: Input image size
            class_names: Dict mapping class ID to name
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz

        # Default class names for tennis
        self.class_names = class_names or {
            0: "player",
            1: "ball",
            2: "racket"
        }

        # Load model
        if model_path and Path(model_path).exists():
            logger.info(f"Loading custom YOLO model from {model_path}")
            self.model = YOLO(model_path)
        else:
            # Handle YOLO naming conventions
            # YOLOv8 uses 'v' (yolov8m.pt)
            # YOLO11 and YOLO12 don't use 'v' (yolo11m.pt, yolo12m.pt)
            if model_version == "yolov11":
                model_name = f"yolo11{model_size}.pt"  # No 'v' for YOLO11
            elif model_version == "yolov12":
                model_name = f"yolo12{model_size}.pt"  # No 'v' for YOLO12
            else:
                model_name = f"{model_version}{model_size}.pt"  # YOLOv8 keeps the 'v'
            logger.info(f"Loading pretrained YOLO model: {model_name}")
            self.model = YOLO(model_name)

        # Move to device
        self.model.to(self.device)
        logger.info(f"YOLO detector initialized on {self.device}")

    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> Dict:
        """
        Detect objects in a single frame

        Args:
            frame: Input image (BGR format)
            classes: List of class IDs to detect (None = all)

        Returns:
            Dictionary containing:
                - boxes: nx4 array [x1, y1, x2, y2]
                - scores: n array of confidence scores
                - class_ids: n array of class IDs
                - class_names: n list of class names
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=classes,
            verbose=False,
            device=self.device
        )[0]

        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.empty((0, 4))
        scores = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.empty(0)
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) > 0 else np.empty(0, dtype=int)

        # Map class names
        class_names_list = [self.class_names.get(cid, f"class_{cid}") for cid in class_ids]

        return {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_names": class_names_list,
            "frame_shape": frame.shape
        }

    def detect_batch(
        self,
        frames: List[np.ndarray],
        classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Detect objects in batch of frames

        Args:
            frames: List of input images
            classes: List of class IDs to detect

        Returns:
            List of detection dictionaries
        """
        results_list = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=classes,
            verbose=False,
            device=self.device
        )

        detections = []
        for results in results_list:
            boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.empty((0, 4))
            scores = results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else np.empty(0)
            class_ids = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) > 0 else np.empty(0, dtype=int)
            class_names_list = [self.class_names.get(cid, f"class_{cid}") for cid in class_ids]

            detections.append({
                "boxes": boxes,
                "scores": scores,
                "class_ids": class_ids,
                "class_names": class_names_list
            })

        return detections

    def filter_by_class(
        self,
        detections: Dict,
        class_id: int
    ) -> Dict:
        """
        Filter detections by class ID

        Args:
            detections: Detection dictionary
            class_id: Target class ID

        Returns:
            Filtered detection dictionary
        """
        mask = detections["class_ids"] == class_id

        return {
            "boxes": detections["boxes"][mask],
            "scores": detections["scores"][mask],
            "class_ids": detections["class_ids"][mask],
            "class_names": [detections["class_names"][i] for i, m in enumerate(mask) if m]
        }

    def get_largest_detection(
        self,
        detections: Dict,
        class_id: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Get largest detection (by area), optionally filtered by class

        Args:
            detections: Detection dictionary
            class_id: Optional class ID filter

        Returns:
            Single detection dict or None
        """
        if class_id is not None:
            detections = self.filter_by_class(detections, class_id)

        if len(detections["boxes"]) == 0:
            return None

        # Calculate areas
        boxes = detections["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = np.argmax(areas)

        return {
            "box": boxes[largest_idx],
            "score": detections["scores"][largest_idx],
            "class_id": detections["class_ids"][largest_idx],
            "class_name": detections["class_names"][largest_idx]
        }

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "tennis_yolo",
        **kwargs
    ):
        """
        Train YOLO model on custom dataset

        Args:
            data_yaml: Path to data.yaml config file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Input image size
            project: Project directory
            name: Experiment name
            **kwargs: Additional training arguments
        """
        logger.info(f"Starting YOLO training for {epochs} epochs")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            device=self.device,
            **kwargs
        )

        logger.info("Training completed")
        return results

    def export_model(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None
    ):
        """
        Export model to different formats

        Args:
            format: Export format (onnx, torchscript, etc.)
            output_path: Output file path
        """
        logger.info(f"Exporting model to {format} format")
        self.model.export(format=format, output=output_path)
        logger.info(f"Model exported successfully")
