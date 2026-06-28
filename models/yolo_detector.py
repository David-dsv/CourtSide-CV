"""
YOLO Detector Module
Uses the fine-tuned CourtSide-CV model from HuggingFace for tennis-specific
objects (ball, racket, court zones) and a generic YOLO model for player detection.

Outputs use COCO-compatible class IDs (0=person, 32=sports_ball) so the
downstream pipeline (tracking, action recognition, annotation) works unchanged.
"""

import numpy as np
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from loguru import logger
from typing import Dict, List, Optional


# Mapping from custom model classes to pipeline-expected IDs
# Custom model: 0=racket, 1=tennis_ball, 2-9=court zones
# Pipeline expects COCO IDs: 0=person, 32=sports_ball
CUSTOM_TO_PIPELINE = {
    0: 38,    # racket -> COCO racket-like ID (unused by tracking, kept for annotation)
    1: 32,    # tennis_ball -> sports_ball (used by tracking)
    2: 102,   # bottom-dead-zone
    3: 103,   # court
    4: 104,   # left-doubles-alley
    5: 105,   # left-service-box
    6: 106,   # net
    7: 107,   # right-doubles-alley
    8: 108,   # right-service-box
    9: 109,   # top-dead-zone
}

COURT_ZONE_NAMES = {
    2: "bottom-dead-zone",
    3: "court",
    4: "left-doubles-alley",
    5: "left-service-box",
    6: "net",
    7: "right-doubles-alley",
    8: "right-service-box",
    9: "top-dead-zone",
}

CUSTOM_CLASS_NAMES = {
    0: "racket",
    1: "tennis_ball",
    2: "bottom-dead-zone",
    3: "court",
    4: "left-doubles-alley",
    5: "left-service-box",
    6: "net",
    7: "right-doubles-alley",
    8: "right-service-box",
    9: "top-dead-zone",
}

# HuggingFace model ID
HUGGINGFACE_MODEL = "Davidsv/CourtSide-Computer-Vision-v1"


class TennisYOLODetector:
    """
    Tennis-specific YOLO detector combining:
    - Fine-tuned model (HuggingFace) for ball, racket, court zones
    - Generic YOLO for player (person) detection

    All outputs use COCO-compatible class IDs for downstream compatibility.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: str = "yolov8",
        model_size: str = "m",
        conf_threshold: float = 0.2,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
        ball_conf: Optional[float] = None,
        ball_imgsz: Optional[int] = None,
        class_names: Optional[Dict] = None
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.imgsz = imgsz
        # The tennis ball is tiny/faint and sat below the PERSON conf (the ball
        # model ran at the shared conf_threshold). Measured vs the human ball GT
        # on the real prod path (imgsz 1280 + Kalman gating): the ball-conf
        # default 0.16 lifts demo3 coverage 82.6%->86.2% with CORRECT flat
        # (79.1%), and felix bounce F1 0.667->0.750 (>=floor) with margin from
        # felix's 0.13 cliff — without the demo3 CORRECT dip / felix knife-edge
        # that conf 0.10 showed. So the BALL model gets its OWN low conf +
        # (optional) imgsz, decoupled from the person detector (its 0.2 conf
        # stays; lowering it would flood player FPs). Defaults preserve the
        # legacy single-threshold behaviour when unset.
        # See docs/research/ball-track-density-CR.md.
        self.ball_conf = conf_threshold if ball_conf is None else ball_conf
        self.ball_imgsz = imgsz if ball_imgsz is None else ball_imgsz

        # Load fine-tuned model from HuggingFace (or local path)
        # Check for YOLO26 ball model first
        yolo26_path = Path(__file__).parent.parent / "training" / "yolo26_ball_best.pt"
        self.is_yolo26_ball = False

        if model_path and Path(model_path).exists():
            logger.info(f"Loading custom model from local: {model_path}")
            self.custom_model = YOLO(model_path)
            # Detect if it's a single-class YOLO26 ball model
            if len(self.custom_model.names) == 1:
                self.is_yolo26_ball = True
                logger.info("Single-class YOLO26 ball model detected")
        elif yolo26_path.exists():
            logger.info(f"Loading YOLO26 ball model: {yolo26_path}")
            self.custom_model = YOLO(str(yolo26_path))
            self.is_yolo26_ball = True
        else:
            logger.info(f"Downloading fine-tuned model from HuggingFace: {HUGGINGFACE_MODEL}")
            weights_path = hf_hub_download(
                repo_id=HUGGINGFACE_MODEL,
                filename="model.pt"
            )
            logger.info(f"Model downloaded to: {weights_path}")
            self.custom_model = YOLO(weights_path)

        # Load generic YOLO for person detection
        generic_model_name = f"{model_version}{model_size}.pt"
        logger.info(f"Loading generic YOLO for person detection: {generic_model_name}")
        self.person_model = YOLO(generic_model_name)

        logger.info(
            f"TennisYOLODetector initialized: device={device}, "
            f"conf={conf_threshold}, imgsz={imgsz}"
        )

    def detect(
        self,
        frame: np.ndarray,
        classes: Optional[List[int]] = None
    ) -> Dict:
        """
        Run detection on a single frame.

        Combines results from both the custom tennis model and the generic
        person detector. Returns COCO-compatible class IDs.

        Args:
            frame: Input frame (BGR)
            classes: Optional filter by COCO class IDs (0=person, 32=ball).
                     If None, returns all detections.

        Returns:
            Dict with keys: boxes, scores, class_ids, class_names
        """
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_class_names = []

        # Detect persons with generic model (class 0 in COCO)
        want_persons = classes is None or 0 in classes
        if want_persons:
            person_results = self.person_model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                imgsz=self.imgsz,
                classes=[0],  # COCO person
                verbose=False
            )
            if person_results[0].boxes is not None and len(person_results[0].boxes) > 0:
                boxes = person_results[0].boxes.xyxy.cpu().numpy()
                scores = person_results[0].boxes.conf.cpu().numpy()
                for i in range(len(boxes)):
                    all_boxes.append(boxes[i])
                    all_scores.append(float(scores[i]))
                    all_class_ids.append(0)  # COCO person
                    all_class_names.append("person")

        # Detect tennis objects with custom model
        want_ball = classes is None or 32 in classes
        if want_ball:
            if self.is_yolo26_ball:
                # YOLO26 single-class model: class 0 = tennis_ball.
                # Ball runs at its OWN low conf + (optional higher) imgsz so the
                # tiny/faint far ball isn't dropped by the person threshold.
                custom_results = self.custom_model(
                    frame,
                    conf=self.ball_conf,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=self.ball_imgsz,
                    classes=[0],
                    verbose=False
                )
                if custom_results[0].boxes is not None and len(custom_results[0].boxes) > 0:
                    boxes = custom_results[0].boxes.xyxy.cpu().numpy()
                    scores = custom_results[0].boxes.conf.cpu().numpy()
                    for i in range(len(boxes)):
                        all_boxes.append(boxes[i])
                        all_scores.append(float(scores[i]))
                        all_class_ids.append(32)  # Map to COCO sports_ball
                        all_class_names.append("tennis_ball")
            else:
                # HuggingFace multi-class model: 0=racket, 1=tennis_ball, 2-9=zones
                custom_results = self.custom_model(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    imgsz=self.imgsz,
                    classes=[0, 1],
                    verbose=False
                )
                if custom_results[0].boxes is not None and len(custom_results[0].boxes) > 0:
                    boxes = custom_results[0].boxes.xyxy.cpu().numpy()
                    scores = custom_results[0].boxes.conf.cpu().numpy()
                    cls_ids = custom_results[0].boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(boxes)):
                        custom_cls = int(cls_ids[i])
                        pipeline_cls = CUSTOM_TO_PIPELINE.get(custom_cls, custom_cls)
                        name = CUSTOM_CLASS_NAMES.get(custom_cls, f"class_{custom_cls}")
                        all_boxes.append(boxes[i])
                        all_scores.append(float(scores[i]))
                        all_class_ids.append(pipeline_cls)
                        all_class_names.append(name)

        # Build output
        if len(all_boxes) > 0:
            return {
                "boxes": np.array(all_boxes),
                "scores": np.array(all_scores),
                "class_ids": np.array(all_class_ids),
                "class_names": all_class_names
            }
        else:
            return {
                "boxes": np.empty((0, 4)),
                "scores": np.empty(0),
                "class_ids": np.empty(0, dtype=int),
                "class_names": []
            }

    def detect_court_zones(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.3
    ) -> Dict[str, Dict]:
        """
        Detect court zones using the fine-tuned model.

        Args:
            frame: Input frame (BGR)
            conf_threshold: Minimum confidence for court zone detections

        Returns:
            Dict of {zone_name: {'box': [x1,y1,x2,y2], 'score': float}}.
            Keeps only the highest-confidence detection per zone type.
        """
        results = self.custom_model(
            frame,
            conf=conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            imgsz=self.imgsz,
            classes=list(range(2, 10)),  # court zone classes only
            verbose=False
        )

        zones = {}
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cls = int(cls_ids[i])
                zone_name = COURT_ZONE_NAMES.get(cls)
                if zone_name is None:
                    continue
                score = float(scores[i])
                # Keep only the highest-confidence detection per zone
                if zone_name not in zones or score > zones[zone_name]["score"]:
                    zones[zone_name] = {
                        "box": boxes[i].tolist(),
                        "score": score
                    }

        return zones

    def detect_batch(
        self,
        frames: List[np.ndarray],
        classes: Optional[List[int]] = None
    ) -> List[Dict]:
        """Run detection on a batch of frames."""
        return [self.detect(frame, classes) for frame in frames]

    def probe_ball_density(
        self,
        frames: List[np.ndarray],
        probe_conf: float = 0.05,
        probe_imgsz: int = 1280,
        subsample: int = 10,
    ) -> float:
        """Mean ball-candidate count per probed frame — a footage-cleanliness
        metric used to choose the ball detection resolution (no GT, no per-video
        constant).

        WHY: the tiny/far ball wins big from a high-res pass (imgsz 1920 +
        low conf) on CLEAN broadcast — but the SAME high-res/low-conf floods a
        parasite-heavy oblique/amateur clip with false candidates the tracker
        then locks onto (measured: demo3 ~2 cand/frame, felix ~7). The candidate
        DENSITY at a low probe floor separates the two robustly (a ~3.4x gap,
        stable across probe conf and subsample), so it can gate the resolution
        choice: clean -> 1920/low-conf, parasite -> 1280/safe-conf. See
        docs/research/s1-ball-sota-CR.md.

        Runs the ball model at a LOW probe conf on every `subsample`-th frame and
        returns the mean number of candidates per probed frame. Cheap: ~1/10th of
        a full pass. Measured stable down to subsample=10.
        """
        probe = frames[::max(1, subsample)]
        if not probe:
            return 0.0
        counts = []
        for frame in probe:
            res = self.custom_model(
                frame,
                conf=probe_conf,
                iou=self.iou_threshold,
                device=self.device,
                imgsz=probe_imgsz,
                classes=[0] if self.is_yolo26_ball else [1],
                verbose=False,
            )
            b = res[0].boxes
            counts.append(0 if b is None else len(b))
        return float(np.mean(counts)) if counts else 0.0
