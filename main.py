"""
Tennis Action Analysis - Main Pipeline
Complete end-to-end pipeline for tennis action recognition
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import torch
from loguru import logger

from models.yolo_detector import TennisYOLODetector
from models.tracker import OCSort
from models.action_classifier import TennisActionClassifier, load_action_classifier
from models.pose_estimator import SimplePoseEstimator, load_pose_estimator

from inference.detect import detect_video
from inference.track import track_video
from inference.extract_clips import extract_player_clips, extract_action_clips
from inference.predict_actions import (
    predict_actions_from_video,
    predict_actions_from_tracked_clips,
    non_maximum_suppression_temporal
)
from inference.annotate_video import annotate_comprehensive

from utils.file_utils import load_yaml, save_json, save_results_csv, ensure_dir
from utils.video_utils import get_video_info


class TennisAnalysisPipeline:
    """
    Complete pipeline for tennis action analysis
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml(config_path)

        # Setup paths
        self.input_video = self.config["paths"]["input_video"]
        self.output_dir = Path(self.config["paths"]["output_dir"])
        ensure_dir(self.output_dir)

        # Setup device
        self.device = self.config["yolo"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"

        # Initialize models
        logger.info("Initializing models...")
        self._init_models()

        logger.info("Pipeline initialized")

    def _init_models(self):
        """Initialize all models"""
        # YOLO detector
        yolo_config = self.config["yolo"]
        yolo_weights = self.config["paths"].get("yolo_weights")

        self.detector = TennisYOLODetector(
            model_path=yolo_weights if yolo_weights and Path(yolo_weights).exists() else None,
            model_version=yolo_config["model_version"],
            model_size=yolo_config["model_size"],
            conf_threshold=yolo_config["conf_threshold"],
            iou_threshold=yolo_config["iou_threshold"],
            device=self.device,
            imgsz=yolo_config["imgsz"],
            class_names=yolo_config.get("classes", {0: "player", 1: "ball", 2: "racket"})
        )

        # Tracker
        tracking_config = self.config["tracking"]
        self.tracker = OCSort(
            det_thresh=tracking_config["track_high_thresh"],
            max_age=tracking_config["track_buffer"],
            min_hits=tracking_config["min_hits"],
            iou_threshold=tracking_config["iou_threshold"]
        )

        # Action classifier
        action_config = self.config["action_recognition"]
        action_weights = self.config["paths"].get("action_classifier_weights")

        if action_weights and Path(action_weights).exists():
            self.action_classifier = load_action_classifier(
                action_weights,
                model_type=action_config["model_type"],
                model_size=action_config["model_size"],
                num_classes=action_config["num_classes"],
                device=self.device
            )
        else:
            logger.warning("Action classifier weights not found, using pretrained model")
            self.action_classifier = TennisActionClassifier(
                model_type=action_config["model_type"],
                model_size=action_config["model_size"],
                num_classes=action_config["num_classes"],
                pretrained=True
            ).to(self.device)

        # Pose estimator (optional)
        if self.config["pose"]["enabled"]:
            pose_weights = self.config["paths"].get("pose_weights")

            if pose_weights and Path(pose_weights).exists():
                self.pose_estimator = load_pose_estimator(
                    pose_weights,
                    device=self.device
                )
            else:
                logger.warning("Pose estimator weights not found, using pretrained model")
                self.pose_estimator = SimplePoseEstimator(pretrained=True).to(self.device)
        else:
            self.pose_estimator = None

    def run(
        self,
        video_path: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        middle_segment: Optional[float] = None
    ):
        """
        Run complete pipeline

        Args:
            video_path: Optional video path (overrides config)
            start_time: Start time in seconds
            end_time: End time in seconds
            duration: Duration in seconds (from start_time)
            middle_segment: Extract middle N seconds of video
        """
        if video_path:
            self.input_video = video_path

        logger.info(f"Starting pipeline on video: {self.input_video}")

        # Get video info
        video_info = get_video_info(self.input_video)
        logger.info(
            f"Video info: {video_info['width']}x{video_info['height']}, "
            f"{video_info['fps']:.2f} fps, {video_info['duration']:.2f}s"
        )

        # Handle video segment extraction
        original_video = self.input_video
        temp_video = None

        if middle_segment is not None:
            # Extract middle segment
            total_duration = video_info['duration']
            start_time = (total_duration - middle_segment) / 2
            end_time = start_time + middle_segment
            logger.info(f"Extracting middle {middle_segment}s: {start_time:.1f}s to {end_time:.1f}s")

        if start_time is not None or end_time is not None or duration is not None:
            # Create temporary clipped video
            import tempfile
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

            if start_time is None:
                start_time = 0
            if duration is not None:
                end_time = start_time + duration
            if end_time is None:
                end_time = video_info['duration']

            logger.info(f"Processing video segment: {start_time:.1f}s to {end_time:.1f}s ({end_time - start_time:.1f}s)")

            # Extract segment using ffmpeg
            import subprocess
            cmd = [
                'ffmpeg', '-i', self.input_video,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c', 'copy',
                '-y',
                temp_video
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                self.input_video = temp_video
                # Update video info for the segment
                video_info = get_video_info(self.input_video)
                logger.info(f"Segment extracted: {video_info['duration']:.2f}s")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract video segment: {e}")
                self.input_video = original_video
                temp_video = None

        # Step 1: Detection
        logger.info("=" * 60)
        logger.info("STEP 1: Object Detection")
        logger.info("=" * 60)

        detections_json = self.output_dir / "detections.json"
        detections_list = detect_video(
            self.input_video,
            self.detector,
            output_json=str(detections_json),
            visualize=False
        )

        # Step 2: Tracking
        logger.info("=" * 60)
        logger.info("STEP 2: Multi-Object Tracking")
        logger.info("=" * 60)

        tracking_json = self.output_dir / "tracking.json"
        tracking_results = track_video(
            detections_list,
            self.tracker,
            output_json=str(tracking_json),
            visualize=False
        )

        # Step 3: Extract clips (optional)
        if self.config["clips"]["enabled"]:
            logger.info("=" * 60)
            logger.info("STEP 3: Extract Player Clips")
            logger.info("=" * 60)

            clips_dir = self.output_dir / "clips"
            player_clips = extract_player_clips(
                self.input_video,
                tracking_results,
                str(clips_dir)
            )

        # Step 4: Action Recognition
        logger.info("=" * 60)
        logger.info("STEP 4: Action Recognition")
        logger.info("=" * 60)

        action_config = self.config["action_recognition"]

        # Option A: Sliding window on full video
        # action_detections = predict_actions_from_video(
        #     self.input_video,
        #     self.action_classifier,
        #     clip_length=action_config["clip_duration"],
        #     clip_stride=action_config["clip_stride"],
        #     conf_threshold=action_config["conf_threshold"],
        #     device=self.device
        # )

        # Option B: Per-track analysis (better for tennis)
        action_detections = predict_actions_from_tracked_clips(
            self.input_video,
            tracking_results,
            self.action_classifier,
            clip_length=action_config["clip_duration"],
            temporal_window=action_config["temporal_window"],
            conf_threshold=action_config["conf_threshold"],
            device=self.device
        )

        # Apply temporal NMS
        action_detections = non_maximum_suppression_temporal(
            action_detections,
            iou_threshold=0.5
        )

        # Save action results
        if self.config["output"]["save_json"]:
            actions_json = self.output_dir / "actions.json"
            save_json(action_detections, str(actions_json))

        if self.config["output"]["save_csv"]:
            actions_csv = self.output_dir / "actions.csv"
            save_results_csv(
                action_detections,
                str(actions_csv),
                columns=["frame_id", "timestamp", "action", "confidence", "track_id"]
            )

        # Extract action clips
        if self.config["clips"]["enabled"]:
            logger.info("Extracting action clips...")
            action_clips_dir = self.output_dir / "action_clips"
            extract_action_clips(
                self.input_video,
                action_detections,
                str(action_clips_dir),
                margin_frames=self.config["clips"]["margin_frames"]
            )

        # Step 5: Generate annotated video
        if self.config["output"]["save_video"]:
            logger.info("=" * 60)
            logger.info("STEP 5: Generate Annotated Video")
            logger.info("=" * 60)

            output_video = self.output_dir / "output_annotated.mp4"

            annotate_comprehensive(
                self.input_video,
                str(output_video),
                detections_list=detections_list if self.config["visualization"]["show_bbox"] else None,
                tracking_results=tracking_results if self.config["visualization"]["show_track_id"] else None,
                action_detections=action_detections if self.config["visualization"]["show_action_label"] else None,
                show_tracks=self.config["visualization"]["show_track_id"],
                show_actions=self.config["visualization"]["show_action_label"],
                show_legend=True
            )

        # Summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {len(detections_list)}")
        logger.info(f"Total tracks: {len(set(t for r in tracking_results for t in r['tracks']['track_ids']))}")
        logger.info(f"Total actions detected: {len(action_detections)}")

        # Action breakdown
        action_counts = {}
        for det in action_detections:
            action = det["action"]
            action_counts[action] = action_counts.get(action, 0) + 1

        logger.info("\nAction breakdown:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {action}: {count}")

        logger.info(f"\nResults saved to: {self.output_dir}")

        # Cleanup temporary video if created
        if temp_video:
            import os
            try:
                os.unlink(temp_video)
                logger.debug(f"Cleaned up temporary file: {temp_video}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file {temp_video}: {e}")

        return {
            "detections": detections_list,
            "tracking": tracking_results,
            "actions": action_detections,
            "summary": {
                "total_frames": len(detections_list),
                "total_actions": len(action_detections),
                "action_counts": action_counts
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Tennis Action Analysis Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Input video path (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Start time in seconds"
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="End time in seconds"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration in seconds (from start-time)"
    )
    parser.add_argument(
        "--middle",
        type=float,
        help="Extract middle N seconds of video (e.g., --middle 300 for 5 minutes)"
    )
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )

    # Initialize pipeline
    pipeline = TennisAnalysisPipeline(args.config)

    # Override output dir if specified
    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir)
        ensure_dir(pipeline.output_dir)

    # Run pipeline
    results = pipeline.run(
        video_path=args.video,
        start_time=args.start_time,
        end_time=args.end_time,
        duration=args.duration,
        middle_segment=args.middle
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()