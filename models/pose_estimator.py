"""
Pose Estimation for Tennis Players
Detects keypoints for enhanced action recognition
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
from loguru import logger


class SimplePoseEstimator(nn.Module):
    """
    Lightweight pose estimator for real-time tennis analysis
    Based on ResNet backbone with heatmap prediction
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        backbone: str = "resnet50",
        pretrained: bool = True
    ):
        """
        Initialize pose estimator

        Args:
            num_keypoints: Number of keypoints (17 for COCO format)
            backbone: Backbone architecture
            pretrained: Use pretrained weights
        """
        super().__init__()

        self.num_keypoints = num_keypoints

        # COCO keypoint names
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # Skeleton connections for visualization
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Build backbone
        if backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            if pretrained:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50()

            # Remove fully connected layers
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_out_channels = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Deconvolution layers for upsampling
        self.deconv_layers = self._make_deconv_layers(
            backbone_out_channels,
            [256, 256, 256],
            [4, 4, 4]
        )

        # Final heatmap prediction layer
        self.final_layer = nn.Conv2d(
            256,
            num_keypoints,
            kernel_size=1,
            stride=1,
            padding=0
        )

        logger.info(f"Pose estimator initialized with {num_keypoints} keypoints")

    def _make_deconv_layers(
        self,
        in_channels: int,
        num_filters: List[int],
        num_kernels: List[int]
    ) -> nn.Sequential:
        """Create deconvolution layers"""
        layers = []

        for i in range(len(num_filters)):
            kernel = num_kernels[i]
            padding = 1
            output_padding = 0

            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters[i],
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))

            in_channels = num_filters[i]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Heatmaps [B, num_keypoints, H/4, W/4]
        """
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def predict(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        conf_threshold: float = 0.3
    ) -> Dict:
        """
        Predict keypoints from image

        Args:
            image: Input image [H, W, 3] in BGR
            bbox: Optional bounding box [x1, y1, x2, y2] to crop person
            conf_threshold: Confidence threshold for keypoints

        Returns:
            Dictionary with keypoints, scores, heatmaps
        """
        self.eval()

        # Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            cropped = image[y1:y2, x1:x2]
            offset = np.array([x1, y1])
        else:
            cropped = image
            offset = np.array([0, 0])

        # Preprocess
        input_tensor = self._preprocess(cropped)

        # Inference
        with torch.no_grad():
            heatmaps = self.forward(input_tensor)

        # Post-process
        keypoints, scores = self._postprocess(heatmaps, cropped.shape[:2])

        # Adjust keypoints to original image coordinates
        keypoints = keypoints + offset

        # Filter by confidence
        valid_mask = scores >= conf_threshold

        return {
            "keypoints": keypoints,
            "scores": scores,
            "valid_mask": valid_mask,
            "heatmaps": heatmaps.cpu().numpy()
        }

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize to model input size
        img_resized = cv2.resize(image, (256, 256))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize
        img_normalized = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_normalized - mean) / std

        # To tensor [1, 3, H, W]
        tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()

        return tensor

    def _postprocess(
        self,
        heatmaps: torch.Tensor,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract keypoints from heatmaps

        Args:
            heatmaps: Predicted heatmaps [1, num_keypoints, H, W]
            original_size: Original image size (H, W)

        Returns:
            Tuple of (keypoints, scores)
            keypoints: [num_keypoints, 2] (x, y)
            scores: [num_keypoints]
        """
        heatmaps = heatmaps[0].cpu().numpy()  # [num_keypoints, H, W]

        num_keypoints = heatmaps.shape[0]
        heatmap_h, heatmap_w = heatmaps.shape[1:]

        keypoints = np.zeros((num_keypoints, 2))
        scores = np.zeros(num_keypoints)

        for i in range(num_keypoints):
            heatmap = heatmaps[i]

            # Find max location
            flat_idx = np.argmax(heatmap)
            y, x = np.unravel_index(flat_idx, heatmap.shape)

            # Confidence score
            scores[i] = heatmap[y, x]

            # Scale to original image size
            keypoints[i, 0] = x * original_size[1] / heatmap_w
            keypoints[i, 1] = y * original_size[0] / heatmap_h

        return keypoints, scores


class PoseTracker:
    """
    Track poses across frames for temporal consistency
    """

    def __init__(
        self,
        estimator: SimplePoseEstimator,
        smooth_alpha: float = 0.5
    ):
        """
        Initialize pose tracker

        Args:
            estimator: Pose estimator model
            smooth_alpha: Smoothing factor for temporal filtering
        """
        self.estimator = estimator
        self.smooth_alpha = smooth_alpha
        self.prev_keypoints = None

    def track(
        self,
        image: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        conf_threshold: float = 0.3
    ) -> Dict:
        """
        Track pose with temporal smoothing

        Args:
            image: Input image
            bbox: Bounding box
            conf_threshold: Confidence threshold

        Returns:
            Pose prediction dict
        """
        # Get current pose
        result = self.estimator.predict(image, bbox, conf_threshold)

        # Temporal smoothing
        if self.prev_keypoints is not None:
            smoothed = (
                self.smooth_alpha * result["keypoints"] +
                (1 - self.smooth_alpha) * self.prev_keypoints
            )
            result["keypoints"] = smoothed

        self.prev_keypoints = result["keypoints"].copy()

        return result

    def reset(self):
        """Reset tracker state"""
        self.prev_keypoints = None


def load_pose_estimator(
    checkpoint_path: str,
    num_keypoints: int = 17,
    device: str = "cuda"
) -> SimplePoseEstimator:
    """
    Load pretrained pose estimator

    Args:
        checkpoint_path: Path to checkpoint
        num_keypoints: Number of keypoints
        device: Device to load on

    Returns:
        Loaded model
    """
    model = SimplePoseEstimator(num_keypoints=num_keypoints, pretrained=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Pose estimator loaded from {checkpoint_path}")
    return model
