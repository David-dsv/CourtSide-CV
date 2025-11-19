"""
Action Recognition Model for Tennis
Supports X3D and SlowFast architectures for temporal action classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from loguru import logger

# Try to import pytorchvideo, provide graceful fallback
try:
    from pytorchvideo.models import x3d, slowfast
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False
    logger.warning(
        "pytorchvideo not installed. Action classification will not work. "
        "Install with: pip install fvcore pytorchvideo"
    )


class TennisActionClassifier(nn.Module):
    """
    Action classifier for tennis strokes
    Classes: forehand, backhand, slice, serve, smash, volley
    """

    def __init__(
        self,
        model_type: str = "x3d",
        model_size: str = "m",
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize action classifier

        Args:
            model_type: x3d or slowfast
            model_size: s, m, l
            num_classes: Number of action classes
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()

        self.model_type = model_type
        self.model_size = model_size
        self.num_classes = num_classes

        # Class names
        self.class_names = [
            "forehand",
            "backhand",
            "slice",
            "serve",
            "smash",
            "volley"
        ]

        # Check if pytorchvideo is available
        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError(
                "pytorchvideo is required for action classification. "
                "Install with: pip install fvcore pytorchvideo"
            )

        # Build backbone
        if model_type == "x3d":
            self.backbone = self._build_x3d(model_size, pretrained)
        elif model_type == "slowfast":
            self.backbone = self._build_slowfast(model_size, pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Replace final classification head
        self._replace_head(dropout)

        logger.info(f"Action classifier initialized: {model_type}_{model_size} with {num_classes} classes")

    def _build_x3d(self, size: str, pretrained: bool) -> nn.Module:
        """Build X3D model"""
        if size == "s":
            model = x3d.create_x3d(
                input_clip_length=4,
                input_crop_size=182,
                model_num_class=400 if pretrained else self.num_classes,
                dropout_rate=0.5,
                width_factor=1.0,
                depth_factor=1.0
            )
        elif size == "m":
            model = x3d.create_x3d(
                input_clip_length=16,
                input_crop_size=224,
                model_num_class=400 if pretrained else self.num_classes,
                dropout_rate=0.5,
                width_factor=2.0,
                depth_factor=2.2
            )
        elif size == "l":
            model = x3d.create_x3d(
                input_clip_length=16,
                input_crop_size=312,
                model_num_class=400 if pretrained else self.num_classes,
                dropout_rate=0.5,
                width_factor=2.0,
                depth_factor=5.0
            )
        else:
            raise ValueError(f"Unknown X3D size: {size}")

        return model

    def _build_slowfast(self, size: str, pretrained: bool) -> nn.Module:
        """Build SlowFast model"""
        if size == "s":
            model = slowfast.create_slowfast(
                model_num_class=400 if pretrained else self.num_classes,
                slowfast_channel_reduction_ratio=8,
                slowfast_conv_channel_fusion_ratio=2,
                slowfast_fusion_conv_kernel_size=(5, 7, 7),
                slowfast_fusion_conv_stride=(1, 1, 1)
            )
        else:
            model = slowfast.create_slowfast(
                model_num_class=400 if pretrained else self.num_classes
            )

        return model

    def _replace_head(self, dropout: float):
        """Replace classification head for custom number of classes"""
        if self.model_type == "x3d":
            # X3D uses projection head
            in_features = self.backbone.blocks[-1].proj.in_features
            self.backbone.blocks[-1].proj = nn.Linear(in_features, self.num_classes)
            if hasattr(self.backbone.blocks[-1], 'dropout'):
                self.backbone.blocks[-1].dropout = nn.Dropout(dropout)

        elif self.model_type == "slowfast":
            # SlowFast uses head.projection
            if hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'projection'):
                in_features = self.backbone.head.projection.in_features
                self.backbone.head.projection = nn.Linear(in_features, self.num_classes)
                if hasattr(self.backbone.head, 'dropout'):
                    self.backbone.head.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [B, C, T, H, W]
               B: batch size
               C: channels (3)
               T: temporal frames
               H, W: spatial dimensions

        Returns:
            Logits [B, num_classes]
        """
        return self.backbone(x)

    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict action class

        Args:
            x: Input tensor [B, C, T, H, W]
            return_probs: Return probabilities instead of logits

        Returns:
            Tuple of (class_indices, scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)

            if return_probs:
                scores = F.softmax(logits, dim=1)
            else:
                scores = logits

            class_indices = torch.argmax(scores, dim=1)

        return class_indices, scores

    def predict_with_names(
        self,
        x: torch.Tensor
    ) -> List[Dict[str, any]]:
        """
        Predict with class names

        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            List of prediction dicts with class_name, class_id, confidence
        """
        class_indices, scores = self.predict(x, return_probs=True)

        results = []
        for idx, score_vec in zip(class_indices, scores):
            idx = idx.item()
            confidence = score_vec[idx].item()

            results.append({
                "class_id": idx,
                "class_name": self.class_names[idx],
                "confidence": confidence,
                "all_scores": score_vec.cpu().numpy()
            })

        return results

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head

        Args:
            x: Input tensor [B, C, T, H, W]

        Returns:
            Feature tensor
        """
        self.eval()
        with torch.no_grad():
            if self.model_type == "x3d":
                # Extract from before final projection
                features = self.backbone.blocks[:-1](x)
            elif self.model_type == "slowfast":
                # Extract from before head
                for name, module in self.backbone.named_children():
                    if name != 'head':
                        x = module(x)
                features = x
            else:
                features = self.backbone(x)

        return features


class ActionClassifierEnsemble:
    """
    Ensemble of action classifiers for improved robustness
    """

    def __init__(self, models: List[TennisActionClassifier], weights: Optional[List[float]] = None):
        """
        Initialize ensemble

        Args:
            models: List of TennisActionClassifier models
            weights: Optional weights for each model (default: uniform)
        """
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)
        self.num_classes = models[0].num_classes
        self.class_names = models[0].class_names

        logger.info(f"Ensemble initialized with {len(models)} models")

    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction

        Args:
            x: Input tensor [B, C, T, H, W]
            return_probs: Return probabilities

        Returns:
            Tuple of (class_indices, scores)
        """
        all_scores = []

        for model, weight in zip(self.models, self.weights):
            _, scores = model.predict(x, return_probs=True)
            all_scores.append(scores * weight)

        # Average predictions
        ensemble_scores = torch.stack(all_scores).mean(dim=0)
        class_indices = torch.argmax(ensemble_scores, dim=1)

        return class_indices, ensemble_scores

    def predict_with_names(self, x: torch.Tensor) -> List[Dict[str, any]]:
        """Predict with class names"""
        class_indices, scores = self.predict(x, return_probs=True)

        results = []
        for idx, score_vec in zip(class_indices, scores):
            idx = idx.item()
            confidence = score_vec[idx].item()

            results.append({
                "class_id": idx,
                "class_name": self.class_names[idx],
                "confidence": confidence,
                "all_scores": score_vec.cpu().numpy()
            })

        return results


def load_action_classifier(
    checkpoint_path: str,
    model_type: str = "x3d",
    model_size: str = "m",
    num_classes: int = 6,
    device: str = "cuda"
) -> TennisActionClassifier:
    """
    Load pretrained action classifier

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Model architecture
        model_size: Model size
        num_classes: Number of classes
        device: Device to load model on

    Returns:
        Loaded model
    """
    model = TennisActionClassifier(
        model_type=model_type,
        model_size=model_size,
        num_classes=num_classes,
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Action classifier loaded from {checkpoint_path}")
    return model
