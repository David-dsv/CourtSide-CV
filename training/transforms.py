"""
Data Augmentation Transforms
Video augmentation for training
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import random


class VideoCompose:
    """Compose multiple transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for transform in self.transforms:
            clip = transform(clip)
        return clip


class VideoToTensor:
    """Convert numpy array to torch tensor"""

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            [C, T, H, W] tensor
        """
        # Normalize to [0, 1]
        clip = clip.astype(np.float32) / 255.0

        # Convert to tensor and permute
        tensor = torch.from_numpy(clip).permute(3, 0, 1, 2).float()

        return tensor


class VideoNormalize:
    """Normalize video with mean and std"""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
        std: Tuple[float, float, float] = (0.225, 0.225, 0.225)
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: [C, T, H, W] tensor

        Returns:
            Normalized tensor
        """
        return (clip - self.mean) / self.std


class VideoRandomHorizontalFlip:
    """Randomly flip video horizontally"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Flipped clip
        """
        if random.random() < self.p:
            clip = np.flip(clip, axis=2).copy()

        return clip


class VideoRandomCrop:
    """Random spatial crop"""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Cropped clip
        """
        h, w = clip.shape[1:3]
        target_h, target_w = self.size

        if h == target_h and w == target_w:
            return clip

        # Random crop position
        top = random.randint(0, h - target_h)
        left = random.randint(0, w - target_w)

        # Crop all frames
        clip = clip[:, top:top + target_h, left:left + target_w, :]

        return clip


class VideoCenterCrop:
    """Center crop"""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Cropped clip
        """
        h, w = clip.shape[1:3]
        target_h, target_w = self.size

        if h == target_h and w == target_w:
            return clip

        # Center crop position
        top = (h - target_h) // 2
        left = (w - target_w) // 2

        # Crop all frames
        clip = clip[:, top:top + target_h, left:left + target_w, :]

        return clip


class VideoResize:
    """Resize video frames"""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Resized clip
        """
        resized_frames = []

        for frame in clip:
            resized = cv2.resize(frame, self.size)
            resized_frames.append(resized)

        return np.array(resized_frames)


class VideoRandomResizedCrop:
    """Random resized crop (like ImageNet training)"""

    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33)
    ):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Cropped and resized clip
        """
        h, w = clip.shape[1:3]
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))

            if new_w <= w and new_h <= h:
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)

                # Crop
                clip = clip[:, top:top + new_h, left:left + new_w, :]

                # Resize
                return VideoResize(self.size)(clip)

        # Fallback to center crop
        return VideoResize(self.size)(VideoCenterCrop(min(h, w))(clip))


class VideoColorJitter:
    """Random color jittering"""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array in BGR

        Returns:
            Jittered clip
        """
        # Random adjustments (same for all frames)
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_shift = random.uniform(-self.hue, self.hue)

        jittered_frames = []

        for frame in clip:
            # Convert to HSV for saturation and hue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

            # Adjust saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)

            # Adjust hue
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift * 180, 0, 180)

            # Convert back to BGR
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Adjust brightness and contrast
            frame = frame.astype(np.float32)
            frame = frame * contrast_factor + (brightness_factor - 1) * 128
            frame = np.clip(frame, 0, 255).astype(np.uint8)

            jittered_frames.append(frame)

        return np.array(jittered_frames)


class VideoRandomRotation:
    """Random rotation"""

    def __init__(self, degrees: float = 15):
        self.degrees = degrees

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Rotated clip
        """
        angle = random.uniform(-self.degrees, self.degrees)
        h, w = clip.shape[1:3]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_frames = []

        for frame in clip:
            rotated = cv2.warpAffine(frame, rotation_matrix, (w, h))
            rotated_frames.append(rotated)

        return np.array(rotated_frames)


class VideoTemporalCrop:
    """Temporal cropping (sample subset of frames)"""

    def __init__(self, clip_length: int):
        self.clip_length = clip_length

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array

        Returns:
            Temporally cropped clip
        """
        total_frames = len(clip)

        if total_frames <= self.clip_length:
            # Pad if necessary
            padding = self.clip_length - total_frames
            clip = np.concatenate([clip] + [clip[-1:]] * padding, axis=0)
            return clip

        # Random start
        start = random.randint(0, total_frames - self.clip_length)
        return clip[start:start + self.clip_length]


class BGRToRGB:
    """Convert BGR to RGB"""

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        """
        Args:
            clip: [T, H, W, C] numpy array in BGR

        Returns:
            Clip in RGB
        """
        rgb_frames = []

        for frame in clip:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb)

        return np.array(rgb_frames)


def get_train_transforms(
    spatial_size: int = 224,
    clip_length: int = 32,
    augment: bool = True
):
    """
    Get training transforms

    Args:
        spatial_size: Target spatial size
        clip_length: Target clip length
        augment: Apply augmentations

    Returns:
        Transform pipeline
    """
    transforms = [VideoTemporalCrop(clip_length)]

    if augment:
        transforms.extend([
            VideoRandomResizedCrop((spatial_size, spatial_size)),
            VideoRandomHorizontalFlip(p=0.5),
            VideoColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            VideoRandomRotation(degrees=15)
        ])
    else:
        transforms.append(VideoResize((spatial_size, spatial_size)))

    transforms.extend([
        BGRToRGB(),
        VideoToTensor(),
        VideoNormalize()
    ])

    return VideoCompose(transforms)


def get_val_transforms(
    spatial_size: int = 224,
    clip_length: int = 32
):
    """
    Get validation transforms

    Args:
        spatial_size: Target spatial size
        clip_length: Target clip length

    Returns:
        Transform pipeline
    """
    transforms = [
        VideoTemporalCrop(clip_length),
        VideoResize((spatial_size, spatial_size)),
        BGRToRGB(),
        VideoToTensor(),
        VideoNormalize()
    ]

    return VideoCompose(transforms)
