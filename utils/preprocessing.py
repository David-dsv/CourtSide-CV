"""
Preprocessing Utilities
Image and video preprocessing for models
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
from PIL import Image


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    to_rgb: bool = True
) -> np.ndarray:
    """
    Normalize image with mean and std

    Args:
        image: Input image [H, W, 3] in range [0, 255]
        mean: Mean values for each channel
        std: Std values for each channel
        to_rgb: Convert BGR to RGB

    Returns:
        Normalized image
    """
    # Convert to float
    image = image.astype(np.float32) / 255.0

    # Convert BGR to RGB if needed
    if to_rgb and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean) / std

    return image


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    to_bgr: bool = True
) -> np.ndarray:
    """
    Denormalize image

    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Std values used for normalization
        to_bgr: Convert RGB to BGR

    Returns:
        Denormalized image in range [0, 255]
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    # Denormalize
    image = image * std + mean

    # Clip to valid range
    image = np.clip(image, 0, 1)

    # Convert to uint8
    image = (image * 255).astype(np.uint8)

    # Convert RGB to BGR if needed
    if to_bgr and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
    interpolation: int = cv2.INTER_LINEAR
) -> Tuple[np.ndarray, float]:
    """
    Resize image maintaining aspect ratio

    Args:
        image: Input image
        target_size: Target size for shorter side
        interpolation: Interpolation method

    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = image.shape[:2]

    # Calculate scale
    if h < w:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)
    else:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    return resized, scale


def center_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int]
) -> np.ndarray:
    """
    Center crop image

    Args:
        image: Input image [H, W, ...]
        crop_size: (crop_height, crop_width)

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Calculate crop coordinates
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    # Crop
    cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

    return cropped


def random_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int]
) -> np.ndarray:
    """
    Random crop image

    Args:
        image: Input image
        crop_size: (crop_height, crop_width)

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Random coordinates
    max_h = h - crop_h
    max_w = w - crop_w

    start_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
    start_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0

    # Crop
    cropped = image[start_h:start_h + crop_h, start_w:start_w + crop_w]

    return cropped


def pad_to_square(
    image: np.ndarray,
    pad_value: int = 114
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad image to square

    Args:
        image: Input image
        pad_value: Padding value

    Returns:
        Tuple of (padded_image, (pad_h, pad_w))
    """
    h, w = image.shape[:2]

    if h == w:
        return image, (0, 0)

    # Calculate padding
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    # Pad
    padded = cv2.copyMakeBorder(
        image,
        pad_h,
        max_dim - h - pad_h,
        pad_w,
        max_dim - w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value)
    )

    return padded, (pad_h, pad_w)


def extract_clip_frames(
    frames: List[np.ndarray],
    clip_length: int,
    stride: int = 1,
    mode: str = "center"
) -> np.ndarray:
    """
    Extract clip of specific length from frame sequence

    Args:
        frames: List of frames
        clip_length: Number of frames to extract
        stride: Temporal stride
        mode: Sampling mode (center, random, uniform)

    Returns:
        Array of frames [T, H, W, C]
    """
    total_frames = len(frames)

    if total_frames == 0:
        raise ValueError("Empty frame list")

    # Calculate required frames with stride
    required_frames = (clip_length - 1) * stride + 1

    if total_frames < required_frames:
        # Pad by repeating frames
        frames = frames + [frames[-1]] * (required_frames - total_frames)
        total_frames = len(frames)

    # Sample frames
    if mode == "center":
        start_idx = (total_frames - required_frames) // 2
    elif mode == "random":
        start_idx = np.random.randint(0, total_frames - required_frames + 1)
    elif mode == "uniform":
        indices = np.linspace(0, total_frames - 1, clip_length, dtype=int)
        return np.array([frames[i] for i in indices])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Extract frames with stride
    indices = range(start_idx, start_idx + required_frames, stride)
    clip_frames = [frames[i] for i in indices]

    return np.array(clip_frames)


def temporal_crop_clips(
    frames: List[np.ndarray],
    clip_length: int,
    clip_stride: int
) -> List[np.ndarray]:
    """
    Extract multiple overlapping clips from video

    Args:
        frames: List of frames
        clip_length: Frames per clip
        clip_stride: Stride between clips

    Returns:
        List of clips, each [T, H, W, C]
    """
    clips = []
    total_frames = len(frames)

    for start_idx in range(0, total_frames - clip_length + 1, clip_stride):
        end_idx = start_idx + clip_length
        clip = np.array(frames[start_idx:end_idx])
        clips.append(clip)

    return clips


def preprocess_for_x3d(
    frames: np.ndarray,
    input_size: int = 224,
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)
) -> torch.Tensor:
    """
    Preprocess frames for X3D model

    Args:
        frames: Frame array [T, H, W, C] in BGR
        input_size: Target spatial size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Tensor [1, C, T, H, W]
    """
    processed_frames = []

    for frame in frames:
        # Resize
        frame = cv2.resize(frame, (input_size, input_size))

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array(mean)) / np.array(std)

        processed_frames.append(frame)

    # Stack and convert to tensor [T, H, W, C]
    frames_array = np.stack(processed_frames)

    # Convert to [C, T, H, W]
    frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2).float()

    # Add batch dimension [1, C, T, H, W]
    frames_tensor = frames_tensor.unsqueeze(0)

    return frames_tensor


def preprocess_for_slowfast(
    frames: np.ndarray,
    input_size: int = 224,
    alpha: int = 4,
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225)
) -> List[torch.Tensor]:
    """
    Preprocess frames for SlowFast model

    Args:
        frames: Frame array [T, H, W, C] in BGR
        input_size: Target spatial size
        alpha: Slow/fast frame ratio
        mean: Normalization mean
        std: Normalization std

    Returns:
        List of [slow_pathway, fast_pathway] tensors
    """
    processed_frames = []

    for frame in frames:
        # Resize
        frame = cv2.resize(frame, (input_size, input_size))

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array(mean)) / np.array(std)

        processed_frames.append(frame)

    frames_array = np.stack(processed_frames)

    # Slow pathway: all frames
    slow_frames = torch.from_numpy(frames_array).permute(3, 0, 1, 2).float().unsqueeze(0)

    # Fast pathway: subsample by alpha
    fast_indices = np.arange(0, len(frames_array), alpha)
    fast_frames_array = frames_array[fast_indices]
    fast_frames = torch.from_numpy(fast_frames_array).permute(3, 0, 1, 2).float().unsqueeze(0)

    return [slow_frames, fast_frames]


def augment_frame(
    frame: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    horizontal_flip: bool = False
) -> np.ndarray:
    """
    Apply augmentations to frame

    Args:
        frame: Input frame [H, W, C]
        brightness: Brightness adjustment [-1, 1]
        contrast: Contrast adjustment [-1, 1]
        saturation: Saturation adjustment [-1, 1]
        hue: Hue adjustment [-1, 1]
        horizontal_flip: Apply horizontal flip

    Returns:
        Augmented frame
    """
    # Convert to PIL for color augmentations
    from PIL import ImageEnhance

    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Brightness
    if brightness != 0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1 + brightness)

    # Contrast
    if contrast != 0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1 + contrast)

    # Saturation
    if saturation != 0:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1 + saturation)

    # Convert back to numpy
    frame = np.array(pil_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Horizontal flip
    if horizontal_flip:
        frame = cv2.flip(frame, 1)

    return frame


def create_sliding_windows(
    total_frames: int,
    window_size: int,
    stride: int
) -> List[Tuple[int, int]]:
    """
    Create sliding window indices for temporal processing

    Args:
        total_frames: Total number of frames
        window_size: Window size
        stride: Stride between windows

    Returns:
        List of (start_frame, end_frame) tuples
    """
    windows = []

    for start in range(0, total_frames - window_size + 1, stride):
        end = start + window_size
        windows.append((start, end))

    # Add final window if needed
    if len(windows) == 0 or windows[-1][1] < total_frames:
        windows.append((max(0, total_frames - window_size), total_frames))

    return windows
