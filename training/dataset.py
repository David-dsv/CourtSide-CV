"""
Dataset Classes for Training
Video dataset for tennis action recognition
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import json
from loguru import logger


class TennisActionDataset(Dataset):
    """
    Dataset for tennis action recognition
    Expects data structure:
    data_root/
        forehand/
            video1.mp4
            video2.mp4
        backhand/
            video1.mp4
        ...
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        clip_length: int = 32,
        clip_stride: int = 1,
        spatial_size: int = 224,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset

        Args:
            data_root: Root directory of dataset
            split: train, val, or test
            clip_length: Number of frames per clip
            clip_stride: Temporal stride for sampling
            spatial_size: Spatial size for frames
            transform: Optional transform function
            class_names: List of class names
        """
        self.data_root = Path(data_root)
        self.split = split
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.spatial_size = spatial_size
        self.transform = transform

        # Default class names
        if class_names is None:
            self.class_names = [
                "forehand",
                "backhand",
                "slice",
                "serve",
                "smash",
                "volley"
            ]
        else:
            self.class_names = class_names

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Load video list
        self.samples = self._load_samples()

        logger.info(
            f"Loaded {len(self.samples)} samples for {split} split "
            f"with {len(self.class_names)} classes"
        )

    def _load_samples(self) -> List[Dict]:
        """
        Load video samples

        Returns:
            List of sample dictionaries
        """
        samples = []
        split_dir = self.data_root / self.split

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return samples

        # Iterate through class directories
        for class_name in self.class_names:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Find all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []

            for ext in video_extensions:
                video_files.extend(list(class_dir.glob(f"*{ext}")))

            # Add to samples
            for video_path in video_files:
                samples.append({
                    "path": str(video_path),
                    "class": class_name,
                    "class_idx": self.class_to_idx[class_name]
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item

        Args:
            idx: Sample index

        Returns:
            Tuple of (clip_tensor, label)
            clip_tensor: [C, T, H, W]
            label: Class index
        """
        sample = self.samples[idx]
        video_path = sample["path"]
        label = sample["class_idx"]

        # Load video clip
        frames = self._load_video(video_path)

        if frames is None or len(frames) == 0:
            # Return dummy data on error
            logger.warning(f"Failed to load video: {video_path}")
            frames = np.zeros(
                (self.clip_length, self.spatial_size, self.spatial_size, 3),
                dtype=np.uint8
            )

        # Sample clip
        clip = self._sample_clip(frames)

        # Apply transforms
        if self.transform:
            clip = self.transform(clip)
        else:
            clip = self._default_transform(clip)

        return clip, label

    def _load_video(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Load video frames

        Args:
            video_path: Path to video file

        Returns:
            List of frames or None on error
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to spatial size
            frame = cv2.resize(frame, (self.spatial_size, self.spatial_size))
            frames.append(frame)

        cap.release()

        return frames

    def _sample_clip(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Sample clip from frames

        Args:
            frames: List of frames

        Returns:
            Clip array [T, H, W, C]
        """
        total_frames = len(frames)
        required_frames = (self.clip_length - 1) * self.clip_stride + 1

        # Pad if necessary
        if total_frames < required_frames:
            padding = [frames[-1]] * (required_frames - total_frames)
            frames = frames + padding
            total_frames = len(frames)

        # Random start for train, center for val/test
        if self.split == "train":
            start_idx = np.random.randint(0, total_frames - required_frames + 1)
        else:
            start_idx = (total_frames - required_frames) // 2

        # Extract frames
        indices = range(start_idx, start_idx + required_frames, self.clip_stride)
        clip_frames = [frames[i] for i in indices]

        return np.array(clip_frames)

    def _default_transform(self, clip: np.ndarray) -> torch.Tensor:
        """
        Default preprocessing transform

        Args:
            clip: Clip array [T, H, W, C] in BGR

        Returns:
            Tensor [C, T, H, W]
        """
        # Convert BGR to RGB
        clip_rgb = []
        for frame in clip:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_rgb.append(frame_rgb)

        clip_rgb = np.array(clip_rgb)

        # Normalize
        clip_normalized = clip_rgb.astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45])
        std = np.array([0.225, 0.225, 0.225])
        clip_normalized = (clip_normalized - mean) / std

        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        clip_tensor = torch.from_numpy(clip_normalized).permute(3, 0, 1, 2).float()

        return clip_tensor


class TennisActionAnnotationDataset(Dataset):
    """
    Dataset loading from annotation file
    Expects JSON format:
    {
        "videos": [
            {
                "path": "video.mp4",
                "annotations": [
                    {"frame_start": 0, "frame_end": 32, "action": "forehand"},
                    ...
                ]
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        annotation_file: str,
        video_root: str,
        clip_length: int = 32,
        spatial_size: int = 224,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize dataset

        Args:
            annotation_file: Path to annotation JSON file
            video_root: Root directory for videos
            clip_length: Frames per clip
            spatial_size: Spatial size
            transform: Optional transform
            class_names: List of class names
        """
        self.video_root = Path(video_root)
        self.clip_length = clip_length
        self.spatial_size = spatial_size
        self.transform = transform

        # Load class names
        if class_names is None:
            self.class_names = [
                "forehand",
                "backhand",
                "slice",
                "serve",
                "smash",
                "volley"
            ]
        else:
            self.class_names = class_names

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Load annotations
        self.samples = self._load_annotations(annotation_file)

        logger.info(f"Loaded {len(self.samples)} annotated clips")

    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Load annotations from file"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        samples = []

        for video_info in data.get("videos", []):
            video_path = self.video_root / video_info["path"]

            for ann in video_info.get("annotations", []):
                action = ann["action"]

                if action not in self.class_to_idx:
                    logger.warning(f"Unknown action: {action}")
                    continue

                samples.append({
                    "video_path": str(video_path),
                    "frame_start": ann["frame_start"],
                    "frame_end": ann["frame_end"],
                    "action": action,
                    "class_idx": self.class_to_idx[action]
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item"""
        sample = self.samples[idx]

        # Load video segment
        frames = self._load_video_segment(
            sample["video_path"],
            sample["frame_start"],
            sample["frame_end"]
        )

        if frames is None or len(frames) == 0:
            logger.warning(f"Failed to load video segment: {sample['video_path']}")
            frames = np.zeros(
                (self.clip_length, self.spatial_size, self.spatial_size, 3),
                dtype=np.uint8
            )

        # Sample to fixed length
        clip = self._sample_to_length(frames)

        # Transform
        if self.transform:
            clip = self.transform(clip)
        else:
            clip = self._default_transform(clip)

        return clip, sample["class_idx"]

    def _load_video_segment(
        self,
        video_path: str,
        frame_start: int,
        frame_end: int
    ) -> Optional[List[np.ndarray]]:
        """Load video segment"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        frames = []

        for _ in range(frame_end - frame_start):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.spatial_size, self.spatial_size))
            frames.append(frame)

        cap.release()

        return frames

    def _sample_to_length(self, frames: List[np.ndarray]) -> np.ndarray:
        """Sample frames to target length"""
        total_frames = len(frames)

        if total_frames == self.clip_length:
            return np.array(frames)

        elif total_frames < self.clip_length:
            # Pad with last frame
            padding = [frames[-1]] * (self.clip_length - total_frames)
            return np.array(frames + padding)

        else:
            # Uniformly sample
            indices = np.linspace(0, total_frames - 1, self.clip_length, dtype=int)
            return np.array([frames[i] for i in indices])

    def _default_transform(self, clip: np.ndarray) -> torch.Tensor:
        """Default transform"""
        clip_rgb = []
        for frame in clip:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_rgb.append(frame_rgb)

        clip_rgb = np.array(clip_rgb)

        clip_normalized = clip_rgb.astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45])
        std = np.array([0.225, 0.225, 0.225])
        clip_normalized = (clip_normalized - mean) / std

        clip_tensor = torch.from_numpy(clip_normalized).permute(3, 0, 1, 2).float()

        return clip_tensor
