"""
Video Processing Utilities
Read, write, and manipulate video files
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Generator
from loguru import logger
import subprocess


class VideoReader:
    """
    Efficient video reader with frame caching and seeking
    """

    def __init__(self, video_path: str, resize: Optional[Tuple[int, int]] = None):
        """
        Initialize video reader

        Args:
            video_path: Path to video file
            resize: Optional (width, height) to resize frames
        """
        self.video_path = video_path
        self.resize = resize

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        logger.info(
            f"Video loaded: {video_path} | "
            f"{self.width}x{self.height} | "
            f"{self.fps:.2f} fps | "
            f"{self.frame_count} frames | "
            f"{self.duration:.2f}s"
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame

        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()

        if ret and self.resize:
            frame = cv2.resize(frame, self.resize)

        return ret, frame

    def read_frames(self, num_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Read multiple frames

        Args:
            num_frames: Number of frames to read (None = all remaining)

        Returns:
            List of frames
        """
        frames = []
        count = 0

        while True:
            ret, frame = self.read_frame()

            if not ret:
                break

            frames.append(frame)
            count += 1

            if num_frames and count >= num_frames:
                break

        return frames

    def iter_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over all frames

        Yields:
            Frame arrays
        """
        while True:
            ret, frame = self.read_frame()

            if not ret:
                break

            yield frame

    def seek(self, frame_number: int) -> bool:
        """
        Seek to specific frame

        Args:
            frame_number: Target frame number

        Returns:
            Success status
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_current_frame_number(self) -> int:
        """Get current frame number"""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def get_timestamp(self) -> float:
        """Get current timestamp in seconds"""
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            logger.info(f"Video reader released: {self.video_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __len__(self) -> int:
        return self.frame_count


class VideoWriter:
    """
    Video writer with codec selection and quality control
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = "mp4v",
        quality: int = 90
    ):
        """
        Initialize video writer

        Args:
            output_path: Output file path
            fps: Frames per second
            frame_size: (width, height) of output video
            codec: Video codec (mp4v, avc1, x264, etc.)
            quality: Video quality (0-100)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.quality = quality
        self.writer = None
        self.frame_count = 0

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Video writer initialized: {output_path} | {fps} fps | codec: {codec}")

    def write_frame(self, frame: np.ndarray):
        """
        Write single frame

        Args:
            frame: Frame array [H, W, 3]
        """
        if self.writer is None:
            # Initialize writer on first frame
            if self.frame_size is None:
                self.frame_size = (frame.shape[1], frame.shape[0])

            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.frame_size
            )

            if not self.writer.isOpened():
                raise ValueError(f"Cannot open video writer: {self.output_path}")

        # Resize if necessary
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)
        self.frame_count += 1

    def write_frames(self, frames: List[np.ndarray]):
        """
        Write multiple frames

        Args:
            frames: List of frame arrays
        """
        for frame in frames:
            self.write_frame(frame)

    def release(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
            logger.info(
                f"Video written: {self.output_path} | "
                f"{self.frame_count} frames | "
                f"{self.frame_count / self.fps:.2f}s"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoClipper:
    """
    Extract clips from video based on time ranges or frame ranges
    """

    def __init__(self, video_path: str):
        """
        Initialize video clipper

        Args:
            video_path: Path to source video
        """
        self.video_path = video_path
        self.reader = VideoReader(video_path)

    def extract_clip(
        self,
        output_path: str,
        start_frame: int,
        end_frame: int,
        fps: Optional[float] = None
    ):
        """
        Extract clip by frame range

        Args:
            output_path: Output video path
            start_frame: Start frame number
            end_frame: End frame number
            fps: Output FPS (None = same as source)
        """
        if fps is None:
            fps = self.reader.fps

        # Seek to start frame
        self.reader.seek(start_frame)

        # Create writer
        with VideoWriter(output_path, fps=fps) as writer:
            for i in range(start_frame, end_frame):
                ret, frame = self.reader.read_frame()

                if not ret:
                    break

                writer.write_frame(frame)

        logger.info(
            f"Clip extracted: {output_path} | "
            f"frames {start_frame}-{end_frame} ({end_frame - start_frame} frames)"
        )

    def extract_clip_by_time(
        self,
        output_path: str,
        start_time: float,
        end_time: float,
        fps: Optional[float] = None
    ):
        """
        Extract clip by time range

        Args:
            output_path: Output video path
            start_time: Start time in seconds
            end_time: End time in seconds
            fps: Output FPS
        """
        start_frame = int(start_time * self.reader.fps)
        end_frame = int(end_time * self.reader.fps)

        self.extract_clip(output_path, start_frame, end_frame, fps)

    def extract_clips_batch(
        self,
        output_dir: str,
        clip_ranges: List[Tuple[int, int]],
        prefix: str = "clip"
    ):
        """
        Extract multiple clips

        Args:
            output_dir: Output directory
            clip_ranges: List of (start_frame, end_frame) tuples
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (start, end) in enumerate(clip_ranges):
            output_path = output_dir / f"{prefix}_{i:04d}.mp4"
            self.extract_clip(str(output_path), start, end)

        logger.info(f"Extracted {len(clip_ranges)} clips to {output_dir}")

    def release(self):
        """Release resources"""
        self.reader.release()


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)

    info = {
        "path": video_path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "duration": 0
    }

    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0

    cap.release()

    return info


def resize_video(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: Optional[float] = None
):
    """
    Resize video to new dimensions

    Args:
        input_path: Input video path
        output_path: Output video path
        width: Target width
        height: Target height
        fps: Target FPS (None = same as source)
    """
    with VideoReader(input_path) as reader:
        if fps is None:
            fps = reader.fps

        with VideoWriter(output_path, fps=fps, frame_size=(width, height)) as writer:
            for frame in reader.iter_frames():
                writer.write_frame(frame)

    logger.info(f"Video resized: {output_path} | {width}x{height}")


def concatenate_videos(video_paths: List[str], output_path: str):
    """
    Concatenate multiple videos

    Args:
        video_paths: List of input video paths
        output_path: Output video path
    """
    if len(video_paths) == 0:
        raise ValueError("No videos to concatenate")

    # Get properties from first video
    first_reader = VideoReader(video_paths[0])
    fps = first_reader.fps
    frame_size = (first_reader.width, first_reader.height)
    first_reader.release()

    with VideoWriter(output_path, fps=fps, frame_size=frame_size) as writer:
        for video_path in video_paths:
            with VideoReader(video_path, resize=frame_size) as reader:
                for frame in reader.iter_frames():
                    writer.write_frame(frame)

    logger.info(f"Concatenated {len(video_paths)} videos to {output_path}")
