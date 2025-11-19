"""
File I/O Utilities
Handle JSON, CSV, and configuration files
"""

import json
import csv
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger


def load_yaml(file_path: str) -> Dict:
    """
    Load YAML configuration file

    Args:
        file_path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {file_path}")
    return config


def save_yaml(data: Dict, file_path: str):
    """
    Save dictionary to YAML file

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {file_path}")


def load_json(file_path: str) -> Dict:
    """
    Load JSON file

    Args:
        file_path: Path to JSON file

    Returns:
        Data dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded JSON from {file_path}")
    return data


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    Save data to JSON file

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Saved JSON to {file_path}")


def save_results_json(
    results: List[Dict],
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save action detection results to JSON

    Args:
        results: List of result dictionaries
        output_path: Output JSON path
        metadata: Optional metadata to include
    """
    output_data = {
        "metadata": metadata or {},
        "results": results,
        "total_detections": len(results)
    }

    save_json(output_data, output_path)


def save_results_csv(
    results: List[Dict],
    output_path: str,
    columns: Optional[List[str]] = None
):
    """
    Save results to CSV file

    Args:
        results: List of result dictionaries
        output_path: Output CSV path
        columns: Optional column order
    """
    if len(results) == 0:
        logger.warning("No results to save")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns if specified
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]

    # Save to CSV
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(results)} results to {output_path}")


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame
    """
    df = pd.read_csv(file_path)
    logger.info(f"Loaded CSV from {file_path} ({len(df)} rows)")
    return df


def create_results_dict(
    frame_id: int,
    timestamp: float,
    action: str,
    confidence: float,
    bbox: Optional[List[float]] = None,
    track_id: Optional[int] = None,
    **kwargs
) -> Dict:
    """
    Create standardized result dictionary

    Args:
        frame_id: Frame number
        timestamp: Timestamp in seconds
        action: Action class name
        confidence: Confidence score
        bbox: Bounding box [x1, y1, x2, y2]
        track_id: Track ID
        **kwargs: Additional fields

    Returns:
        Result dictionary
    """
    result = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "action": action,
        "confidence": confidence
    }

    if bbox is not None:
        result["bbox"] = bbox

    if track_id is not None:
        result["track_id"] = track_id

    # Add any additional fields
    result.update(kwargs)

    return result


def merge_config(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value

    return merged


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists

    Args:
        directory: Directory path

    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_string() -> str:
    """
    Get current timestamp as string

    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_files(
    directory: str,
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files matching pattern

    Args:
        directory: Directory to search
        pattern: Glob pattern
        recursive: Search recursively

    Returns:
        List of file paths
    """
    path = Path(directory)

    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))

    # Filter only files
    files = [f for f in files if f.is_file()]

    return sorted(files)


def get_video_files(directory: str) -> List[Path]:
    """
    Get all video files in directory

    Args:
        directory: Directory path

    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(list_files(directory, f"*{ext}"))

    return sorted(video_files)


class ResultsManager:
    """
    Manage detection results with buffering and batch saving
    """

    def __init__(
        self,
        output_dir: str,
        buffer_size: int = 100,
        save_json: bool = True,
        save_csv: bool = True
    ):
        """
        Initialize results manager

        Args:
            output_dir: Output directory
            buffer_size: Number of results to buffer before saving
            save_json: Save JSON output
            save_csv: Save CSV output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.save_json = save_json
        self.save_csv = save_csv

        self.results_buffer = []
        self.all_results = []

    def add_result(self, result: Dict):
        """Add single result"""
        self.results_buffer.append(result)
        self.all_results.append(result)

        # Flush if buffer is full
        if len(self.results_buffer) >= self.buffer_size:
            self.flush()

    def add_results(self, results: List[Dict]):
        """Add multiple results"""
        for result in results:
            self.add_result(result)

    def flush(self):
        """Flush buffer to disk"""
        if len(self.results_buffer) == 0:
            return

        logger.debug(f"Flushing {len(self.results_buffer)} results to buffer")
        self.results_buffer = []

    def save(self, filename: str = "results"):
        """
        Save all results to file

        Args:
            filename: Output filename (without extension)
        """
        if len(self.all_results) == 0:
            logger.warning("No results to save")
            return

        # Save JSON
        if self.save_json:
            json_path = self.output_dir / f"{filename}.json"
            save_results_json(self.all_results, str(json_path))

        # Save CSV
        if self.save_csv:
            csv_path = self.output_dir / f"{filename}.csv"
            save_results_csv(self.all_results, str(csv_path))

        logger.info(f"Saved {len(self.all_results)} results")

    def get_results(self) -> List[Dict]:
        """Get all results"""
        return self.all_results

    def clear(self):
        """Clear all results"""
        self.results_buffer = []
        self.all_results = []


def parse_class_names(file_path: str) -> List[str]:
    """
    Parse class names from text file

    Args:
        file_path: Path to class names file (one per line)

    Returns:
        List of class names
    """
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(classes)} class names from {file_path}")
    return classes


def save_class_names(classes: List[str], file_path: str):
    """
    Save class names to text file

    Args:
        classes: List of class names
        file_path: Output file path
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    logger.info(f"Saved {len(classes)} class names to {file_path}")
