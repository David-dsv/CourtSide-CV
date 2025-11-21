# Tennis Analysis

Deep learning pipeline for tennis video analysis with object detection, multi-object tracking, and action recognition.

## Features

- **Object Detection**: YOLOv8/v11/v12 for detecting players, balls, rackets, and court zones
- **Multi-Object Tracking**: OC-SORT tracker for robust player tracking
- **Action Recognition**: Geometric and ML-based detection of tennis actions (forehand, backhand, serve, volley, slice, smash)
- **Video Annotation**: Generate annotated videos with bounding boxes, track IDs, and action labels
- **Custom Model**: Fine-tuned YOLOv11 for tennis-specific detection with 92% mAP

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tennis_analysis.git
cd tennis_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run analysis (interactive YOLO version selection)
python main.py --video path/to/tennis_match.mp4

# Specify YOLO version directly
python main.py --video path/to/video.mp4 --yolo-version yolov11

# Process a specific segment
python main.py --video path/to/video.mp4 --start-time 60 --duration 120

# Extract middle 5 minutes
python main.py --video path/to/video.mp4 --middle 300
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--config` | Path to config file (default: config.yaml) |
| `--video` | Input video path |
| `--output-dir` | Output directory |
| `--yolo-version` | YOLO version: yolov8, yolov11, yolov12 |
| `--start-time` | Start time in seconds |
| `--end-time` | End time in seconds |
| `--duration` | Duration in seconds |
| `--middle` | Extract middle N seconds |
| `--log-level` | DEBUG, INFO, WARNING, ERROR |

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
yolo:
  model_version: "yolov11"
  model_size: "m"
  conf_threshold: 0.2
  device: "cuda"  # or "cpu" for Mac

tracking:
  min_track_length: 30
  max_person_tracks: 2

action_recognition:
  conf_threshold: 0.2
  classes:
    - forehand
    - backhand
    - slice
    - serve
    - smash
    - volley
```

## Project Structure

```
tennis_analysis/
├── main.py                 # Main pipeline
├── config.yaml             # Configuration
├── models/
│   ├── yolo_detector.py    # YOLO detection wrapper
│   ├── tracker.py          # OC-SORT tracker
│   ├── action_classifier.py
│   └── pose_estimator.py
├── inference/
│   ├── detect.py           # Detection
│   ├── track.py            # Tracking
│   ├── predict_actions.py  # Action recognition
│   ├── annotate_video.py   # Video annotation
│   └── extract_clips.py    # Clip extraction
├── training/               # Training scripts
└── utils/                  # Utilities
```

## Custom Model

A fine-tuned YOLOv11 model is available on Hugging Face:

**[Davidsv/CourtSide-Computer-Vision-v1](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1)**

### Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | racket | 5 | left-service-box |
| 1 | tennis_ball | 6 | net |
| 2 | bottom-dead-zone | 7 | right-doubles-alley |
| 3 | court | 8 | right-service-box |
| 4 | left-doubles-alley | 9 | top-dead-zone |

### Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 92.13% |
| mAP@50-95 | 85.49% |
| Precision | 92.9% |
| Recall | 92.0% |

## Output

The pipeline generates:

```
output/
├── detections.json      # Raw detections
├── tracking.json        # Tracking with IDs
├── actions.json         # Detected actions
├── actions.csv          # Actions (CSV format)
└── output_annotated.mp4 # Annotated video
```

### Output Format

**actions.json:**
```json
[
  {
    "frame_id": 150,
    "timestamp": 5.0,
    "action": "forehand",
    "confidence": 0.92,
    "track_id": 1
  }
]
```

## Python API

```python
from main import TennisAnalysisPipeline

pipeline = TennisAnalysisPipeline("config.yaml")
results = pipeline.run(video_path="match.mp4")

print(f"Actions detected: {results['summary']['total_actions']}")
```

## Requirements

- Python 3.9+
- PyTorch
- Ultralytics
- OpenCV
- FFmpeg

## License

MIT License

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [PyTorchVideo](https://pytorchvideo.org/)
