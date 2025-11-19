# Tennis Action Analysis 🎾

Complete deep learning pipeline for tennis action recognition from videos. This system detects players, tracks their movements, and classifies tennis strokes in real-time.

## Features

- **Object Detection**: YOLOv8/v11 for detecting players, ball, and racket
- **Multi-Object Tracking**: OCSORT tracker for robust player tracking
- **Action Recognition**: X3D/SlowFast models for recognizing 6 tennis actions:
  - Forehand
  - Backhand
  - Slice
  - Serve
  - Smash
  - Volley
- **Pose Estimation**: Optional skeleton tracking for enhanced analysis
- **Automated Clip Extraction**: Extract action clips automatically
- **Comprehensive Visualization**: Annotated videos with all detections

## Project Structure

```
tennis_analysis/
├── README.md
├── requirements.txt
├── config.yaml
│
├── models/                    # Model implementations
│   ├── yolo_detector.py      # YOLO detection
│   ├── tracker.py            # OCSORT tracking
│   ├── action_classifier.py  # X3D/SlowFast models
│   └── pose_estimator.py     # Pose estimation
│
├── utils/                     # Utility functions
│   ├── video_utils.py        # Video I/O
│   ├── drawing.py            # Visualization
│   ├── preprocessing.py      # Data preprocessing
│   └── file_utils.py         # File operations
│
├── training/                  # Training scripts
│   ├── train_action_classifier.py
│   ├── dataset.py
│   └── transforms.py
│
├── inference/                 # Inference pipeline
│   ├── detect.py             # Run detection
│   ├── track.py              # Run tracking
│   ├── extract_clips.py      # Extract clips
│   ├── predict_actions.py    # Predict actions
│   └── annotate_video.py     # Annotate videos
│
└── main.py                    # Main pipeline
```

## Installation

### 1. Clone Repository

```bash
cd tennis_analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pretrained Weights (Optional)

```bash
# Create weights directory
mkdir -p weights

# Download YOLOv8 pretrained weights (automatic on first run)
# Or train your own (see Training section)

# Download action classifier weights (train your own or use pretrained)
# Place in weights/action_classifier.pth
```

## Quick Start

### Run Complete Pipeline

```bash
python main.py \
    --config config.yaml \
    --video path/to/tennis_video.mp4 \
    --output-dir results/
```

This will:
1. Detect players, ball, and racket
2. Track players across frames
3. Recognize tennis actions
4. Generate annotated video and results (JSON/CSV)

### Configuration

Edit `config.yaml` to customize:

```yaml
paths:
  input_video: "data/input/tennis_match.mp4"
  output_dir: "data/output"
  yolo_weights: "weights/yolo_tennis.pt"
  action_classifier_weights: "weights/action_classifier.pth"

yolo:
  model_version: "yolov8"
  model_size: "m"
  conf_threshold: 0.5

tracking:
  tracker_type: "ocsort"
  max_age: 30
  min_hits: 3

action_recognition:
  model_type: "x3d"
  model_size: "m"
  num_classes: 6
  clip_duration: 32
  conf_threshold: 0.6
```

## Training

### 1. Prepare Dataset

#### Option A: Directory Structure

Organize videos by action class:

```
data/
├── train/
│   ├── forehand/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   ├── backhand/
│   │   ├── video1.mp4
│   ├── serve/
│   │   ├── video1.mp4
│   └── ...
├── val/
│   ├── forehand/
│   ├── backhand/
│   └── ...
```

#### Option B: Annotation File

Create JSON annotation file:

```json
{
  "videos": [
    {
      "path": "match1.mp4",
      "annotations": [
        {"frame_start": 0, "frame_end": 32, "action": "forehand"},
        {"frame_start": 50, "frame_end": 82, "action": "backhand"}
      ]
    }
  ]
}
```

### 2. Train Action Classifier

```bash
python training/train_action_classifier.py \
    --data-root data/ \
    --output-dir runs/train \
    --model-type x3d \
    --model-size m \
    --num-classes 6 \
    --batch-size 8 \
    --num-epochs 50 \
    --lr 0.001 \
    --num-workers 4 \
    --device cuda
```

**Training Arguments:**

- `--data-root`: Dataset directory
- `--model-type`: x3d or slowfast
- `--model-size`: s, m, l (small, medium, large)
- `--batch-size`: Batch size (reduce if OOM)
- `--num-epochs`: Training epochs
- `--lr`: Learning rate
- `--resume`: Resume from checkpoint

**Monitor Training:**

```bash
# Training logs
tail -f runs/train/training.log

# Check training history
cat runs/train/training_history.json
```

### 3. Train Custom YOLO Detector

#### Prepare YOLO Dataset

Create dataset in YOLO format:

```
yolo_data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

**data.yaml:**

```yaml
path: /path/to/yolo_data
train: images/train
val: images/val

nc: 3
names: ['player', 'ball', 'racket']
```

#### Train YOLO

```python
from models.yolo_detector import TennisYOLODetector

detector = TennisYOLODetector(
    model_version="yolov8",
    model_size="m"
)

detector.train(
    data_yaml="yolo_data/data.yaml",
    epochs=100,
    batch_size=16,
    imgsz=640,
    project="runs/yolo",
    name="tennis_detector"
)
```

## Inference

### Run Individual Components

#### 1. Detection Only

```python
from models.yolo_detector import TennisYOLODetector
from inference.detect import detect_video

detector = TennisYOLODetector(
    model_path="weights/yolo_tennis.pt",
    device="cuda"
)

detections = detect_video(
    "video.mp4",
    detector,
    output_video="detections.mp4",
    output_json="detections.json"
)
```

#### 2. Tracking

```python
from models.tracker import OCSort
from inference.track import track_video

tracker = OCSort(max_age=30, min_hits=3)

tracking_results = track_video(
    detections,
    tracker,
    video_path="video.mp4",
    output_video="tracking.mp4"
)
```

#### 3. Action Recognition

```python
from models.action_classifier import load_action_classifier
from inference.predict_actions import predict_actions_from_video

model = load_action_classifier(
    "weights/action_classifier.pth",
    device="cuda"
)

actions = predict_actions_from_video(
    "video.mp4",
    model,
    clip_length=32,
    conf_threshold=0.6
)
```

#### 4. Extract Clips

```python
from inference.extract_clips import extract_action_clips

clips = extract_action_clips(
    "video.mp4",
    actions,
    output_dir="clips/",
    margin_frames=16
)
```

#### 5. Annotate Video

```python
from inference.annotate_video import annotate_comprehensive

annotate_comprehensive(
    "video.mp4",
    "output_annotated.mp4",
    tracking_results=tracking_results,
    action_detections=actions,
    show_tracks=True,
    show_actions=True
)
```

## Output Format

### JSON Results

**detections.json:**
```json
[
  {
    "frame_id": 0,
    "timestamp": 0.0,
    "detections": {
      "boxes": [[100, 200, 300, 400]],
      "scores": [0.95],
      "class_ids": [0],
      "class_names": ["player"]
    }
  }
]
```

**tracking.json:**
```json
[
  {
    "frame_id": 0,
    "timestamp": 0.0,
    "tracks": {
      "boxes": [[100, 200, 300, 400]],
      "track_ids": [1],
      "scores": [0.95],
      "class_ids": [0]
    }
  }
]
```

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

### CSV Results

**actions.csv:**
```csv
frame_id,timestamp,action,confidence,track_id
150,5.0,forehand,0.92,1
180,6.0,backhand,0.88,1
210,7.0,serve,0.95,2
```

## Advanced Usage

### Custom Action Classes

Modify `config.yaml`:

```yaml
action_recognition:
  num_classes: 8
  classes:
    - "forehand"
    - "backhand"
    - "slice"
    - "serve"
    - "smash"
    - "volley"
    - "lob"
    - "drop_shot"
```

### Batch Processing

```python
from pathlib import Path
from main import TennisAnalysisPipeline

pipeline = TennisAnalysisPipeline("config.yaml")

video_dir = Path("videos/")
for video_file in video_dir.glob("*.mp4"):
    print(f"Processing {video_file}")
    pipeline.run(video_path=str(video_file))
```

### Real-time Processing

```python
import cv2
from models.yolo_detector import TennisYOLODetector

detector = TennisYOLODetector(device="cuda")
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    # Draw and display...

cap.release()
```

## Performance Optimization

### GPU Acceleration

```yaml
yolo:
  device: "cuda"

action_recognition:
  batch_size: 16  # Increase for faster processing
```

### Model Size Trade-offs

| Model Size | Speed | Accuracy | VRAM |
|------------|-------|----------|------|
| YOLOv8n + X3D-S | Fast | Good | 2GB |
| YOLOv8m + X3D-M | Medium | Better | 6GB |
| YOLOv8l + X3D-L | Slow | Best | 12GB |

### Reduce Memory Usage

```python
# Process video in chunks
from utils.preprocessing import create_sliding_windows

windows = create_sliding_windows(
    total_frames=1000,
    window_size=100,
    stride=100
)

for start, end in windows:
    # Process chunk...
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch-size 4

# Use smaller model
--model-size s

# Use CPU
--device cpu
```

**2. Video Read Error**
```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Or use different codec
config.yaml: video.codec: "mp4v"
```

**3. Low Detection Accuracy**
```bash
# Lower confidence threshold
config.yaml: yolo.conf_threshold: 0.3

# Train on custom data
python training/train_action_classifier.py ...
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tennis_action_analysis,
  title={Tennis Action Analysis Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tennis-action-analysis}
}
```

## References

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- OCSORT: [Observation-Centric SORT](https://github.com/noahcao/OC_SORT)
- X3D: [PyTorchVideo](https://pytorchvideo.org/)
- SlowFast: [Facebook Research](https://github.com/facebookresearch/SlowFast)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/yourusername/tennis-action-analysis/issues)
- Email: your.email@example.com

## Acknowledgments

Built with:
- PyTorch
- Ultralytics YOLO
- PyTorchVideo
- OpenCV

---

**Happy Tennis Analysis! 🎾🎯**
