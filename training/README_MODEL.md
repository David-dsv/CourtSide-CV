---
language: en
license: mit
tags:
  - yolo
  - yolov11
  - object-detection
  - tennis
  - racket
  - tennis-ball
  - court-detection
  - sports
  - computer-vision
  - pytorch
  - ultralytics
  - courtside
datasets:
  - roboflow/tennis-ball-detection
  - roboflow/tennis-court-detection
  - roboflow/racket-detection
metrics:
  - precision
  - recall
  - mAP
library_name: ultralytics
pipeline_tag: object-detection
model-index:
- name: CourtSide Computer Vision v1
  results:
  - task:
      type: object-detection
    metrics:
    - type: mAP@50
      value: 92.13
    - type: mAP@50-95
      value: 85.49
    - type: precision
      value: 92.9
    - type: recall
      value: 92.0
---

# CourtSide Computer Vision v1 - Complete Tennis Detection

Fine-tuned YOLOv11n model for comprehensive tennis analysis with **10-class detection**: rackets, balls, and court zones. The most complete model in the CourtSide Computer Vision suite.

## Model Details

- **Model Name**: CourtSide Computer Vision v1
- **Model ID**: `Davidsv/CourtSide-Computer-Vision-v1`
- **Model Type**: Object Detection
- **Architecture**: YOLOv11 Nano (n)
- **Framework**: Ultralytics YOLOv11
- **Parameters**: 2.6M
- **Input Size**: 640x640
- **Classes**: 10

## Classes Detected

| ID | Class | Description |
|----|-------|-------------|
| 0 | `racket` | Tennis rackets |
| 1 | `tennis_ball` | Tennis balls |
| 2 | `bottom-dead-zone` | Bottom baseline area |
| 3 | `court` | Full court area |
| 4 | `left-doubles-alley` | Left doubles alley |
| 5 | `left-service-box` | Left service box |
| 6 | `net` | Tennis net |
| 7 | `right-doubles-alley` | Right doubles alley |
| 8 | `right-service-box` | Right service box |
| 9 | `top-dead-zone` | Top baseline area |

## Performance Metrics

| Metric | Value |
|--------|-------|
| **mAP@50** | **92.13%** |
| **mAP@50-95** | **85.49%** |
| **Precision** | 92.9% |
| **Recall** | 92.0% |

## Training Details

### Datasets

This model was trained on **3 combined datasets**:

1. **Tennis Ball Dataset** - Ball detection
2. **Tennis Racket Dataset** - Racket detection
3. **Tennis Court Dataset** - Court zones and net detection

### Training Configuration
```yaml
Model: YOLOv11n (nano)
Epochs: 150
Batch size: 16
Image size: 640x640
Device: Apple Silicon (MPS)
Optimizer: AdamW
Learning rate: 0.001 → 0.01
Patience: 50 (early stopping)
```

### Augmentation
- HSV color jitter (h=0.015, s=0.7, v=0.4)
- Random horizontal flip (p=0.5)
- Translation (±10%)
- Scaling (±50%)
- Mosaic augmentation

### Loss Weights
- Box loss: 7.5
- Class loss: 0.5
- DFL loss: 1.5

## Usage

### Installation
```bash
pip install ultralytics
```

### Python API
```python
from ultralytics import YOLO

# Load CourtSide Computer Vision v1 model
model = YOLO('Davidsv/CourtSide-Computer-Vision-v1')

# Predict on image
results = model.predict('tennis_match.jpg', conf=0.25)

# Display results
results[0].show()

# Get detections by class
for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = model.names[cls]
    print(f"{class_name}: {conf:.2%}")
```

### Video Processing
```python
from ultralytics import YOLO

model = YOLO('Davidsv/CourtSide-Computer-Vision-v1')

# Process video with tracking
results = model.track(
    source='tennis_match.mp4',
    conf=0.25,
    tracker='bytetrack.yaml',
    save=True
)
```

### Command Line
```bash
# Predict on image
yolo detect predict model=Davidsv/CourtSide-Computer-Vision-v1 source=image.jpg conf=0.25

# Predict on video
yolo detect predict model=Davidsv/CourtSide-Computer-Vision-v1 source=video.mp4 conf=0.25 save=True

# Track objects in video
yolo detect track model=Davidsv/CourtSide-Computer-Vision-v1 source=video.mp4 conf=0.25
```

## Recommended Hyperparameters

### Inference Settings
```python
# Balanced (recommended)
conf_threshold = 0.25  # Confidence threshold
iou_threshold = 0.45   # NMS IoU threshold

# High precision (fewer false positives)
conf_threshold = 0.40
iou_threshold = 0.45

# High recall (detect more objects)
conf_threshold = 0.15
iou_threshold = 0.40
```

## Use Cases

- Real-time tennis match analysis
- Player position and movement tracking
- Ball trajectory prediction
- Court zone occupancy analysis
- Automated highlight generation
- Swing detection and technique analysis
- Sports analytics dashboards
- Training video analysis

## CourtSide Computer Vision Suite

| Version | Description | mAP@50 |
|---------|-------------|--------|
| v0.1 | Tennis Ball Detection | 85.6% |
| v0.2 | Tennis Racket Detection | 66.7% |
| **v1** | **Complete Tennis Detection (10 classes)** | **92.1%** |

## Model Card Authors

- **Developed by**: Davidsv (Vuong)
- **Model date**: November 2024
- **Model version**: v1
- **Model type**: Object Detection (YOLOv11)
- **Part of**: CourtSide Computer Vision Suite

## Citations

### This Model
```bibtex
@misc{courtsidecv_v1_2024,
  title={CourtSide Computer Vision v1: Complete Tennis Detection with YOLOv11},
  author={Vuong},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1}}
}
```

### Ultralytics YOLOv11
```bibtex
@software{yolov11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## License

MIT License - Free for commercial and academic use.

## Acknowledgments

- Built with [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Training datasets from [Roboflow Universe](https://universe.roboflow.com)
- Part of the CourtSide Computer Vision project for tennis analysis

## Contact & Support

- Hugging Face: [@Davidsv](https://huggingface.co/Davidsv)

---

**Model Size**: ~5.4 MB
**Supported Formats**: PyTorch (.pt), ONNX, TensorRT, CoreML
**Model Hub**: [Davidsv/CourtSide-Computer-Vision-v1](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1)
