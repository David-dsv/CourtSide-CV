---
language: en
license: mit
tags:
  - yolo
  - yolov11
  - object-detection
  - tennis
  - sports
  - computer-vision
  - pytorch
  - ultralytics
datasets:
  - custom
metrics:
  - precision
  - recall
  - mAP
library_name: ultralytics
pipeline_tag: object-detection
---

# YOLOv11 Tennis Ball Detection 🎾

Fine-tuned YOLOv11n model for detecting tennis balls in images and videos.

## Model Details

- **Model Type**: Object Detection
- **Architecture**: YOLOv11 Nano (n)
- **Framework**: Ultralytics YOLOv11
- **Parameters**: 2.6M
- **Input Size**: 640x640
- **Classes**: 1 (`tennis_ball`)

## Performance Metrics

Evaluated on validation set (62 images):

| Metric | Value |
|--------|-------|
| **mAP@50** | **67.87%** |
| **mAP@50-95** | 24.93% |
| **Precision** | 84.3% |
| **Recall** | 59.5% |
| **Inference Speed** (M4 Pro) | 10.3ms |

## Training Details

### Dataset
- **Training images**: 408
- **Validation images**: 62
- **Test images**: 50
- **Total**: 520 annotated images
- **Annotation format**: YOLO format (bounding boxes)

### Training Configuration
```yaml
Model: YOLOv11n (nano)
Epochs: 100
Batch size: 16
Image size: 640x640
Device: Apple M4 Pro (MPS)
Optimizer: AdamW
Learning rate: 0.001 → 0.01
Training time: ~23 minutes
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
from PIL import Image

# Load model
model = YOLO('path/to/tennis_ball_subset_best.pt')

# Predict on image
results = model.predict('tennis_match.jpg', conf=0.3)

# Display results
results[0].show()

# Get bounding boxes
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    confidence = box.conf[0]
    print(f"Ball detected at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] with {confidence:.2%} confidence")
```

### Video Processing

```python
from ultralytics import YOLO

model = YOLO('path/to/tennis_ball_subset_best.pt')

# Process video
results = model.predict(
    source='tennis_match.mp4',
    conf=0.3,
    save=True,
    save_txt=True
)
```

### Command Line

```bash
# Predict on image
yolo detect predict model=tennis_ball_subset_best.pt source=image.jpg conf=0.3

# Predict on video
yolo detect predict model=tennis_ball_subset_best.pt source=video.mp4 conf=0.3 save=True

# Validate model
yolo detect val model=tennis_ball_subset_best.pt data=dataset.yaml
```

## Recommended Hyperparameters

### Inference Settings

```python
# Balanced (recommended)
conf_threshold = 0.30  # Confidence threshold
iou_threshold = 0.45   # NMS IoU threshold
max_det = 50           # Maximum detections per image

# High precision (fewer false positives)
conf_threshold = 0.50
iou_threshold = 0.45
max_det = 30

# High recall (detect more balls, more false positives)
conf_threshold = 0.20
iou_threshold = 0.40
max_det = 100
```

## Limitations

- **Small objects**: Performance may degrade for tennis balls that are very far from the camera (< 20px)
- **Motion blur**: Fast-moving balls may be harder to detect
- **Occlusion**: Partially hidden balls may not be detected
- **Similar objects**: May occasionally detect other small round objects
- **Lighting**: Optimized for outdoor tennis lighting conditions

## Model Biases

- Trained primarily on standard yellow tennis balls
- Dataset includes various court types but may have dataset-specific biases
- Better performance on professional match footage vs amateur recordings

## Use Cases

✅ **Recommended:**
- Tennis match analysis
- Automated highlight generation
- Player training and coaching
- Sports analytics
- Ball tracking for statistics

⚠️ **Not Recommended:**
- Real-time umpiring decisions (use as assistance only)
- Safety-critical applications
- Detection of non-yellow tennis balls without fine-tuning

## Example Results

### Sample Detections

**Precision: 84.3%** - When the model detects a ball, it's correct 84% of the time
**Recall: 59.5%** - The model detects approximately 6 out of 10 tennis balls

### Confidence Interpretation

| Confidence Range | Interpretation |
|------------------|----------------|
| > 0.7 | High confidence - very likely a tennis ball |
| 0.5 - 0.7 | Medium confidence - probably a tennis ball |
| 0.3 - 0.5 | Low confidence - possible tennis ball |
| < 0.3 | Very low confidence - likely false positive |

## Model Card Authors

- **Developed by**: Vuong
- **Model date**: November 2024
- **Model version**: 1.0
- **Model type**: Object Detection (YOLOv11)

## Citation

If you use this model, please cite:

```bibtex
@misc{yolov11_tennis_ball_2024,
  title={YOLOv11 Tennis Ball Detection},
  author={Vuong},
  year={2024},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/...}}
}
```

## License

MIT License - Free for commercial and academic use.

## Acknowledgments

- Built with [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- Trained on custom annotated tennis dataset
- Part of the Tennis Analysis project

## Contact & Support

For questions, issues, or collaboration:
- GitHub Issues: [tennis_analysis/issues](https://github.com/...)
- Model Updates: Check for newer versions on Hugging Face

## Related Models

- [YOLOv11 Tennis Racket Detection](https://huggingface.co/...) - Companion model for racket detection

## Model Changelog

### v1.0 (2024-11-20)
- Initial release
- YOLOv11n architecture
- mAP@50: 67.87%
- 520 training images

---

**Model Size**: 5.4 MB
**Inference Speed**: 10-65ms (device dependent)
**Supported Formats**: PyTorch (.pt), ONNX, TensorRT, CoreML

🎾 Ready for production use in tennis analysis applications!