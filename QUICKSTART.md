# Quick Start Guide 🚀

Get started with Tennis Action Analysis in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation
python test_installation.py
```

## Basic Usage

### Option 1: Use Sample Data

```bash
# Create sample directories
mkdir -p data/input data/output weights

# Place your tennis video in data/input/
# cp your_tennis_video.mp4 data/input/tennis_match.mp4

# Run analysis
python main.py --video data/input/tennis_match.mp4 --output-dir data/output
```

### Option 2: Quick Test with Config

```bash
# Edit config.yaml with your paths
nano config.yaml

# Run with config
python main.py --config config.yaml
```

## What You'll Get

After running the pipeline, you'll find in your output directory:

```
data/output/
├── detections.json          # All object detections
├── tracking.json            # Player tracking data
├── actions.json             # Detected tennis actions
├── actions.csv              # Actions in CSV format
├── output_annotated.mp4     # Annotated video
└── clips/                   # Extracted action clips
    ├── clip_0001_forehand.mp4
    ├── clip_0002_serve.mp4
    └── ...
```

## Configuration

Quick config editing for common scenarios:

### Use CPU instead of GPU

```yaml
yolo:
  device: "cpu"
```

### Lower memory usage

```yaml
yolo:
  model_size: "s"  # s instead of m

action_recognition:
  model_size: "s"
```

### Faster processing (lower accuracy)

```yaml
yolo:
  conf_threshold: 0.6  # Higher threshold
  model_size: "n"      # Nano model

action_recognition:
  clip_stride: 16      # Process fewer clips
```

### Higher accuracy (slower)

```yaml
yolo:
  conf_threshold: 0.3  # Lower threshold
  model_size: "l"      # Large model

action_recognition:
  conf_threshold: 0.7  # Higher confidence
  clip_stride: 4       # More clips
```

## Common Commands

### Process single video

```bash
python main.py --video video.mp4 --output-dir results/
```

### Process with custom config

```bash
python main.py --config my_config.yaml --video video.mp4
```

### Train custom action classifier

```bash
python training/train_action_classifier.py \
    --data-root data/ \
    --output-dir runs/train \
    --num-epochs 50 \
    --batch-size 8
```

### Test installation

```bash
python test_installation.py
```

## Troubleshooting

### Error: CUDA out of memory

**Solution 1:** Use smaller models

```yaml
yolo:
  model_size: "n"  # nano model
```

**Solution 2:** Use CPU

```yaml
yolo:
  device: "cpu"
```

### Error: Cannot read video

**Solution:** Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Error: Module not found

**Solution:** Install requirements

```bash
pip install -r requirements.txt
```

### Low detection accuracy

**Solution 1:** Lower confidence threshold

```yaml
yolo:
  conf_threshold: 0.3
```

**Solution 2:** Train on custom data (see README.md)

## Next Steps

1. **Check Examples**: See `example_usage.py` for more usage patterns
2. **Read Full Documentation**: Check `README.md` for detailed information
3. **Train Custom Models**: Follow training instructions in README
4. **Customize Pipeline**: Modify `config.yaml` for your needs

## Getting Help

- Check `README.md` for detailed documentation
- Run `python test_installation.py` to diagnose issues
- See `example_usage.py` for code examples

## Performance Tips

| Hardware | Recommended Settings |
|----------|---------------------|
| GPU 8GB+ | model_size: "m", device: "cuda" |
| GPU 4GB  | model_size: "s", device: "cuda" |
| CPU Only | model_size: "n", device: "cpu" |

## Minimal Working Example

```python
from main import TennisAnalysisPipeline

# Initialize
pipeline = TennisAnalysisPipeline("config.yaml")

# Run
results = pipeline.run(video_path="video.mp4")

# Print results
print(f"Detected {results['summary']['total_actions']} actions")
for action, count in results['summary']['action_counts'].items():
    print(f"  {action}: {count}")
```

---

**Happy analyzing! 🎾**
