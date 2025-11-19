# Installation Guide

## Quick Installation

### Step 1: Install Core Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Install Action Recognition Dependencies

**Important:** `pytorchvideo` is required for action recognition.

```bash
# Install fvcore first
pip install fvcore

# Then install pytorchvideo
pip install pytorchvideo
```

If the above fails, try installing from source:

```bash
pip install fvcore
pip install git+https://github.com/facebookresearch/pytorchvideo.git
```

### Step 3: Test Installation

```bash
python test_installation.py
```

## Installation Issues & Solutions

### Issue 1: pytorchvideo Installation Fails

**Symptoms:**
```
ModuleNotFoundError: No module named 'pytorchvideo'
```

**Solution A - Install from PyPI:**
```bash
pip install fvcore
pip install pytorchvideo
```

**Solution B - Install from GitHub:**
```bash
pip install fvcore
pip install git+https://github.com/facebookresearch/pytorchvideo.git
```

**Solution C - Use without pytorchvideo:**

If you don't need action recognition, you can use the detection and tracking features without pytorchvideo. The system will work for:
- ✅ Player detection
- ✅ Ball detection
- ✅ Multi-object tracking
- ❌ Action recognition (requires pytorchvideo)

### Issue 2: mmcv/mmpose Installation Fails

**Note:** These are optional dependencies for advanced pose estimation.

**Solution:**

The basic pose estimator works without mmpose. If you need mmpose:

```bash
# For macOS ARM (M1/M2/M3)
pip install mmengine
pip install mmcv
pip install mmpose
```

If this fails, you can skip it - the built-in pose estimator will work.

### Issue 3: CUDA Not Available

**Symptoms:**
```
⚠ CUDA not available - will use CPU
```

**This is OK!** The system works on CPU, just slower.

**To use GPU:**
1. Install CUDA from NVIDIA
2. Reinstall PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Platform-Specific Instructions

### macOS (M1/M2/M3)

```bash
# Use Metal acceleration for PyTorch
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt

# Install pytorchvideo
pip install fvcore pytorchvideo
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip ffmpeg

# Install Python packages
pip install -r requirements.txt
pip install fvcore pytorchvideo
```

### Windows

```bash
# Install Python 3.9+
# Then in PowerShell or CMD:

pip install -r requirements.txt
pip install fvcore pytorchvideo
```

## Minimal Installation (Without Action Recognition)

If you only need detection and tracking:

```bash
pip install -r requirements-minimal.txt
```

This installs everything except pytorchvideo and fvcore.

**Features available:**
- ✅ Object detection (YOLO)
- ✅ Multi-object tracking (OCSORT)
- ✅ Video processing
- ✅ Visualization
- ❌ Action recognition

## Verify Installation

After installation, verify all components:

```bash
python test_installation.py
```

Expected output:
```
============================================================
TEST SUMMARY
============================================================
Package Imports           ✅ PASSED
CUDA Support              ✅ PASSED (or ⚠ warning for CPU)
Project Structure         ✅ PASSED
Model Initialization      ✅ PASSED
Utilities                 ✅ PASSED
Simple Inference          ✅ PASSED
============================================================

🎉 All tests passed! Installation is complete.
```

## Troubleshooting

### Python Version

Ensure you're using Python 3.9 or later:

```bash
python --version
# Should show Python 3.9.x or higher
```

### Virtual Environment

Always use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependency Conflicts

If you get dependency conflicts:

```bash
# Create fresh environment
rm -rf venv
python -m venv venv
source venv/bin/activate

# Install step by step
pip install --upgrade pip
pip install torch torchvision
pip install opencv-python numpy pandas pyyaml
pip install ultralytics
pip install scipy filterpy lap
pip install matplotlib seaborn pillow
pip install tqdm loguru moviepy ffmpeg-python
pip install fvcore pytorchvideo
```

### Still Having Issues?

1. Check your Python version: `python --version`
2. Update pip: `pip install --upgrade pip`
3. Clear pip cache: `pip cache purge`
4. Reinstall in fresh environment

## Next Steps

After successful installation:

1. **Test the system:**
   ```bash
   python test_installation.py
   ```

2. **Read the quick start:**
   ```bash
   cat QUICKSTART.md
   ```

3. **Run a simple example:**
   ```bash
   # Place a video in data/input/
   python main.py --video data/input/your_video.mp4
   ```

4. **Check examples:**
   ```bash
   python example_usage.py
   ```

## Support

If you continue to have issues:

1. Run `python test_installation.py` and share the output
2. Check `requirements.txt` is up to date
3. Ensure you're in the correct directory
4. Try the minimal installation first

---

**Installation complete! Ready to analyze tennis videos! 🎾**
