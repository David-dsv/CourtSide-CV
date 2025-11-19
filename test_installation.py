"""
Installation Test Script
Verify that all dependencies are installed correctly
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    print("=" * 60)

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "yaml": "PyYAML",
        "ultralytics": "Ultralytics YOLO",
        "scipy": "SciPy",
        "pytorchvideo": "PyTorchVideo",
        "loguru": "Loguru",
        "tqdm": "tqdm",
        "PIL": "Pillow"
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20} OK")
        except ImportError as e:
            print(f"✗ {name:20} FAILED: {e}")
            failed.append(name)

    print("=" * 60)

    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nPlease install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    print("=" * 60)

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA is available")
            print(f"  - Device count: {device_count}")
            print(f"  - Device name: {device_name}")
            print(f"  - CUDA version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available - will use CPU")
            print("  This is OK but inference will be slower")

        print("=" * 60)
        return True

    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        print("=" * 60)
        return False


def test_models():
    """Test that model classes can be instantiated"""
    print("\nTesting model instantiation...")
    print("=" * 60)

    try:
        # Test YOLO
        from models.yolo_detector import TennisYOLODetector

        detector = TennisYOLODetector(
            model_version="yolov8",
            model_size="n",  # Use nano for speed
            device="cpu"
        )
        print("✓ YOLO detector initialized")

        # Test Tracker
        from models.tracker import OCSort

        tracker = OCSort()
        print("✓ Tracker initialized")

        # Test Action Classifier
        from models.action_classifier import TennisActionClassifier

        classifier = TennisActionClassifier(
            model_type="x3d",
            model_size="s",  # Use small for speed
            num_classes=6,
            pretrained=False
        )
        print("✓ Action classifier initialized")

        # Test Pose Estimator
        from models.pose_estimator import SimplePoseEstimator

        pose = SimplePoseEstimator(pretrained=False)
        print("✓ Pose estimator initialized")

        print("=" * 60)
        print("\n✅ All models initialized successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error initializing models: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def test_utilities():
    """Test utility functions"""
    print("\nTesting utilities...")
    print("=" * 60)

    try:
        # Test file utils
        from utils.file_utils import load_yaml, save_json
        import tempfile
        import os

        # Test YAML loading
        if Path("config.yaml").exists():
            config = load_yaml("config.yaml")
            print("✓ Config file loaded")

        # Test JSON save/load
        test_data = {"test": "data", "number": 123}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        save_json(test_data, temp_file)
        os.unlink(temp_file)
        print("✓ File utilities working")

        # Test video utils
        from utils.video_utils import VideoReader, VideoWriter
        print("✓ Video utilities imported")

        # Test drawing utils
        from utils.drawing import draw_bboxes, draw_tracks
        print("✓ Drawing utilities imported")

        # Test preprocessing
        from utils.preprocessing import normalize_image, preprocess_for_x3d
        print("✓ Preprocessing utilities imported")

        print("=" * 60)
        print("\n✅ All utilities working!")
        return True

    except Exception as e:
        print(f"\n✗ Error testing utilities: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def test_project_structure():
    """Test that all required files exist"""
    print("\nTesting project structure...")
    print("=" * 60)

    required_files = [
        "config.yaml",
        "requirements.txt",
        "main.py",
        "models/yolo_detector.py",
        "models/tracker.py",
        "models/action_classifier.py",
        "models/pose_estimator.py",
        "utils/video_utils.py",
        "utils/drawing.py",
        "utils/preprocessing.py",
        "utils/file_utils.py",
        "training/train_action_classifier.py",
        "training/dataset.py",
        "training/transforms.py",
        "inference/detect.py",
        "inference/track.py",
        "inference/extract_clips.py",
        "inference/predict_actions.py",
        "inference/annotate_video.py"
    ]

    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing.append(file_path)

    print("=" * 60)

    if missing:
        print(f"\n❌ {len(missing)} file(s) missing:")
        for f in missing:
            print(f"   - {f}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def test_simple_inference():
    """Test simple inference on dummy data"""
    print("\nTesting simple inference...")
    print("=" * 60)

    try:
        import torch
        import numpy as np
        from models.yolo_detector import TennisYOLODetector

        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test detection
        detector = TennisYOLODetector(
            model_version="yolov8",
            model_size="n",
            device="cpu"
        )

        detections = detector.detect(dummy_frame)
        print(f"✓ Detection ran successfully")
        print(f"  - Detected {len(detections['boxes'])} objects")

        # Test action classifier
        from models.action_classifier import TennisActionClassifier

        model = TennisActionClassifier(
            model_type="x3d",
            model_size="s",
            num_classes=6,
            pretrained=False
        )

        # Create dummy clip
        dummy_clip = torch.randn(1, 3, 16, 224, 224)
        output = model(dummy_clip)
        print(f"✓ Action classifier ran successfully")
        print(f"  - Output shape: {output.shape}")

        print("=" * 60)
        print("\n✅ Inference tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Inference test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("TENNIS ACTION ANALYSIS - Installation Test")
    print("=" * 60)
    print()

    results = {
        "Package Imports": test_imports(),
        "CUDA Support": test_cuda(),
        "Project Structure": test_project_structure(),
        "Model Initialization": test_models(),
        "Utilities": test_utilities(),
        "Simple Inference": test_simple_inference()
    }

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25} {status}")

    print("=" * 60)

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! Installation is complete.")
        print("\nYou can now run:")
        print("   python main.py --video path/to/video.mp4")
        print("\nOr check example_usage.py for more examples.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("You may need to:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check that all project files are present")
        print("   3. Ensure CUDA is properly configured (optional)")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
