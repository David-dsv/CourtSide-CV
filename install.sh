#!/bin/bash

echo "=========================================="
echo "Tennis Action Analysis - Installation"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install core dependencies
echo "Step 1/3: Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements-minimal.txt
echo ""

# Install action recognition dependencies with specific handling
echo "Step 2/3: Installing action recognition models..."
echo "Installing pytorchvideo and fvcore..."

# Try to install pytorchvideo
pip install pytorchvideo fvcore || {
    echo "Warning: pytorchvideo installation failed."
    echo "Trying alternative installation..."
    pip install fvcore
    pip install git+https://github.com/facebookresearch/pytorchvideo.git || {
        echo "Warning: Could not install pytorchvideo."
        echo "Action recognition may not work without it."
    }
}
echo ""

# Optional: Pose estimation
echo "Step 3/3: Optional dependencies..."
read -p "Do you want to install pose estimation dependencies? (may fail on some systems) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing pose estimation dependencies..."
    pip install mmengine mmcv mmpose || {
        echo "Warning: Pose estimation installation failed."
        echo "You can skip this - pose estimation is optional."
    }
else
    echo "Skipping pose estimation dependencies (optional)."
fi
echo ""

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test installation: python test_installation.py"
echo "2. Run quick start: python main.py --help"
echo "3. Read documentation: cat QUICKSTART.md"
echo ""
