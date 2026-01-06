#!/bin/bash
# Setup script for Apertus 8B Post-Training Framework

echo "=================================="
echo "Apertus 8B Post-Training Setup"
echo "=================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "✓ Python version: $python_version"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment found"
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
echo ""

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All dependencies installed successfully!"
else
    echo ""
    echo "⚠ Installation completed with warnings (this is normal)"
    echo "  flash-attn and triton are optional - training will work without them"
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Login to Hugging Face: huggingface-cli login"
echo "3. (Optional) Setup W&B: wandb login"
echo "4. Read QUICKSTART.md to begin training"
echo ""
echo "Quick test:"
echo "  bash configs/sft_qlora_single_gpu.sh"
echo ""
