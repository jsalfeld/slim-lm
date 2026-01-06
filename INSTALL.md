# Installation Guide

## Quick Installation (Recommended)

### Option 1: Automated Setup
```bash
./setup.sh
```

### Option 2: Manual Installation
```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies (no flash-attn)
pip install -r requirements-core.txt

# Or install all (may fail on flash-attn, that's okay)
pip install -r requirements.txt
```

## Common Installation Issues

### Issue 1: flash-attn Build Failure

**Error**: `Failed to build 'flash-attn'`

**Solution**: This is **normal and okay**! Flash-attn is optional and only works with:
- CUDA installed
- Compatible NVIDIA GPU (Ampere or newer: RTX 30xx/40xx, A100, H100)
- Build tools installed

**What to do**:
1. Use `requirements-core.txt` instead (already excludes flash-attn)
2. Or ignore the error - training will work fine without it
3. Flash-attn only provides ~20-30% speedup, not critical

```bash
pip install -r requirements-core.txt
```

### Issue 2: bitsandbytes Issues

**Error**: `bitsandbytes CUDA not found` or similar

**For Linux/Windows with CUDA**:
```bash
pip install bitsandbytes --force-reinstall
```

**For macOS or CPU-only**:
```bash
# bitsandbytes doesn't work on macOS/CPU
# You'll need to disable quantization in training scripts:
--use_4bit false --use_8bit false
```

**Note**: Without bitsandbytes, you can't use QLoRA (4-bit training). You'll need more VRAM or use CPU training (very slow).

### Issue 3: PyTorch CUDA Not Available

**Error**: `torch.cuda.is_available() returns False`

**Check CUDA installation**:
```bash
nvidia-smi  # Should show your GPU
```

**Reinstall PyTorch with CUDA**:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Visit https://pytorch.org/get-started/locally/ for your specific setup.

### Issue 4: Triton Installation Fails

**Error**: `Failed to build triton`

**Solution**: Triton is optional. Comment it out in requirements.txt:
```bash
# Edit requirements.txt and ensure triton line is commented:
# triton>=2.1.0  # Optional: for optimizations
```

## Platform-Specific Instructions

### Linux with NVIDIA GPU (Recommended Setup)

```bash
# 1. Ensure CUDA is installed
nvidia-smi

# 2. Install dependencies
pip install -r requirements-core.txt

# 3. (Optional) Install flash-attn if you have Ampere+ GPU
pip install flash-attn

# 4. Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### macOS (CPU or Metal)

```bash
# macOS doesn't support CUDA, so no bitsandbytes/flash-attn
# Install core dependencies except bitsandbytes

pip install torch transformers accelerate peft trl datasets evaluate
pip install pandas numpy scikit-learn jsonlines scipy sentencepiece protobuf
pip install wandb tensorboard

# Note: Training will be VERY slow on CPU
# Consider using cloud GPU instead
```

### Windows with NVIDIA GPU

```bash
# 1. Install CUDA toolkit from NVIDIA
# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements-core.txt

# 5. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### CPU-Only (Not Recommended)

```bash
# Install core dependencies
pip install -r requirements-core.txt

# Training will be extremely slow (days instead of hours)
# Use configs/cpu_training.sh with minimal settings
```

## Verification

After installation, verify everything works:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. Check CUDA (if GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Check key libraries
python -c "import transformers, peft, trl, datasets; print('✓ All imports successful')"

# 5. (Optional) Check bitsandbytes
python -c "import bitsandbytes as bnb; print('✓ bitsandbytes working')"
```

## Minimal Installation (No GPU, Testing Only)

If you just want to test the scripts without actual training:

```bash
pip install torch transformers datasets pandas jsonlines
```

This is enough to:
- Run data preparation utilities
- Validate dataset formats
- Read and understand the code
- Test on tiny datasets (will be very slow)

## Dependency Size Reference

| Package | Size | Required | Notes |
|---------|------|----------|-------|
| torch | ~2GB | Yes | Core deep learning |
| transformers | ~500MB | Yes | Model loading |
| bitsandbytes | ~10MB | For GPU | Enables QLoRA |
| flash-attn | ~50MB | Optional | 20-30% speedup |
| trl | ~20MB | Yes | Training utilities |
| datasets | ~100MB | Yes | Data loading |
| Other | ~200MB | Yes | Various utilities |

**Total**: ~3GB with all dependencies

## Getting Help

If you're still having issues:

1. **Check Python version**: `python --version` (need 3.10+)
2. **Check pip version**: `pip --version` (update with `pip install --upgrade pip`)
3. **Check CUDA version**: `nvidia-smi` (if using GPU)
4. **Try minimal install**: `pip install -r requirements-core.txt`
5. **Check error logs**: Look for specific error messages

## Cloud GPU Setup (If Local Installation Fails)

If you can't get it working locally, consider cloud GPU:

### Vast.ai
```bash
# 1. Create account at vast.ai
# 2. Rent GPU instance ($0.20-0.50/hour)
# 3. Clone project
git clone [your-repo]
cd [your-repo]

# 4. Setup
./setup.sh

# 5. Train
bash configs/sft_qlora_single_gpu.sh
```

### Google Colab (Free/Paid)
```python
# In Colab notebook:
!git clone [your-repo]
%cd [your-repo]
!pip install -r requirements-core.txt
!bash configs/sft_qlora_single_gpu.sh
```

### RunPod
Similar to Vast.ai - rent GPU, clone repo, run setup.

## Next Steps

Once installation is successful:

1. ✓ **Login to Hugging Face**: `huggingface-cli login`
2. ✓ **Review QUICKSTART.md**: Get started with training
3. ✓ **Prepare your dataset**: Use templates in `datasets/`
4. ✓ **Start training**: `bash configs/sft_qlora_single_gpu.sh`

---

**Still stuck?** Create an issue with:
- Your OS and Python version
- GPU type (if any)
- Full error message
- Output of `pip list`
