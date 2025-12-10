# 🛠️ Complete Setup Guide

## System Requirements

### Minimum Configuration
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or higher
- **RAM**: 16GB system memory
- **Storage**: 50GB free space
- **Python**: 3.10+
- **CUDA**: 12.0+ (for NVIDIA GPUs)

### Recommended Configuration
- **GPU**: RTX 4080 (24GB+) or A100
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD
- **Python**: 3.11+

## Step 1: Clone Repository

```bash
git clone https://github.com/caizongxun/mistral-quantization-distillation.git
cd mistral-quantization-distillation
```

## Step 2: Run Environment Setup

```bash
python setup_environment.py
```

This script will:
- ✅ Check Python version
- ✅ Verify CUDA/GPU setup
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Create project structure
- ✅ Generate config files

## Step 3: Activate Virtual Environment

### Windows:
```cmd
.\venv\Scripts\activate
```

### macOS/Linux:
```bash
source venv/bin/activate
```

## Step 4: Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should output: `True` (for GPU) or `False` (for CPU)

## Step 5: Download and Quantize Mistral

```bash
python mistral_quantization.py
```

**First run will take ~10 minutes** (downloading model)

### What happens:
1. Downloads Mistral-7B from Hugging Face
2. Applies 4-bit BitsAndBytes quantization
3. Tests inference
4. Saves to `models/mistral-7b-4bit/`

## Step 6: Run Benchmarks (Optional)

```bash
python benchmark.py
```

Outputs comparison table to `outputs/benchmark_results.csv`

## Step 7: Launch Interactive Demo

```bash
python app.py
```

Then open: **http://localhost:7860**

---

## Troubleshooting

### CUDA Not Found

```bash
# Verify CUDA installation
nvidia-smi

# If error, reinstall NVIDIA drivers
# Windows: https://www.nvidia.com/Download/driverDetails.aspx
# Linux: sudo apt install nvidia-driver-52X (adjust version)
```

### Out of Memory (OOM)

```bash
# Use smaller batch size
python distillation_training.py --batch-size 2 --gradient-accumulation-steps 4
```

### Model Download Fails

```bash
# Login to Hugging Face
huggingface-cli login
# Paste your token when prompted
```

### Port 7860 Already in Use

```bash
# Use different port
python app.py --port 7861
```

---

## Next Steps

1. **Quantization Deep Dive**: See `docs/QUANTIZATION.md`
2. **Distillation Guide**: See `docs/DISTILLATION.md`
3. **Advanced Usage**: Check individual script docstrings

