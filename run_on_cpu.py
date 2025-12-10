#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU-Optimized Runner with Automatic Device Detection
Use this to run on CPU with optimizations when CUDA is not available
"""

import torch
import os
import sys
from pathlib import Path

def check_device():
    """
    Automatically detect and recommend best device
    """
    print("\n" + "="*60)
    print("🔍 Device Detection Report")
    print("="*60)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n✅ CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Check CPU
    cpu_count = os.cpu_count()
    print(f"\n✅ CPU Cores: {cpu_count}")
    
    # Check MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"\n✅ Apple MPS: {mps_available}")
    
    # Recommendation
    print("\n" + "-"*60)
    print("🎯 Recommendation:")
    
    if cuda_available:
        print("   🚀 Use GPU (CUDA) for fastest training!")
        print(f"   Command: python -m torch.utils.benchmark -m main")
        return "cuda"
    elif mps_available:
        print("   🚀 Use Apple Metal (MPS) - good performance!")
        print("   ⚠️  Still slower than NVIDIA GPU for LLMs")
        return "mps"
    else:
        print("   ⚠️  Using CPU - training will be slow")
        print("   💡 Recommendation: Use Google Colab (Free GPU!)")
        print("   💡 Or AWS/Azure with GPU instances")
        return "cpu"

def optimize_for_cpu():
    """
    Apply CPU optimizations
    """
    print("\n🔧 Applying CPU Optimizations...")
    
    # Disable CUDA to force CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TORCH_HOME'] = './torch_cache'
    
    # CPU threading
    torch.set_num_threads(os.cpu_count())
    print(f"   ✅ Set threads to {os.cpu_count()}")
    
    # Disable benchmarking (can slow CPU inference)
    torch.backends.cudnn.benchmark = False
    print("   ✅ Disabled CUDA benchmarking")
    
    # Optimize memory
    torch.backends.cudnn.deterministic = True
    print("   ✅ Enabled deterministic mode")
    
    return "cpu"

def run_on_device(script_name: str, device: str = None, **kwargs):
    """
    Run a script on the best available device
    
    Args:
        script_name: Name of the script to run (e.g., 'mistral_quantization')
        device: Force specific device ('cuda', 'cpu', 'mps'). If None, auto-detect.
        **kwargs: Additional arguments to pass to the script
    """
    print("\n" + "="*60)
    print(f"🚀 Running {script_name}")
    print("="*60)
    
    # Detect device if not specified
    if device is None:
        device = check_device()
    
    # Optimize for CPU if needed
    if device == "cpu":
        device = optimize_for_cpu()
        print("\n⏱️  Warning: CPU inference is slow!")
        print("   First run: ~15-30 minutes per model")
        print("   Consider using Google Colab instead")
    
    # Build command
    import subprocess
    cmd = ["python", f"{script_name}.py", f"--device", device]
    
    # Add extra arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))
    
    print(f"\n📝 Command: {' '.join(cmd)}\n")
    
    # Run
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    """
    Interactive CPU training menu
    """
    print("\n" + "="*60)
    print("🤖 Mistral LLM CPU Training Suite")
    print("="*60)
    
    device = check_device()
    
    print("\n" + "="*60)
    print("📋 Available Options:")
    print("="*60)
    print("\n1. Setup environment and check devices")
    print("2. Download and quantize Mistral-7B (4-bit)")
    print("3. Run benchmark (FP16 vs 4-bit)")
    print("4. Train Phi-2 with distillation")
    print("5. Run inference comparison")
    print("6. Launch Gradio demo")
    print("\n0. Exit")
    
    choice = input("\nSelect option (0-6): ").strip()
    
    if choice == "1":
        print("\n✅ Device check complete!")
        if device == "cpu":
            print("\n💡 Suggestion: Use Colab for faster training")
            print("   1. Go to https://colab.research.google.com")
            print("   2. Upload the colab_full_pipeline.ipynb notebook")
            print("   3. Runtime > Change runtime type > GPU (T4)")
    
    elif choice == "2":
        print("\n⏱️  Quantization on CPU will take 20-40 minutes...")
        if input("Continue? (y/n): ").lower() != 'y':
            print("Skipped.")
            return
        run_on_device("mistral_quantization", device=device, output="models/mistral-7b-4bit")
    
    elif choice == "3":
        print("\n⏱️  Benchmark on CPU will take 30+ minutes...")
        if input("Continue? (y/n): ").lower() != 'y':
            print("Skipped.")
            return
        run_on_device("benchmark", device=device)
    
    elif choice == "4":
        print("\n⏱️  Distillation training on CPU will take 2-4 hours...")
        print("   ⚠️  STRONGLY recommend using Google Colab instead!")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            print("Skipped.")
            return
        run_on_device(
            "distillation_training",
            device=device,
            samples=500,
            epochs=3,
            batch_size=4
        )
    
    elif choice == "5":
        run_on_device("inference_comparison", device=device)
    
    elif choice == "6":
        print("\nLaunching Gradio demo...")
        import subprocess
        subprocess.run(["python", "app.py"])
    
    elif choice == "0":
        print("\nGoodbye!")
        return
    
    else:
        print("Invalid option.")
        return
    
    # Show recommendations
    print("\n" + "="*60)
    print("✨ Next Steps:")
    print("="*60)
    if device == "cpu":
        print("\n1. For faster training, use Google Colab:")
        print("   - colab_full_pipeline.ipynb")
        print("   - Free T4 GPU (25x faster than CPU)")
        print("\n2. Then download trained models to use locally")
    else:
        print(f"\n✅ You have {device.upper()} available!")
        print("   Training should be reasonably fast.")

if __name__ == "__main__":
    main()
