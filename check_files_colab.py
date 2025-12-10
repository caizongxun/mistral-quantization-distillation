#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colab 檔案檢查和下載助手
用一行指令檢查所有檔案位置
"""

import os
import json
from pathlib import Path
from datetime import datetime

def format_size(bytes_size):
    """
    Format bytes to human readable
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f}TB"

def check_files():
    """
    Check all generated files in Colab
    """
    print("\n" + "="*70)
    print("📁 COLAB FILES STATUS REPORT")
    print("="*70)
    
    # Current location
    cwd = os.getcwd()
    print(f"\n📂 Working Directory: {cwd}")
    print(f"\n🖭 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Dictionary of important locations
    locations = {
        "Quantized Model": "models/mistral-7b-4bit",
        "Distilled Model": "models/phi-2-distilled",
        "Checkpoints": "models/checkpoints",
        "Results": "outputs",
        "Logs": "logs"
    }
    
    print(f"\n" + "-"*70)
    print("📄 FILES LOCATION")
    print("-"*70)
    
    total_size = 0
    
    for name, path in locations.items():
        full_path = os.path.join(cwd, path)
        
        if os.path.exists(full_path):
            status = "✅"
            
            if os.path.isdir(full_path):
                files = os.listdir(full_path)
                dir_size = sum(
                    os.path.getsize(os.path.join(full_path, f))
                    for f in files
                    if os.path.isfile(os.path.join(full_path, f))
                )
                total_size += dir_size
                size_str = format_size(dir_size)
                print(f"\n{status} {name}")
                print(f"   Path: {full_path}")
                print(f"   Files: {len(files)}")
                print(f"   Size: {size_str}")
                
                # List important files
                important = [
                    'config.json',
                    'tokenizer.json',
                    'model-00001-of-00002.safetensors',
                    'benchmark_results.csv',
                    'chat_history.csv',
                    'distillation_metadata.json',
                    'quantization_metadata.json'
                ]
                
                found_files = []
                for f in files:
                    if any(imp in f for imp in important):
                        fpath = os.path.join(full_path, f)
                        if os.path.isfile(fpath):
                            fsize = os.path.getsize(fpath)
                            found_files.append((f, fsize))
                
                if found_files:
                    print(f"   Key files:")
                    for fname, fsize in sorted(found_files):
                        print(f"      - {fname} ({format_size(fsize)})")
            else:
                print(f"\n{status} {name}")
                print(f"   Path: {full_path}")
                print(f"   Size: {format_size(os.path.getsize(full_path))}")
        else:
            print(f"\n❌ {name}")
            print(f"   Path: {full_path}")
            print(f"   Status: NOT FOUND")
    
    print(f"\n" + "-"*70)
    print("📊 TOTAL SIZE")
    print("-"*70)
    print(f"Total: {format_size(total_size)}")
    
    # Disk space
    print(f"\n" + "-"*70)
    print("🖥️  DISK SPACE (Colab)")
    print("-"*70)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"Total: {format_size(total)}")
        print(f"Used: {format_size(used)}")
        print(f"Free: {format_size(free)}")
        percent_used = (used / total) * 100
        print(f"Usage: {percent_used:.1f}%")
    except Exception as e:
        print(f"Could not read disk space: {e}")
    
    # Download instructions
    print(f"\n" + "="*70)
    print("📥 DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    print("""
✅ Option 1: Google Drive (Recommended for large files)
   
   # In Colab:
   from google.colab import drive
   import shutil
   drive.mount('/content/gdrive')
   
   backup_dir = '/content/gdrive/My Drive/mistral-backup'
   shutil.copytree('models/mistral-7b-4bit', 
                   f'{backup_dir}/mistral-7b-4bit', 
                   dirs_exist_ok=True)
   
   # Then: Download from drive.google.com

✅ Option 2: Direct Download (CSV files only)
   
   from google.colab import files
   files.download('outputs/benchmark_results.csv')
   files.download('outputs/chat_history.csv')

✅ Option 3: Check for existing files
   
   !ls -lah models/mistral-7b-4bit/
   !ls -lah models/phi-2-distilled/
   !ls -lah outputs/
    """)
    
    print("="*70)
    print("🎉 Status check complete!")
    print("="*70)

if __name__ == "__main__":
    check_files()
