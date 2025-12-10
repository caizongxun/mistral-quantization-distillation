#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for Mistral quantization and distillation project
"""

import os
import sys
import json
import torch
import psutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# System Information Functions
# ============================================================================

def get_system_info() -> Dict:
    """
    Get comprehensive system information
    """
    return {
        'os': platform.system(),
        'os_version': platform.version(),
        'python_version': sys.version.split()[0],
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'total_ram_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'available_ram_gb': round(psutil.virtual_memory().available / (1024**3), 2),
    }

def get_gpu_info() -> Dict:
    """
    Get GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpus': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                'compute_capability': torch.cuda.get_device_capability(i)
            }
            info['gpus'].append(gpu_info)
    
    return info

def check_mps_available() -> bool:
    """
    Check if Apple Metal Performance Shaders (MPS) is available
    """
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def get_device() -> str:
    """
    Get optimal device for computation
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif check_mps_available():
        return 'mps'
    else:
        return 'cpu'

# ============================================================================
# Memory Monitoring Functions
# ============================================================================

class MemoryMonitor:
    """
    Monitor GPU and CPU memory usage
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.start_memory = 0
    
    def reset(self):
        """
        Reset memory tracking
        """
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.start_memory = self.get_current_memory()
    
    def get_current_memory(self) -> float:
        """
        Get current memory usage in GB
        """
        if self.device == 'cuda':
            return torch.cuda.memory_allocated() / (1024**3)
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024**3)
    
    def get_peak_memory(self) -> float:
        """
        Get peak memory usage in GB
        """
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / (1024**3)
        else:
            return self.get_current_memory()
    
    def get_memory_diff(self) -> float:
        """
        Get memory difference since reset in GB
        """
        return self.get_current_memory() - self.start_memory
    
    def log_memory(self, prefix=""):
        """
        Log current memory usage
        """
        current = self.get_current_memory()
        peak = self.get_peak_memory()
        diff = self.get_memory_diff()
        
        msg = f"{prefix} | Current: {current:.2f}GB | Peak: {peak:.2f}GB | Diff: {diff:+.2f}GB"
        logger.info(msg)
        print(msg)

# ============================================================================
# Configuration Functions
# ============================================================================

def load_config(config_path: str) -> Dict:
    """
    Load JSON configuration file
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config: Dict, config_path: str):
    """
    Save configuration to JSON file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

# ============================================================================
# Logging Functions
# ============================================================================

def setup_logging(log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    return root_logger

# ============================================================================
# Directory Functions
# ============================================================================

def ensure_directory(directory: str) -> str:
    """
    Ensure directory exists and return path
    """
    os.makedirs(directory, exist_ok=True)
    return directory

def create_project_structure():
    """
    Create standard project directory structure
    """
    directories = [
        'models/mistral-7b-4bit',
        'models/mistral-7b-fp16',
        'models/phi-2-distilled',
        'models/checkpoints',
        'configs',
        'outputs',
        'data',
        'logs',
        'docs'
    ]
    
    for directory in directories:
        ensure_directory(directory)
    
    logger.info(f"Project structure created at {os.getcwd()}")

# ============================================================================
# Timing Functions
# ============================================================================

class Timer:
    """
    Context manager for timing code execution
    """
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.end = None
    
    def __enter__(self):
        self.start = datetime.now()
        logger.info(f"Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.end = datetime.now()
        duration = (self.end - self.start).total_seconds()
        logger.info(f"Finished: {self.name} | Duration: {duration:.2f}s")
    
    @property
    def elapsed(self) -> float:
        if self.end is None:
            return (datetime.now() - self.start).total_seconds()
        return (self.end - self.start).total_seconds()

# ============================================================================
# Formatting Functions
# ============================================================================

def format_memory(bytes_size: float) -> str:
    """
    Format bytes to human-readable format
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f}PB"

def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"

# ============================================================================
# Model Functions
# ============================================================================

def get_model_size(model) -> Tuple[float, int]:
    """
    Get model size in GB and parameter count
    """
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    
    return param_size / (1024**3), param_count

def print_model_stats(model, model_name: str = "Model"):
    """
    Print model statistics
    """
    size_gb, param_count = get_model_size(model)
    logger.info(f"{model_name} Stats:")
    logger.info(f"  Parameters: {param_count:,}")
    logger.info(f"  Size: {size_gb:.2f}GB")
    print(f"\n{model_name} Stats:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Size: {size_gb:.2f}GB\n")

if __name__ == "__main__":
    # Test utility functions
    setup_logging()
    
    print("\n=== System Information ===")
    print(json.dumps(get_system_info(), indent=2, ensure_ascii=False))
    
    print("\n=== GPU Information ===")
    print(json.dumps(get_gpu_info(), indent=2, ensure_ascii=False))
    
    print(f"\n=== Optimal Device ===")
    print(f"Device: {get_device()}")
    print(f"MPS Available: {check_mps_available()}")
