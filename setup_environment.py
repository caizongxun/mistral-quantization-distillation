#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated environment setup for Mistral quantization and distillation
Supports: Windows, macOS, Linux
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
from typing import Tuple, Dict

class EnvironmentSetup:
    """
    Comprehensive environment setup manager
    """
    
    def __init__(self):
        self.os_type = platform.system()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.absolute()
    
    def print_header(self, title: str):
        """
        Print formatted header
        """
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    
    def print_success(self, message: str):
        """
        Print success message
        """
        print(f"✅ {message}")
    
    def print_error(self, message: str):
        """
        Print error message
        """
        print(f"❌ {message}")
    
    def print_info(self, message: str):
        """
        Print info message
        """
        print(f"ℹ️  {message}")
    
    def print_warning(self, message: str):
        """
        Print warning message
        """
        print(f"⚠️  {message}")
    
    # ========================================================================
    # System Check Functions
    # ========================================================================
    
    def check_python_version(self) -> bool:
        """
        Check if Python version meets requirements (3.10+)
        """
        self.print_header("Python Version Check")
        
        print(f"Current Python: {sys.version}")
        
        if self.python_version.major < 3 or (self.python_version.major == 3 and self.python_version.minor < 10):
            self.print_error(f"Python 3.10+ required, but found {self.python_version.major}.{self.python_version.minor}")
            return False
        
        self.print_success(f"Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} is compatible")
        return True
    
    def check_cuda(self) -> Tuple[bool, Dict]:
        """
        Check CUDA installation and configuration
        """
        self.print_header("CUDA Configuration Check")
        
        info = {
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_info': []
        }
        
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            info['cuda_version'] = torch.version.cuda
            
            if info['cuda_available']:
                info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(info['gpu_count']):
                    gpu_info = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
                    }
                    info['gpu_info'].append(gpu_info)
                
                self.print_success(f"CUDA {info['cuda_version']} detected")
                self.print_success(f"Found {info['gpu_count']} GPU(s)")
                for gpu in info['gpu_info']:
                    self.print_info(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']}GB)")
                return True, info
            else:
                self.print_warning("CUDA not available. Will use CPU mode (slower)")
                return False, info
        except Exception as e:
            self.print_error(f"Error checking CUDA: {e}")
            return False, info
    
    def check_mps(self) -> Tuple[bool, Dict]:
        """
        Check Apple Metal Performance Shaders (macOS only)
        """
        info = {'mps_available': False, 'mps_version': None}
        
        if self.os_type != 'Darwin':
            return False, info
        
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info['mps_available'] = True
                self.print_success("Apple Metal Performance Shaders (MPS) available")
                return True, info
        except Exception as e:
            self.print_warning(f"MPS check failed: {e}")
        
        return False, info
    
    # ========================================================================
    # Virtual Environment Functions
    # ========================================================================
    
    def create_virtual_environment(self) -> bool:
        """
        Create Python virtual environment
        """
        self.print_header("Virtual Environment Setup")
        
        venv_path = self.project_root / 'venv'
        
        if venv_path.exists():
            self.print_info(f"Virtual environment already exists at {venv_path}")
            return True
        
        try:
            self.print_info(f"Creating virtual environment at {venv_path}...")
            subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], 
                         check=True, capture_output=True)
            self.print_success(f"Virtual environment created at {venv_path}")
            
            # Print activation command
            if self.os_type == 'Windows':
                activate_cmd = f"{venv_path}\\Scripts\\activate"
            else:
                activate_cmd = f"source {venv_path}/bin/activate"
            
            self.print_info(f"To activate virtual environment, run:")
            print(f"  {activate_cmd}")
            
            return True
        except Exception as e:
            self.print_error(f"Failed to create virtual environment: {e}")
            return False
    
    # ========================================================================
    # Dependencies Installation
    # ========================================================================
    
    def install_dependencies(self) -> bool:
        """
        Install project dependencies
        """
        self.print_header("Installing Dependencies")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            self.print_error(f"requirements.txt not found at {requirements_file}")
            return False
        
        try:
            self.print_info("Installing packages from requirements.txt...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                         check=True, capture_output=True)
            
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)],
                         check=True)
            
            self.print_success("All dependencies installed successfully")
            return True
        except Exception as e:
            self.print_error(f"Failed to install dependencies: {e}")
            return False
    
    # ========================================================================
    # Validation Functions
    # ========================================================================
    
    def validate_installation(self) -> bool:
        """
        Validate that all required packages are installed
        """
        self.print_header("Validating Installation")
        
        required_packages = [
            'torch',
            'transformers',
            'bitsandbytes',
            'peft',
            'trl',
            'datasets',
            'gradio',
            'accelerate'
        ]
        
        all_installed = True
        
        for package in required_packages:
            try:
                __import__(package)
                self.print_success(f"{package} is installed")
            except ImportError:
                self.print_error(f"{package} is NOT installed")
                all_installed = False
        
        return all_installed
    
    # ========================================================================
    # Project Structure Functions
    # ========================================================================
    
    def create_project_structure(self) -> bool:
        """
        Create standard project directory structure
        """
        self.print_header("Creating Project Structure")
        
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
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.print_info(f"Directory created: {directory}")
            
            self.print_success("Project structure created successfully")
            return True
        except Exception as e:
            self.print_error(f"Failed to create project structure: {e}")
            return False
    
    # ========================================================================
    # Configuration Functions
    # ========================================================================
    
    def create_default_configs(self) -> bool:
        """
        Create default configuration files
        """
        self.print_header("Creating Configuration Files")
        
        # Quantization config
        quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "device_map": "auto"
        }
        
        # Training config
        training_config = {
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "save_strategy": "steps",
            "save_steps": 100,
            "logging_steps": 10,
            "fp16": True,
            "optim": "paged_adamw_8bit"
        }
        
        # Inference config
        inference_config = {
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "num_beams": 1
        }
        
        configs = {
            'quantization_config.json': quantization_config,
            'training_config.json': training_config,
            'inference_config.json': inference_config
        }
        
        try:
            for config_name, config_data in configs.items():
                config_path = self.project_root / 'configs' / config_name
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                self.print_success(f"Created {config_name}")
            
            return True
        except Exception as e:
            self.print_error(f"Failed to create configuration files: {e}")
            return False
    
    # ========================================================================
    # Main Setup Function
    # ========================================================================
    
    def run_full_setup(self) -> bool:
        """
        Run complete environment setup
        """
        print("\n")
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  🚀 Mistral-7B Quantization & Distillation Setup              ║")
        print("║  Environment Configuration for All Platforms                  ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        
        print(f"\n📊 System Information:")
        print(f"  OS: {self.os_type}")
        print(f"  Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print(f"  Project Root: {self.project_root}")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            self.print_error("Python version check failed. Please upgrade to Python 3.10+")
            return False
        
        # Step 2: Check CUDA/MPS
        cuda_available, cuda_info = self.check_cuda()
        mps_available, mps_info = self.check_mps()
        
        # Step 3: Create virtual environment
        if not self.create_virtual_environment():
            self.print_warning("Virtual environment creation skipped")
        
        # Step 4: Install dependencies
        if not self.install_dependencies():
            self.print_error("Dependency installation failed")
            return False
        
        # Step 5: Validate installation
        if not self.validate_installation():
            self.print_warning("Some packages may not be installed correctly")
        
        # Step 6: Create project structure
        if not self.create_project_structure():
            self.print_error("Project structure creation failed")
            return False
        
        # Step 7: Create configuration files
        if not self.create_default_configs():
            self.print_warning("Configuration file creation had issues")
        
        # Final summary
        self.print_header("✅ Setup Complete!")
        
        print("\n📋 Summary:")
        print(f"  ✅ Python {self.python_version.major}.{self.python_version.minor} configured")
        print(f"  ✅ CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"  ✅ GPU Count: {cuda_info['gpu_count']}")
        print(f"  ✅ Project structure created")
        print(f"  ✅ Dependencies installed")
        
        print("\n🚀 Next Steps:")
        print("  1. Activate virtual environment:")
        if self.os_type == 'Windows':
            print(f"     .\\venv\\Scripts\\activate")
        else:
            print(f"     source venv/bin/activate")
        print("\n  2. Download and quantize Mistral-7B:")
        print("     python mistral_quantization.py")
        print("\n  3. Run the interactive Gradio demo:")
        print("     python app.py")
        print("\n  4. Check documentation:")
        print("     - docs/SETUP_GUIDE.md for detailed setup")
        print("     - docs/QUANTIZATION.md for quantization details")
        print("     - docs/DISTILLATION.md for distillation guide")
        
        return True

def main():
    """
    Main entry point
    """
    setup = EnvironmentSetup()
    success = setup.run_full_setup()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
