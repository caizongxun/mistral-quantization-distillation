#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mistral-7B 4-bit Quantization with Automatic Device Detection
Supports: CUDA, CPU, MPS (Apple Silicon)
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import Timer, MemoryMonitor, setup_logging

logger = setup_logging('logs/quantization.log')

def get_optimal_device():
    """
    Auto-detect optimal device
    """
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {device_name} ({device_memory:.2f}GB)")
        print(f"\n✅ GPU Detected: {device_name} ({device_memory:.2f}GB)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Apple Metal (MPS) detected")
        print(f"\n✅ Apple Metal (MPS) Detected")
    else:
        device = 'cpu'
        logger.warning("No GPU detected - using CPU (slower)")
        print(f"\n⚠️  No GPU detected - using CPU (slower)")
        print(f"   Recommendation: Use Google Colab for faster processing")
        
        # Optimize CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_num_threads(os.cpu_count())
        torch.backends.cudnn.enabled = False
    
    return device

class MistralQuantizer:
    """
    Mistral-7B quantization manager with device auto-detection
    """
    
    def __init__(self, 
                 model_id: str = "mistralai/Mistral-7B-v0.1",
                 output_dir: str = "models/mistral-7b-4bit",
                 device: str = None):
        """
        Initialize quantizer with automatic device detection
        
        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory for quantized model
            device: Force device ('cuda', 'cpu', 'mps'). If None, auto-detect.
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.device = device or get_optimal_device()
        self.memory_monitor = MemoryMonitor(self.device)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Device: {self.device}")
    
    def load_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes 4-bit quantization configuration
        """
        print("\n🔧 Creating BitsAndBytes 4-bit Configuration...")
        
        # Note: BitsAndBytes config can be created even on CPU
        # But actual 4-bit loading requires CUDA or MPS
        if self.device == 'cpu':
            logger.warning("4-bit quantization requires GPU - falling back to config only")
            print("⚠️  Note: 4-bit quantization requires GPU")
            print("         Config created but actual loading needs CUDA/MPS")
        
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        logger.info("BitsAndBytes Configuration created")
        print("✅ Configuration created")
        return config
    
    def download_and_quantize(self) -> tuple:
        """
        Download Mistral model and apply 4-bit quantization
        (Will use CPU fallback if needed)
        
        Returns:
            (model, tokenizer) tuple
        """
        self.memory_monitor.reset()
        
        if self.device == 'cpu':
            print("\n⚠️  CPU mode: Using FP16 instead of 4-bit quantization")
            print("   (4-bit requires GPU. For actual 4-bit, use Colab)")
            return self._load_fp16()
        
        with Timer(f"Downloading and Quantizing {self.model_id}") as timer:
            quant_config = self.load_quantization_config()
            
            print("\n💫 Downloading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            print("✅ Tokenizer downloaded")
            
            print(f"\n🔄 Loading {self.model_id} with 4-bit quantization...")
            print("   This may take several minutes...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            print("✅ Model loaded with 4-bit quantization")
            self.memory_monitor.log_memory("After Quantization")
            
            return model, tokenizer
    
    def _load_fp16(self) -> tuple:
        """
        Fallback: Load model in FP16 (for CPU)
        """
        with Timer(f"Loading {self.model_id} in FP16") as timer:
            print(f"\n🔄 Loading {self.model_id} in FP16...")
            print("   This may take 10-20 minutes on CPU...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            print("✅ Model loaded in FP16")
            return model, tokenizer
    
    def save_quantized_model(self, model, tokenizer):
        """
        Save quantized model and tokenizer
        """
        print(f"\n💾 Saving model to {self.output_dir}...")
        
        try:
            model.save_pretrained(
                str(self.output_dir),
                safe_serialization=True
            )
            logger.info(f"Model saved to {self.output_dir}")
            
            tokenizer.save_pretrained(str(self.output_dir))
            logger.info(f"Tokenizer saved to {self.output_dir}")
            
            # Save metadata
            metadata = {
                'model_id': self.model_id,
                'quantization_method': '4-bit-bnb' if self.device != 'cpu' else 'fp16-cpu',
                'device': self.device,
                'note': 'CPU fallback mode' if self.device == 'cpu' else ''
            }
            
            metadata_path = self.output_dir / 'quantization_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print("✅ Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            print(f"❌ Error: {e}")
            raise
    
    def test_inference(self, model, tokenizer):
        """
        Quick inference test
        """
        print("\n🤖 Testing inference...")
        
        try:
            device = next(model.parameters()).device
            prompt = "Hello, how are you?"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print(f"Prompt: {prompt}")
            print("Generating response (this may take a while on CPU)...")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50)
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response[:100]}...")
            print("✅ Inference test successful")
            
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            print(f"❌ Error: {e}")
    
    def run_complete_pipeline(self, test_inference: bool = True):
        """
        Run the complete pipeline
        """
        print("\n🚀 Starting Mistral Quantization Pipeline...")
        print("="*60)
        
        try:
            model, tokenizer = self.download_and_quantize()
            self.save_quantized_model(model, tokenizer)
            
            if test_inference:
                self.test_inference(model, tokenizer)
            
            print("\n" + "="*60)
            print("✅ Pipeline completed successfully!")
            print(f"\n💾 Model saved to: {self.output_dir}")
            
            if self.device == 'cpu':
                print("\n⚠️  Running on CPU mode. For actual 4-bit quantization:")
                print("   1. Use Google Colab (Free GPU)")
                print("   2. Use: colab_full_pipeline.ipynb")
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"\n❌ Error: {e}")
            raise

def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mistral-7B Quantization (CPU-Compatible)"
    )
    
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--output', type=str, default='models/mistral-7b-4bit')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps', 'auto'],
                       default='auto', help='Device to use')
    parser.add_argument('--no-test', action='store_true')
    
    args = parser.parse_args()
    
    device = args.device if args.device != 'auto' else None
    
    quantizer = MistralQuantizer(
        model_id=args.model,
        output_dir=args.output,
        device=device
    )
    
    quantizer.run_complete_pipeline(test_inference=not args.no_test)

if __name__ == '__main__':
    main()
