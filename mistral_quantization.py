#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mistral-7B 4-bit Quantization with BitsAndBytes
Download, quantize, and save the model locally
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import Timer, MemoryMonitor, setup_logging

logger = setup_logging('logs/quantization.log')

class MistralQuantizer:
    """
    Mistral-7B quantization manager
    """
    
    def __init__(self, 
                 model_id: str = "mistralai/Mistral-7B-v0.1",
                 output_dir: str = "models/mistral-7b-4bit",
                 device: str = "cuda"):
        """
        Initialize quantizer
        
        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory for quantized model
            device: Device to use (cuda, cpu, mps)
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.device = device
        self.memory_monitor = MemoryMonitor(device)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Mistral Quantizer")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Device: {device}")
    
    def load_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes 4-bit quantization configuration
        """
        print("\n🔧 Creating BitsAndBytes 4-bit Configuration...")
        
        config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,       # Double quantization (nested)
            bnb_4bit_quant_type="nf4",            # Use NF4 (Normal Float 4) quantization
            bnb_4bit_compute_dtype=torch.float16  # Use float16 for computation
        )
        
        logger.info("BitsAndBytes Configuration:")
        logger.info(f"  load_in_4bit: {config.load_in_4bit}")
        logger.info(f"  bnb_4bit_use_double_quant: {config.bnb_4bit_use_double_quant}")
        logger.info(f"  bnb_4bit_quant_type: {config.bnb_4bit_quant_type}")
        logger.info(f"  bnb_4bit_compute_dtype: {config.bnb_4bit_compute_dtype}")
        
        print("✅ BitsAndBytes Configuration Created")
        return config
    
    def download_and_quantize(self) -> tuple:
        """
        Download Mistral model and apply 4-bit quantization
        
        Returns:
            (model, tokenizer) tuple
        """
        self.memory_monitor.reset()
        
        with Timer(f"Downloading and Quantizing {self.model_id}") as timer:
            # Step 1: Get quantization config
            quant_config = self.load_quantization_config()
            
            # Step 2: Load tokenizer
            print("\n💫 Downloading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            print("✅ Tokenizer downloaded")
            logger.info(f"Tokenizer loaded successfully")
            
            # Step 3: Load quantized model
            print(f"\n🔄 Loading {self.model_id} with 4-bit quantization...")
            print("   This may take several minutes for the first download...")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            print("✅ Model loaded with 4-bit quantization")
            logger.info(f"Model loaded successfully with 4-bit quantization")
            
            # Log memory usage
            self.memory_monitor.log_memory("After Quantization")
            
            return model, tokenizer
    
    def save_quantized_model(self, model, tokenizer):
        """
        Save quantized model and tokenizer
        """
        print(f"\n💾 Saving quantized model to {self.output_dir}...")
        
        try:
            # Save model
            model.save_pretrained(
                str(self.output_dir),
                safe_serialization=True
            )
            logger.info(f"Model saved to {self.output_dir}")
            
            # Save tokenizer
            tokenizer.save_pretrained(str(self.output_dir))
            logger.info(f"Tokenizer saved to {self.output_dir}")
            
            # Save metadata
            metadata = {
                'model_id': self.model_id,
                'quantization_method': '4-bit-bnb',
                'quantization_type': 'nf4',
                'device': str(self.device),
                'quantization_config': {
                    'load_in_4bit': True,
                    'bnb_4bit_use_double_quant': True,
                    'bnb_4bit_quant_type': 'nf4',
                    'bnb_4bit_compute_dtype': 'float16'
                }
            }
            
            metadata_path = self.output_dir / 'quantization_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to {metadata_path}")
            
            print("✅ Model, tokenizer, and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            print(f"❌ Failed to save model: {e}")
            raise
    
    def test_inference(self, model, tokenizer, test_prompts: list = None):
        """
        Test the quantized model with simple inference
        
        Args:
            model: The quantized model
            tokenizer: The tokenizer
            test_prompts: List of test prompts (optional)
        """
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "Explain quantum computing:",
                "What is artificial intelligence?"
            ]
        
        print("\n🤖 Testing Inference with Quantized Model...")
        print("=" * 60)
        
        device = next(model.parameters()).device
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            print("-" * 60)
            
            try:
                # Tokenize input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(device)
                
                # Generate output
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Decode output
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Response: {response[:200]}...")
                
                # Log token count
                token_count = outputs.shape[1]
                logger.info(f"Test {i} - Generated {token_count} tokens")
                
            except Exception as e:
                logger.error(f"Inference test {i} failed: {e}")
                print(f"❌ Error: {e}")
        
        print("\n" + "="*60)
        print("✅ Inference testing completed")
    
    def run_complete_pipeline(self, test_inference: bool = True):
        """
        Run the complete quantization pipeline
        
        Args:
            test_inference: Whether to test inference after quantization
        """
        try:
            print("\n🚀 Starting Mistral-7B 4-bit Quantization Pipeline...")
            print("=" * 60)
            
            # Download and quantize
            model, tokenizer = self.download_and_quantize()
            
            # Save model
            self.save_quantized_model(model, tokenizer)
            
            # Test inference
            if test_inference:
                self.test_inference(model, tokenizer)
            
            print("\n" + "="*60)
            print("✅ Quantization pipeline completed successfully!")
            print(f"\n💾 Model saved to: {self.output_dir}")
            print(f"\n🚀 You can now use the quantized model with:")
            print(f"   from mistral_inference import load_quantized_model")
            print(f"   model, tokenizer = load_quantized_model('{self.output_dir}')")
            
            logger.info("Quantization pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Quantization pipeline failed: {e}")
            print(f"\n❌ Error: {e}")
            raise

def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mistral-7B 4-bit Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default quantization
  python mistral_quantization.py
  
  # Custom output directory
  python mistral_quantization.py --output models/mistral-custom
  
  # Skip inference testing
  python mistral_quantization.py --no-test
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='HuggingFace model ID (default: mistralai/Mistral-7B-v0.1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/mistral-7b-4bit',
        help='Output directory (default: models/mistral-7b-4bit)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip inference testing'
    )
    
    args = parser.parse_args()
    
    # Create quantizer and run pipeline
    quantizer = MistralQuantizer(
        model_id=args.model,
        output_dir=args.output,
        device=args.device
    )
    
    quantizer.run_complete_pipeline(test_inference=not args.no_test)

if __name__ == '__main__':
    main()
