#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mistral-7B 4-bit Quantization - Colab Optimized Version
With timeout handling and memory optimization for Colab environment
"""

import os
import json
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import Timer, MemoryMonitor, setup_logging

logger = setup_logging('logs/quantization.log')

class MistralQuantizerColab:
    """
    Mistral-7B quantization optimized for Google Colab
    """
    
    def __init__(self, 
                 model_id: str = "mistralai/Mistral-7B-v0.1",
                 output_dir: str = "models/mistral-7b-4bit"):
        """
        Initialize quantizer for Colab
        
        Args:
            model_id: HuggingFace model ID
            output_dir: Output directory for quantized model
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.device = 'cuda'  # Colab always has CUDA
        self.memory_monitor = MemoryMonitor(self.device)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Colab Quantizer")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Output Directory: {output_dir}")
    
    def load_quantization_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes 4-bit quantization configuration
        """
        print("\n🔧 Creating BitsAndBytes 4-bit Configuration...")
        
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
        Optimized for Colab with memory management
        
        Returns:
            (model, tokenizer) tuple
        """
        self.memory_monitor.reset()
        
        with Timer(f"Downloading and Quantizing {self.model_id}") as timer:
            # Step 1: Get quantization config
            quant_config = self.load_quantization_config()
            
            # Step 2: Load tokenizer
            print("\n📝 Downloading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token  # Colab fix
            print("✅ Tokenizer downloaded")
            logger.info(f"Tokenizer loaded successfully")
            
            # Step 3: Load quantized model
            print(f"\n🔄 Loading {self.model_id} with 4-bit quantization...")
            print("   (This takes 2-3 minutes, be patient)\n")
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True  # Colab optimization
            )
            
            print("\n✅ Model loaded with 4-bit quantization")
            logger.info(f"Model loaded successfully with 4-bit quantization")
            
            # Log memory usage
            self.memory_monitor.log_memory("After Quantization")
            
            return model, tokenizer
    
    def save_quantized_model(self, model, tokenizer):
        """
        Save quantized model and tokenizer with Colab optimization
        Includes progress indication and timeout handling
        """
        print(f"\n💾 Saving quantized model to {self.output_dir}...")
        print("   (This may take 2-3 minutes, do not interrupt)")
        
        try:
            # Clear GPU cache before saving
            torch.cuda.empty_cache()
            
            start_time = time.time()
            
            # Save model with progress
            print("\n   📥 Saving model weights...")
            model.save_pretrained(
                str(self.output_dir),
                safe_serialization=True,
                max_shard_size="5GB"  # Colab friendly shard size
            )
            save_time = time.time() - start_time
            print(f"   ✅ Model weights saved ({save_time:.1f}s)")
            logger.info(f"Model saved to {self.output_dir}")
            
            # Save tokenizer
            print("\n   🔤 Saving tokenizer...")
            tokenizer.save_pretrained(str(self.output_dir))
            print("   ✅ Tokenizer saved")
            logger.info(f"Tokenizer saved to {self.output_dir}")
            
            # Save metadata
            print("\n   📋 Saving metadata...")
            metadata = {
                'model_id': self.model_id,
                'quantization_method': '4-bit-bnb',
                'quantization_type': 'nf4',
                'device': 'cuda',
                'environment': 'colab',
                'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'notes': 'Quantized model ready for inference and fine-tuning'
            }
            
            metadata_path = self.output_dir / 'quantization_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print("   ✅ Metadata saved")
            logger.info(f"Metadata saved to {metadata_path}")
            
            print("\n✅ Model saved successfully!")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            print(f"\n❌ Error during save: {e}")
            print(f"\n💡 Troubleshooting:")
            print(f"   - Model weights may still be in {self.output_dir}/")
            print(f"   - Try checking: !ls -lah {self.output_dir}/")
            print(f"   - Storage may be limited: !df -h")
            raise
    
    def test_inference(self, model, tokenizer, test_prompts: list = None):
        """
        Quick inference test
        """
        if test_prompts is None:
            test_prompts = ["Hello, how are you?"]
        
        print("\n🤖 Testing inference...")
        print("="*60)
        
        device = next(model.parameters()).device
        
        for i, prompt in enumerate(test_prompts[:1], 1):  # Only test first one for speed
            print(f"\nTest {i}: {prompt}")
            print("-"*60)
            
            try:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Response: {response[:150]}...")
                
                token_count = outputs.shape[1]
                logger.info(f"Test {i} - Generated {token_count} tokens")
                
            except Exception as e:
                logger.error(f"Inference test {i} failed: {e}")
                print(f"❌ Error: {e}")
        
        print("\n" + "="*60)
        print("✅ Inference testing completed")
    
    def run_complete_pipeline(self, test_inference: bool = True):
        """
        Run the complete Colab-optimized pipeline
        
        Args:
            test_inference: Whether to test inference
        """
        try:
            print("\n" + "="*70)
            print("🚀 MISTRAL-7B 4-BIT QUANTIZATION PIPELINE (Colab Optimized)")
            print("="*70)
            
            # Download and quantize
            model, tokenizer = self.download_and_quantize()
            
            # Save model
            self.save_quantized_model(model, tokenizer)
            
            # Test inference (optional)
            if test_inference:
                self.test_inference(model, tokenizer)
            
            # Final summary
            print("\n" + "="*70)
            print("✅ QUANTIZATION PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            print(f"\n💾 Quantized Model Location:")
            print(f"   {self.output_dir}/")
            
            print(f"\n📊 Next Steps in Colab:")
            print(f"   1. Benchmark: !python benchmark.py")
            print(f"   2. Distillation: !python distillation_training.py --samples 500")
            print(f"   3. Inference: !python inference_comparison.py")
            print(f"   4. Demo: !python app.py")
            
            logger.info("Quantization pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Quantization pipeline failed: {e}")
            print(f"\n❌ Pipeline Error: {e}")
            raise

def main():
    """
    Main entry point for Colab
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Mistral-7B 4-bit Quantization (Colab)"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='HuggingFace model ID'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/mistral-7b-4bit',
        help='Output directory'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip inference testing'
    )
    
    args = parser.parse_args()
    
    # Create quantizer and run pipeline
    quantizer = MistralQuantizerColab(
        model_id=args.model,
        output_dir=args.output
    )
    
    quantizer.run_complete_pipeline(test_inference=not args.no_test)

if __name__ == '__main__':
    main()
