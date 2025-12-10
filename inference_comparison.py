#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference Comparison for Multiple Models
Compare outputs from FP16, 4-bit quantized, and distilled models
"""

import torch
import time
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

from mistral_fp16 import MistralFP16Loader, load_quantized_model
from utils import Timer, MemoryMonitor, setup_logging

logger = setup_logging('logs/inference.log')

class InferenceComparator:
    """
    Compare inference across multiple models
    """
    
    def __init__(self):
        """
        Initialize comparator
        """
        self.models = {}
        self.tokenizers = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Inference Comparator initialized with device: {self.device}")
    
    def load_models(self,
                   fp16_model_id: str = "mistralai/Mistral-7B-v0.1",
                   quantized_path: str = "models/mistral-7b-4bit",
                   distilled_path: str = "models/phi-2-distilled"):
        """
        Load all models for comparison
        
        Args:
            fp16_model_id: HuggingFace ID for FP16 model
            quantized_path: Path to quantized model
            distilled_path: Path to distilled model
        """
        print("\n🔄 Loading models for comparison...\n")
        
        # Load FP16 Mistral
        try:
            print("🔄 Loading Mistral FP16...")
            fp16_loader = MistralFP16Loader(model_id=fp16_model_id)
            self.models['fp16'], self.tokenizers['fp16'] = fp16_loader.load_model()
            print("✅ Mistral FP16 loaded")
        except Exception as e:
            logger.error(f"Failed to load FP16 model: {e}")
            print(f"❌ Failed to load FP16 model: {e}")
        
        # Load Quantized Mistral
        try:
            print("🔄 Loading Mistral 4-bit...")
            if Path(quantized_path).exists():
                self.models['4bit'], self.tokenizers['4bit'] = load_quantized_model(quantized_path)
                print("✅ Mistral 4-bit loaded")
            else:
                print(f"⚠️  Quantized model not found at {quantized_path}")
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            print(f"❌ Failed to load quantized model: {e}")
        
        # Load Distilled Phi-2
        try:
            print("🔄 Loading Distilled Phi-2...")
            if Path(distilled_path).exists():
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizers['distilled'] = AutoTokenizer.from_pretrained(distilled_path)
                self.models['distilled'] = AutoModelForCausalLM.from_pretrained(
                    distilled_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("✅ Distilled Phi-2 loaded")
            else:
                print(f"⚠️  Distilled model not found at {distilled_path}")
        except Exception as e:
            logger.error(f"Failed to load distilled model: {e}")
            print(f"❌ Failed to load distilled model: {e}")
        
        print(f"\n✅ Loaded {len(self.models)} model(s)\n")
    
    def run_inference(self, prompt: str, max_length: int = 100) -> Dict[str, Tuple[str, float, float]]:
        """
        Run inference with all loaded models
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
        
        Returns:
            Dictionary with model outputs and metrics
        """
        results = {}
        
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}\n")
        
        for model_name, model in self.models.items():
            print(f"\n🤖 Running inference with {model_name}...")
            
            try:
                tokenizer = self.tokenizers[model_name]
                device = next(model.parameters()).device
                
                # Prepare input
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(device)
                
                # Measure memory before
                memory_monitor = MemoryMonitor(self.device)
                memory_before = memory_monitor.get_current_memory()
                
                # Inference
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                inference_time = time.time() - start_time
                
                # Measure memory after
                memory_after = memory_monitor.get_peak_memory()
                memory_used = memory_after - memory_before
                
                # Decode output
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                tokens_generated = outputs.shape[1]
                tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
                
                results[model_name] = {
                    'output': output_text,
                    'time': inference_time,
                    'tokens_per_sec': tokens_per_sec,
                    'memory_gb': memory_used,
                    'token_count': tokens_generated
                }
                
                print(f"  ⏱️  Time: {inference_time:.3f}s")
                print(f"  💾 Memory: {memory_used:.2f}GB")
                print(f"  💫 Tokens/s: {tokens_per_sec:.2f}")
                print(f"  🌙 Output: {output_text[:100]}...")
                
                logger.info(f"{model_name} - Time: {inference_time:.3f}s, Tokens/s: {tokens_per_sec:.2f}")
                
            except Exception as e:
                logger.error(f"Inference error for {model_name}: {e}")
                print(f"  ❌ Error: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def print_comparison_table(self, results: Dict):
        """
        Print comparison table
        
        Args:
            results: Dictionary of inference results
        """
        print(f"\n{'='*80}")
        print("INFERENCE COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"{'Model':<20} | {'Time (s)':<12} | {'Tokens/s':<12} | {'Memory (GB)':<12}")
        print(f"{'-'*80}")
        
        for model_name, result in results.items():
            if 'error' not in result:
                print(
                    f"{model_name:<20} | "
                    f"{result['time']:<12.3f} | "
                    f"{result['tokens_per_sec']:<12.2f} | "
                    f"{result['memory_gb']:<12.2f}"
                )
            else:
                print(f"{model_name:<20} | ERROR: {result['error']}")
        
        print(f"{'='*80}\n")

def interactive_mode():
    """
    Run interactive comparison mode
    """
    comparator = InferenceComparator()
    
    # Load models
    comparator.load_models()
    
    print("\n🚀 Interactive Inference Comparison Mode")
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("💬 Enter your prompt: ").strip()
        
        if prompt.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not prompt:
            print("⚠️  Please enter a valid prompt\n")
            continue
        
        results = comparator.run_inference(prompt)
        comparator.print_comparison_table(results)

def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inference Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--prompt', type=str, help='Single prompt to compare')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--fp16-model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--quantized', type=str, default='models/mistral-7b-4bit')
    parser.add_argument('--distilled', type=str, default='models/phi-2-distilled')
    
    args = parser.parse_args()
    
    comparator = InferenceComparator()
    comparator.load_models(
        fp16_model_id=args.fp16_model,
        quantized_path=args.quantized,
        distilled_path=args.distilled
    )
    
    if args.prompt:
        results = comparator.run_inference(args.prompt)
        comparator.print_comparison_table(results)
    elif args.interactive:
        interactive_mode()
    else:
        # Default prompt
        default_prompt = "What is artificial intelligence?"
        results = comparator.run_inference(default_prompt)
        comparator.print_comparison_table(results)

if __name__ == '__main__':
    main()
