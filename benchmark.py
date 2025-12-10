#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Benchmarking for Mistral-7B Models
Compare FP16 vs 4-bit quantization performance
"""

import os
import time
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json

from mistral_fp16 import load_quantized_model
from mistral_fp16 import MistralFP16Loader
from utils import Timer, MemoryMonitor, setup_logging

logger = setup_logging('logs/benchmark.log')

class ModelBenchmark:
    """
    Benchmark different model configurations
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize benchmarker
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initializing Benchmark with device: {self.device}")
    
    def benchmark_model(self,
                       model_name: str,
                       model,
                       tokenizer,
                       test_prompts: List[str],
                       num_runs: int = 3) -> Dict:
        """
        Benchmark a single model
        
        Args:
            model_name: Name of the model
            model: The model to benchmark
            tokenizer: The tokenizer
            test_prompts: List of test prompts
            num_runs: Number of benchmark runs
        
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        memory_monitor = MemoryMonitor(self.device)
        
        # Memory benchmark
        memory_monitor.reset()
        initial_memory = memory_monitor.get_current_memory()
        
        # Speed benchmark
        inference_times = []
        token_counts = []
        
        device = next(model.parameters()).device
        
        for run in range(num_runs):
            print(f"\nRun {run+1}/{num_runs}")
            
            total_time = 0
            total_tokens = 0
            
            for prompt in test_prompts:
                start_time = time.time()
                
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
                    
                    end_time = time.time()
                    inference_time = end_time - start_time
                    token_count = outputs.shape[1]
                    
                    total_time += inference_time
                    total_tokens += token_count
                    
                    print(f"  Prompt: {prompt[:30]}...")
                    print(f"    Time: {inference_time:.3f}s | Tokens: {token_count}")
                    
                except Exception as e:
                    logger.error(f"Benchmark error: {e}")
                    print(f"  Error: {e}")
                    total_time = float('inf')
                    break
            
            if total_time != float('inf'):
                avg_time = total_time / len(test_prompts)
                tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
                
                inference_times.append(avg_time)
                token_counts.append(tokens_per_sec)
                
                print(f"  Avg Time: {avg_time:.3f}s")
                print(f"  Tokens/s: {tokens_per_sec:.2f}")
        
        # Final memory measurement
        final_memory = memory_monitor.get_peak_memory()
        
        # Calculate statistics
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        avg_tokens_per_sec = sum(token_counts) / len(token_counts) if token_counts else 0
        
        result = {
            'Model': model_name,
            'Memory (GB)': round(final_memory, 2),
            'Avg Inference Time (s)': round(avg_inference_time, 3),
            'Tokens/s': round(avg_tokens_per_sec, 2),
            'Timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print(f"Results for {model_name}:")
        print(f"  Memory: {result['Memory (GB)']} GB")
        print(f"  Avg Inference Time: {result['Avg Inference Time (s)']} s")
        print(f"  Tokens/s: {result['Tokens/s']}")
        print(f"{'='*60}")
        
        self.results.append(result)
        logger.info(f"Benchmark for {model_name} completed: {result}")
        
        return result
    
    def save_results(self, filename: str = "benchmark_results.csv"):
        """
        Save benchmark results to CSV
        
        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Benchmark results saved to {output_path}")
        logger.info(f"Benchmark results saved to {output_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return output_path
    
    def run_full_benchmark(self,
                          fp16_model_id: str = "mistralai/Mistral-7B-v0.1",
                          quantized_model_path: str = "models/mistral-7b-4bit"):
        """
        Run full benchmark comparing FP16 and quantized models
        
        Args:
            fp16_model_id: HuggingFace ID for FP16 model
            quantized_model_path: Path to quantized model
        """
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain quantum computing:",
            "How does neural networks work?",
            "What is artificial intelligence?",
            "Tell me about deep learning:"
        ]
        
        print("\n🚀 Starting Comprehensive Benchmark...")
        print(f"Device: {self.device}")
        print(f"Test Prompts: {len(test_prompts)}")
        
        # Benchmark FP16
        try:
            print("\n🔄 Loading FP16 Model...")
            fp16_loader = MistralFP16Loader(model_id=fp16_model_id)
            fp16_model, fp16_tokenizer = fp16_loader.load_model()
            
            self.benchmark_model(
                "Mistral-7B FP16",
                fp16_model,
                fp16_tokenizer,
                test_prompts,
                num_runs=2
            )
            
            # Clean up
            del fp16_model, fp16_tokenizer
            torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"FP16 benchmark failed: {e}")
            print(f"❌ FP16 benchmark failed: {e}")
        
        # Benchmark Quantized
        try:
            print("\n🔄 Loading Quantized Model...")
            if os.path.exists(quantized_model_path):
                quant_model, quant_tokenizer = load_quantized_model(quantized_model_path)
                
                self.benchmark_model(
                    "Mistral-7B 4-bit",
                    quant_model,
                    quant_tokenizer,
                    test_prompts,
                    num_runs=2
                )
                
                # Clean up
                del quant_model, quant_tokenizer
                torch.cuda.empty_cache()
            else:
                print(f"⚠️  Quantized model not found at {quantized_model_path}")
        
        except Exception as e:
            logger.error(f"Quantized benchmark failed: {e}")
            print(f"❌ Quantized benchmark failed: {e}")
        
        # Save results
        if self.results:
            self.save_results()
        else:
            print("❌ No benchmark results to save")

def main():
    """
    Main entry point for benchmarking
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Mistral models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python benchmark.py
  
  # Custom model paths
  python benchmark.py --quantized models/custom-4bit --output results/
        """
    )
    
    parser.add_argument(
        '--fp16-model',
        type=str,
        default='mistralai/Mistral-7B-v0.1',
        help='HuggingFace ID for FP16 model'
    )
    parser.add_argument(
        '--quantized',
        type=str,
        default='models/mistral-7b-4bit',
        help='Path to quantized model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = ModelBenchmark(output_dir=args.output)
    benchmark.run_full_benchmark(
        fp16_model_id=args.fp16_model,
        quantized_model_path=args.quantized
    )

if __name__ == '__main__':
    main()
