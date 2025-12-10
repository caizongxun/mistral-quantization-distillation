#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mistral-7B FP16 Model Loading
For benchmarking and comparison with quantized models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import Timer, MemoryMonitor, setup_logging

logger = setup_logging('logs/mistral_fp16.log')

class MistralFP16Loader:
    """
    Load Mistral-7B in FP16 precision
    """
    
    def __init__(self, 
                 model_id: str = "mistralai/Mistral-7B-v0.1",
                 device: str = "cuda"):
        """
        Initialize FP16 loader
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use (cuda, cpu)
        """
        self.model_id = model_id
        self.device = device
        self.memory_monitor = MemoryMonitor(device)
        
        logger.info(f"Initializing Mistral FP16 Loader")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Device: {device}")
    
    def load_model(self) -> tuple:
        """
        Load Mistral model in FP16 precision
        
        Returns:
            (model, tokenizer) tuple
        """
        self.memory_monitor.reset()
        
        with Timer(f"Loading {self.model_id} in FP16") as timer:
            print(f"\n🔄 Loading {self.model_id} in FP16...")
            print("   This may take several minutes for the first download...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model in FP16
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            print("✅ Model loaded in FP16")
            
            # Log memory usage
            self.memory_monitor.log_memory("After Loading FP16 Model")
            
            return model, tokenizer
    
    def test_inference(self, model, tokenizer, test_prompt: str = "Hello, how are you?"):
        """
        Test inference with FP16 model
        """
        print(f"\n🤖 Testing FP16 Model Inference...")
        print(f"Prompt: {test_prompt}")
        
        device = next(model.parameters()).device
        
        try:
            inputs = tokenizer(
                test_prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response[:200]}...")
            print("✅ Inference successful")
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            print(f"❌ Inference failed: {e}")

def load_quantized_model(model_path: str, device: str = "cuda") -> tuple:
    """
    Load pre-quantized model from local directory
    
    Args:
        model_path: Path to quantized model directory
        device: Device to use
    
    Returns:
        (model, tokenizer) tuple
    """
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    
    print(f"\n🔄 Loading quantized model from {model_path}...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ Quantized model loaded")
    return model, tokenizer

if __name__ == '__main__':
    loader = MistralFP16Loader()
    model, tokenizer = loader.load_model()
    loader.test_inference(model, tokenizer)
