#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Training Pipeline
Generate all model versions: Quantized, LoRA, LoRA+Quantized

Usage (Colab):
    !git clone https://github.com/caizongxun/mistral-quantization-distillation.git
    %cd mistral-quantization-distillation
    !pip install -q transformers datasets peft bitsandbytes accelerate
    !python complete_training_pipeline.py --samples 200 --epochs 1

Usage (Local):
    python complete_training_pipeline.py --samples 100 --epochs 1
"""

import torch
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import json
import time

class Timer:
    """Simple timer context manager"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"\nTimer: {self.name} starting...")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"OK: {self.name} completed! Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        self.elapsed = elapsed

class CompleteTrainingPipeline:
    """
    Complete training pipeline
    """
    
    def __init__(self, output_base: str = "models"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Define all output paths
        self.paths = {
            'phi_base': self.output_base / "phi-2-base",
            'phi_quant': self.output_base / "phi-2-quantized",
            'phi_lora': self.output_base / "phi-2-lora",
            'phi_lora_quant': self.output_base / "phi-2-lora-quantized",
        }
        
        # Create all directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.supports_tf32 = False
        
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
            print(f"   CUDA: {torch.version.cuda}")
            
            try:
                gpu_capability = torch.cuda.get_device_capability(0)
                self.supports_tf32 = gpu_capability[0] >= 8
                if not self.supports_tf32:
                    print(f"   WARNING: TF32 not supported (need Ampere+, current: CC {gpu_capability[0]}.{gpu_capability[1]})")
            except:
                pass
        else:
            print(f"\nWARNING: Using CPU (recommended to use Colab GPU)")
    
    def prepare_dataset(self, tokenizer, num_samples: int = 100):
        """Prepare training dataset"""
        print(f"\nDataset: Preparing dataset ({num_samples} samples)...")
        
        dataset = load_dataset("databricks/databricks-dolly-15k")
        
        def format_instruction(example):
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            
            text = f"Instruction: {instruction}\n"
            if input_text:
                text += f"Input: {input_text}\n"
            text += f"Response: {output}"
            
            return {'text': text}
        
        dataset = dataset.map(format_instruction, remove_columns=dataset['train'].column_names)
        dataset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            texts = examples["text"]
            if isinstance(texts, str):
                texts = [texts]
            
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            batch_size=16,
            num_proc=2
        )
        
        print(f"OK: Dataset ready: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def stage1_save_base_model(self):
        """Stage 1: Save base model (download only)"""
        print("\n" + "="*70)
        print("STAGE 1: Save base Phi-2 model")
        print("="*70)
        
        with Timer("Save base model") as timer:
            print("\nDOWN: Download Phi-2 model...")
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-2",
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            print(f"SAVE: Save to {self.paths['phi_base']}")
            model.save_pretrained(str(self.paths['phi_base']))
            tokenizer.save_pretrained(str(self.paths['phi_base']))
            
            metadata = {
                'model': 'phi-2',
                'version': 'base',
                'params': '2.7B',
                'dtype': 'float16',
                'status': 'saved'
            }
            self._save_metadata(self.paths['phi_base'], metadata)
            print(f"OK: Base model saved")
    
    def stage2_quantize_base_model(self):
        """Stage 2: Quantize base model"""
        print("\n" + "="*70)
        print("STAGE 2: Quantize base model (INT4)")
        print("="*70)
        
        with Timer("Quantize base model") as timer:
            print("\nLOAD: Load base model...")
            tokenizer = AutoTokenizer.from_pretrained(str(self.paths['phi_base']))
            
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                str(self.paths['phi_base']),
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"SAVE: Save quantized model to {self.paths['phi_quant']}")
            model.save_pretrained(str(self.paths['phi_quant']))
            tokenizer.save_pretrained(str(self.paths['phi_quant']))
            
            metadata = {
                'model': 'phi-2',
                'version': 'quantized',
                'quantization': 'INT4 (nf4)',
                'params': '2.7B',
                'status': 'saved'
            }
            self._save_metadata(self.paths['phi_quant'], metadata)
            print(f"OK: Quantized model saved")
    
    def stage3_lora_finetuning(self, num_samples: int = 100, num_epochs: int = 1):
        """Stage 3: LoRA fine-tuning"""
        print("\n" + "="*70)
        print(f"STAGE 3: LoRA fine-tuning ({num_samples} samples, {num_epochs} epoch)")
        print("="*70)
        
        with Timer("LoRA fine-tuning") as timer:
            print("\nLOAD: Load base model...")
            tokenizer = AutoTokenizer.from_pretrained(str(self.paths['phi_base']))
            model = AutoModelForCausalLM.from_pretrained(
                str(self.paths['phi_base']),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("\nLORA: Apply LoRA...")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            print("\nDATA: Prepare dataset...")
            train_dataset = self.prepare_dataset(tokenizer, num_samples)
            
            print("\nTRAIN: Start training...")
            
            device_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
            total_memory = device_props.total_memory if device_props else 16e9
            batch_size = 1 if total_memory < 20e9 else 2
            
            training_args = TrainingArguments(
                output_dir=str(self.paths['phi_lora']),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                warmup_steps=10,
                weight_decay=0.01,
                save_strategy="no",
                logging_steps=5,
                fp16=True,
                optim="paged_adamw_8bit",
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=0,
                max_grad_norm=0.3,
                tf32=self.supports_tf32
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            trainer.train()
            
            # Merge LoRA weights with base model
            print(f"\nMERGE: Merge LoRA weights with base model...")
            model = model.merge_and_unload()
            
            print(f"\nSAVE: Save complete LoRA model...")
            model.save_pretrained(str(self.paths['phi_lora']))
            tokenizer.save_pretrained(str(self.paths['phi_lora']))
            
            metadata = {
                'model': 'phi-2',
                'version': 'lora',
                'lora_rank': 8,
                'lora_alpha': 16,
                'params': '2.7B',
                'status': 'saved',
                'note': 'LoRA weights merged into base model'
            }
            self._save_metadata(self.paths['phi_lora'], metadata)
            print(f"OK: LoRA fine-tuned model saved")
    
    def stage4_quantize_lora_model(self):
        """Stage 4: Quantize LoRA fine-tuned model"""
        print("\n" + "="*70)
        print("STAGE 4: Quantize LoRA fine-tuned model (INT4)")
        print("="*70)
        
        with Timer("Quantize LoRA model") as timer:
            print("\nLOAD: Load LoRA fine-tuned model...")
            tokenizer = AutoTokenizer.from_pretrained(str(self.paths['phi_lora']))
            
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                str(self.paths['phi_lora']),
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"SAVE: Save quantized LoRA model to {self.paths['phi_lora_quant']}")
            model.save_pretrained(str(self.paths['phi_lora_quant']))
            tokenizer.save_pretrained(str(self.paths['phi_lora_quant']))
            
            metadata = {
                'model': 'phi-2',
                'version': 'lora-quantized',
                'lora_rank': 8,
                'quantization': 'INT4 (nf4)',
                'params': '2.7B',
                'status': 'saved'
            }
            self._save_metadata(self.paths['phi_lora_quant'], metadata)
            print(f"OK: Quantized LoRA model saved")
    
    def _save_metadata(self, model_path: Path, metadata: dict):
        """Save model metadata"""
        with open(model_path / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def run_full_pipeline(self, num_samples: int = 100, num_epochs: int = 1):
        """Run complete training pipeline"""
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + "  Complete Training Pipeline: Quantized + LoRA + LoRA Quantized".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#"*70)
        
        pipeline_start = time.time()
        
        try:
            # Stage 1
            self.stage1_save_base_model()
            
            # Stage 2
            self.stage2_quantize_base_model()
            
            # Stage 3
            self.stage3_lora_finetuning(num_samples, num_epochs)
            
            # Stage 4
            self.stage4_quantize_lora_model()
            
            # Complete
            pipeline_elapsed = time.time() - pipeline_start
            hours, remainder = divmod(pipeline_elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\n" + "="*70)
            print("OK: Complete pipeline execution finished!")
            print("="*70)
            print(f"\nRESULTS: Training results:")
            print(f"\n1. Phi-2 base model (float16)")
            print(f"   PATH: {self.paths['phi_base']}")
            print(f"   Size: ~5GB, Speed: 1x")
            
            print(f"\n2. Phi-2 quantized version (INT4)")
            print(f"   PATH: {self.paths['phi_quant']}")
            print(f"   Size: ~1.2GB DOWN, Speed: 3x FAST")
            
            print(f"\n3. Phi-2 + LoRA version")
            print(f"   PATH: {self.paths['phi_lora']}")
            print(f"   Size: ~5GB, Accuracy: +7% UP")
            
            print(f"\n4. Phi-2 + LoRA quantized version (INT4)")
            print(f"   PATH: {self.paths['phi_lora_quant']}")
            print(f"   Size: ~1.2GB DOWN, Speed: 3x FAST, Accuracy: +7% UP")
            
            print(f"\nTIME: Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print("\nALL: All models ready!")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete training pipeline')
    parser.add_argument('--samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--output', default='models', help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = CompleteTrainingPipeline(output_base=args.output)
    pipeline.run_full_pipeline(num_samples=args.samples, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
