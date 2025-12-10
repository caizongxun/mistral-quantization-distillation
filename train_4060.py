#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTX 4060 Optimized Distillation Training
8GB VRAM Memory-Efficient Configuration

Usage:
    python train_4060.py
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
from utils import Timer, MemoryMonitor, setup_logging
import json

logger = setup_logging('logs/training_4060.log')

# 環境優化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class RTX4060Trainer:
    """
    RTX 4060 專用訓練器 (8GB VRAM)
    """
    
    def __init__(self, output_dir: str = "models/phi-2-distilled"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        
        self.memory_monitor = MemoryMonitor('cuda')
    
    def load_teacher_model(self):
        """載入量化的 Mistral"""
        print("\n👨‍🏫 Loading Teacher Model (Mistral-7B 4-bit)...")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            "models/mistral-7b-4bit",
            trust_remote_code=True
        )
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "models/mistral-7b-4bit",
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Teacher model loaded")
        self.memory_monitor.log_memory("After Loading Teacher")
        
        return teacher_model, teacher_tokenizer
    
    def load_student_model(self):
        """載入 Phi-2 並應用 LoRA"""
        print("\n👩‍🎓 Loading Student Model (Phi-2)...")
        
        student_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        student_tokenizer.pad_token = student_tokenizer.eos_token
        
        student_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # 4060 優化
        )
        
        # 應用 LoRA（非常重要！只訓練 5% 的參數）
        print("\n🔗 Applying LoRA...")
        lora_config = LoraConfig(
            r=8,  # 降低秩
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
        
        print("✅ Student model loaded")
        self.memory_monitor.log_memory("After Loading Student")
        
        return student_model, student_tokenizer
    
    def prepare_dataset(self, tokenizer):
        """準備小規模數據集"""
        print("\n📚 Loading dataset...")
        
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
        # 4060: 只用 100 個樣本
        dataset = dataset['train'].select(range(min(100, len(dataset['train']))))
        
        print(f"✅ Dataset prepared: {len(dataset)} samples")
        
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
                max_length=256,  # 降低序列長度
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        print("   Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            batch_size=16
        )
        
        print("✅ Dataset tokenized")
        return tokenized_dataset
    
    def train(self):
        """RTX 4060 優化訓練"""
        print("\n" + "="*70)
        print("🚀 RTX 4060 OPTIMIZED DISTILLATION TRAINING")
        print("="*70)
        
        with Timer("Complete Training") as timer:
            # 載入模型
            teacher_model, teacher_tokenizer = self.load_teacher_model()
            student_model, student_tokenizer = self.load_student_model()
            
            # 準備數據
            train_dataset = self.prepare_dataset(student_tokenizer)
            
            # 訓練配置（4060 極限優化）
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=1,  # 只訓練 1 個 epoch
                per_device_train_batch_size=1,  # 極限小
                gradient_accumulation_steps=4,  # 累積梯度
                learning_rate=5e-5,
                warmup_steps=10,
                weight_decay=0.01,
                save_strategy="no",  # 不保存中間檢查點
                logging_steps=5,
                fp16=True,  # 半精度
                optim="paged_adamw_8bit",  # 8-bit 優化器
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=0,  # 禁用多進程
                max_grad_norm=0.3,  # 梯度裁剪
                tf32=True  # TensorFloat32
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=student_tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=student_model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            print("\n🎓 Training...")
            print(f"   Batch size: 1 (per device)")
            print(f"   Gradient accumulation: 4")
            print(f"   Effective batch size: 4")
            print(f"   Samples: {len(train_dataset)}")
            print(f"   Epochs: 1\n")
            
            trainer.train()
            
            # 保存模型
            print("\n💾 Saving model...")
            student_model.save_pretrained(str(self.output_dir))
            student_tokenizer.save_pretrained(str(self.output_dir))
            
            # 保存元數據
            metadata = {
                'gpu': 'RTX 4060',
                'vram': '8GB',
                'training_method': 'LoRA + 4bit quantization',
                'samples': len(train_dataset),
                'epochs': 1,
                'batch_size': 1,
                'effective_batch_size': 4,
                'training_time': timer.elapsed,
                'status': 'completed'
            }
            
            with open(self.output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n✅ Training completed!")
            print(f"   Time: {timer.elapsed/60:.1f} minutes")
            print(f"   Model saved to: {self.output_dir}")

def main():
    trainer = RTX4060Trainer(output_dir="models/phi-2-distilled")
    trainer.train()

if __name__ == '__main__':
    main()
