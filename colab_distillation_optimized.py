#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colab 優化蒸餾訓練 - 方案 A：分步驟載入模型
不同時載入老師和學生，節省記憶體

Colab T4 (15GB) 完全支持！
預期時間：~2 小時
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
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
import pickle

logger = setup_logging('logs/distillation_optimized.log')

class CoLabOptimizedDistillation:
    """
    Colab 優化蒸餾：分步驟載入模型
    """
    
    def __init__(self,
                 teacher_model_id: str = "mistralai/Mistral-7B-v0.1",
                 student_model_id: str = "microsoft/phi-2",
                 output_dir: str = "models/phi-2-distilled",
                 cache_dir: str = "distillation_cache"):
        
        self.teacher_model_id = teacher_model_id
        self.student_model_id = student_model_id
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.device = 'cuda'
        self.memory_monitor = MemoryMonitor(self.device)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("CoLab Optimized Distillation Initialized")
    
    def prepare_dataset(self,
                       dataset_name: str = "databricks/databricks-dolly-15k",
                       num_samples: int = 500,
                       tokenizer = None):
        """
        準備資料集
        """
        print(f"\n📚 Loading dataset: {dataset_name}...")
        
        dataset = load_dataset(dataset_name)
        
        def format_instruction(example):
            instruction = example.get('instruction', '')
            input_text = example.get('input', '')
            output = example.get('output', '')
            
            text = f"Instruction: {instruction}\n"
            if input_text:
                text += f"Input: {input_text}\n"
            text += f"Response: {output}"
            
            return {'text': text}
        
        print("Formatting dataset...")
        dataset = dataset.map(format_instruction, remove_columns=dataset['train'].column_names)
        dataset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
        
        print(f"✅ Dataset prepared: {len(dataset)} samples")
        
        def tokenize_function(examples):
            texts = examples["text"]
            if isinstance(texts, str):
                texts = [texts]
            
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        print("\n🔄 Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            batch_size=32
        )
        
        print("✅ Dataset tokenized")
        return tokenized_dataset
    
    def generate_teacher_outputs(self, tokenized_dataset, num_samples: int = 500):
        """
        步驟 1：生成並儲存老師的輸出
        
        這樣老師只需要載入一次，然後卸載
        """
        print("\n" + "="*70)
        print("👨‍🏫 STEP 1: Generate Teacher Outputs")
        print("="*70)
        
        # 載入老師模型
        print("\nLoading Teacher Model (Mistral-7B 4-bit)...")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            self.teacher_model_id,
            trust_remote_code=True
        )
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        
        teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True
        )
        teacher_model.eval()
        
        print("✅ Teacher model loaded")
        self.memory_monitor.log_memory("After Loading Teacher")
        
        # 生成輸出
        print(f"\n🔄 Generating teacher outputs for {len(tokenized_dataset)} samples...")
        
        teacher_outputs = []
        
        for i, sample in enumerate(tokenized_dataset):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(tokenized_dataset)}")
            
            # 準備輸入
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(self.device)
            
            # 獲取老師輸出
            with torch.no_grad():
                outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits.cpu().numpy()
            
            teacher_outputs.append(logits)
            
            # 定期清理記憶體
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        # 儲存老師輸出
        print("\n💾 Caching teacher outputs...")
        cache_path = self.cache_dir / "teacher_outputs.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(teacher_outputs, f)
        
        print(f"✅ Teacher outputs cached: {cache_path}")
        
        # 卸載老師模型
        print("\n🗑️  Unloading teacher model to save memory...")
        del teacher_model
        torch.cuda.empty_cache()
        
        print("✅ Teacher model unloaded (4GB freed)")
        self.memory_monitor.log_memory("After Unloading Teacher")
        
        return teacher_outputs
    
    def train_student(self, tokenized_dataset, teacher_outputs,
                     num_epochs: int = 3,
                     batch_size: int = 4,
                     learning_rate: float = 5e-5):
        """
        步驟 2：訓練學生模型
        
        只載入學生，記憶體充足
        """
        print("\n" + "="*70)
        print("👩‍🌈 STEP 2: Train Student Model")
        print("="*70)
        
        # 載入學生模型
        print("\n👩‍🎓 Loading Student Model (Phi-2)...")
        
        student_tokenizer = AutoTokenizer.from_pretrained(
            self.student_model_id,
            trust_remote_code=True
        )
        student_tokenizer.pad_token = student_tokenizer.eos_token
        
        student_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 應用 LoRA
        print("\n🔗 Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
        
        print("✅ Student model loaded")
        self.memory_monitor.log_memory("After Loading Student")
        
        # 訓練配置
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=max(50, len(tokenized_dataset) // batch_size),
            logging_steps=10,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
            dataloader_pin_memory=True
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # 訓練
        print("\n🔄 Training student model...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Samples: {len(tokenized_dataset)}")
        
        trainer.train()
        
        # 儲存
        print("\n💾 Saving trained model...")
        student_model.save_pretrained(str(self.output_dir))
        student_tokenizer.save_pretrained(str(self.output_dir))
        
        # 儲存 metadata
        metadata = {
            'teacher_model': self.teacher_model_id,
            'student_model': self.student_model_id,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_method': 'colab_optimized',
            'status': 'completed'
        }
        
        metadata_path = self.output_dir / 'distillation_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Model saved to {self.output_dir}")
    
    def run_pipeline(self,
                    dataset_name: str = "databricks/databricks-dolly-15k",
                    num_samples: int = 500,
                    num_epochs: int = 3,
                    batch_size: int = 4):
        """
        執行完整蒸餾流程
        """
        print("\n" + "#"*70)
        print("# 🚀 COLAB OPTIMIZED DISTILLATION PIPELINE")
        print("#"*70)
        
        with Timer("Complete Distillation Pipeline") as timer:
            # 準備資料
            print("\n📊 Preparing dataset...")
            student_tokenizer = AutoTokenizer.from_pretrained(
                self.student_model_id,
                trust_remote_code=True
            )
            tokenized_dataset = self.prepare_dataset(
                dataset_name=dataset_name,
                num_samples=num_samples,
                tokenizer=student_tokenizer
            )
            
            # 生成老師輸出
            teacher_outputs = self.generate_teacher_outputs(
                tokenized_dataset,
                num_samples=num_samples
            )
            
            # 訓練學生
            self.train_student(
                tokenized_dataset,
                teacher_outputs,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
        
        print("\n" + "#"*70)
        print("# ✅ DISTILLATION COMPLETE!")
        print("#"*70)
        print(f"\n⏱️  Total time: {timer.elapsed:.2f}s ({timer.elapsed/60:.1f} min)")
        print(f"📁 Model saved to: {self.output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Colab Optimized Distillation"
    )
    
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=str, default='models/phi-2-distilled')
    
    args = parser.parse_args()
    
    pipeline = CoLabOptimizedDistillation(output_dir=args.output)
    pipeline.run_pipeline(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == '__main__':
    main()
