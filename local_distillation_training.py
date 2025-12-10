#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local Distillation Training
Optimized for local GPU training

Usage:
    python local_distillation_training.py --samples 1000 --epochs 5 --batch-size 8
"""

import torch
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

logger = setup_logging('logs/local_distillation.log')

class LocalDistillationTrainer:
    """
    Local distillation trainer
    """
    
    def __init__(self,
                 teacher_model_id: str = "mistralai/Mistral-7B-v0.1",
                 student_model_id: str = "microsoft/phi-2",
                 quantized_teacher_path: str = None,
                 output_dir: str = "models/phi-2-distilled"):
        
        self.teacher_model_id = teacher_model_id
        self.student_model_id = student_model_id
        self.quantized_teacher_path = quantized_teacher_path
        self.output_dir = Path(output_dir)
        self.device = 'cuda'
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {self.gpu_name} ({self.gpu_memory:.2f}GB)")
        
        self.memory_monitor = MemoryMonitor(self.device)
    
    def load_teacher_model(self):
        print("Loading Teacher Model...")
        
        if self.quantized_teacher_path and Path(self.quantized_teacher_path).exists():
            print(f"Using pre-quantized: {self.quantized_teacher_path}")
            
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            teacher_tokenizer = AutoTokenizer.from_pretrained(self.quantized_teacher_path)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                self.quantized_teacher_path,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_id)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_id,
                quantization_config=quant_config,
                device_map="auto"
            )
        
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        print("Teacher model loaded")
        return teacher_model, teacher_tokenizer
    
    def load_student_model(self):
        print("Loading Student Model...")
        
        student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_id)
        student_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
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
        
        student_tokenizer.pad_token = student_tokenizer.eos_token
        print("Student model loaded")
        return student_model, student_tokenizer
    
    def prepare_dataset(self, dataset_name: str, num_samples: int, tokenizer):
        print(f"Loading dataset: {dataset_name}...")
        
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
                max_length=512,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            batch_size=32,
            num_proc=4
        )
        
        print(f"Dataset prepared: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(self, num_samples: int = 1000, num_epochs: int = 5, 
              batch_size: int = 8, learning_rate: float = 1e-4):
        
        print("\n" + "="*70)
        print("LOCAL KNOWLEDGE DISTILLATION TRAINING")
        print("="*70)
        
        with Timer("Complete Training") as timer:
            teacher_model, teacher_tokenizer = self.load_teacher_model()
            student_model, student_tokenizer = self.load_student_model()
            
            train_dataset = self.prepare_dataset(
                "databricks/databricks-dolly-15k",
                num_samples,
                student_tokenizer
            )
            
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,
                learning_rate=learning_rate,
                warmup_steps=100,
                weight_decay=0.01,
                save_strategy="steps",
                save_steps=max(100, len(train_dataset) // (batch_size * 4)),
                logging_steps=10,
                fp16=True,
                optim="paged_adamw_32bit",
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=True,
                dataloader_num_workers=4
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
            
            print(f"\nTraining with {num_epochs} epochs, batch_size={batch_size}")
            trainer.train()
            
            print("Saving model...")
            student_model.save_pretrained(str(self.output_dir))
            student_tokenizer.save_pretrained(str(self.output_dir))
            
            metadata = {
                'teacher_model': self.teacher_model_id,
                'student_model': self.student_model_id,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'training_time': timer.elapsed,
                'status': 'completed'
            }
            
            with open(self.output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nTraining completed in {timer.elapsed/3600:.2f}h")
            print(f"Model saved to: {self.output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--student', default='microsoft/phi-2')
    parser.add_argument('--quantized-teacher', default=None)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', default='models/phi-2-distilled')
    
    args = parser.parse_args()
    
    trainer = LocalDistillationTrainer(
        teacher_model_id=args.teacher,
        student_model_id=args.student,
        quantized_teacher_path=args.quantized_teacher,
        output_dir=args.output
    )
    
    trainer.train(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == '__main__':
    main()
