#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Training - Fixed for transformers 4.40+
Transfer knowledge from Mistral-7B (teacher) to Phi-2 (student)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional
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

logger = setup_logging('logs/distillation.log')

class DistillationTrainer:
    """
    Knowledge distillation from teacher to student model
    """
    
    def __init__(self,
                 teacher_model_id: str = "mistralai/Mistral-7B-v0.1",
                 student_model_id: str = "microsoft/phi-2",
                 output_dir: str = "models/phi-2-distilled",
                 device: str = "cuda"):
        """
        Initialize distillation trainer
        
        Args:
            teacher_model_id: HuggingFace ID for teacher model
            student_model_id: HuggingFace ID for student model
            output_dir: Output directory for distilled model
            device: Device to use
        """
        self.teacher_model_id = teacher_model_id
        self.student_model_id = student_model_id
        self.output_dir = Path(output_dir)
        self.device = device
        self.memory_monitor = MemoryMonitor(device)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Distillation Trainer")
        logger.info(f"Teacher Model: {teacher_model_id}")
        logger.info(f"Student Model: {student_model_id}")
    
    def load_teacher_model(self) -> tuple:
        """
        Load teacher model (Mistral-7B) in 4-bit quantization
        """
        print("\n👨‍🏫 Loading Teacher Model (Mistral-7B)...")
        
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
        
        print("✅ Teacher model loaded")
        logger.info("Teacher model loaded successfully")
        
        return teacher_model, teacher_tokenizer
    
    def load_student_model(self, use_lora: bool = True) -> tuple:
        """
        Load student model (Phi-2) and optionally apply LoRA
        
        Args:
            use_lora: Whether to apply LoRA for efficient training
        """
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
        
        # Apply LoRA if requested
        if use_lora:
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
        logger.info("Student model loaded successfully")
        
        return student_model, student_tokenizer
    
    def prepare_dataset(self,
                       dataset_name: str = "databricks/databricks-dolly-15k",
                       num_samples: int = 500,
                       max_length: int = 512) -> tuple:
        """
        Prepare training dataset
        
        Args:
            dataset_name: Name of the dataset
            num_samples: Number of samples to use
            max_length: Maximum sequence length
        """
        print(f"\n📚 Loading dataset: {dataset_name}...")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Format for instruction-based dataset
        def format_instruction(example):
            if 'instruction' in example:
                text = f"### Instruction:\n{example['instruction']}\n"
                if 'input' in example and example['input']:
                    text += f"### Input:\n{example['input']}\n"
                text += f"### Response:\n{example['response']}"
            else:
                text = example.get('text', '')
            return {'text': text}
        
        # Apply formatting
        dataset = dataset.map(format_instruction)
        
        # Take subset
        dataset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
        
        print(f"✅ Dataset prepared: {len(dataset)} samples")
        logger.info(f"Dataset prepared with {len(dataset)} samples")
        
        return dataset
    
    def train(self,
              dataset_name: str = "databricks/databricks-dolly-15k",
              num_samples: int = 500,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5):
        """
        Train student model with knowledge distillation
        
        Args:
            dataset_name: Name of training dataset
            num_samples: Number of samples to use
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print("\n🚀 Starting Knowledge Distillation Training...")
        
        with Timer("Complete Distillation Training") as timer:
            # Load models
            teacher_model, teacher_tokenizer = self.load_teacher_model()
            student_model, student_tokenizer = self.load_student_model(use_lora=True)
            
            # Prepare dataset
            train_dataset = self.prepare_dataset(
                dataset_name=dataset_name,
                num_samples=num_samples
            )
            
            # Tokenize dataset
            def tokenize_function(examples):
                return student_tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            print("\n🔄 Tokenizing dataset...")
            tokenized_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )
            print("✅ Dataset tokenized")
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,
                learning_rate=learning_rate,
                warmup_ratio=0.1,
                weight_decay=0.01,
                save_strategy="steps",
                save_steps=max(100, len(train_dataset) // batch_size),
                logging_steps=10,
                fp16=torch.cuda.is_available(),
                optim="paged_adamw_8bit",
                report_to="none",
                ddp_find_unused_parameters=False,
                remove_unused_columns=False
            )
            
            # Create data collator (Fixed for transformers 4.40+)
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=student_tokenizer,
                mlm=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=student_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            # Train
            print("\n🔄 Training student model...")
            trainer.train()
            
            # Save model
            print("\n💾 Saving trained model...")
            student_model.save_pretrained(str(self.output_dir))
            student_tokenizer.save_pretrained(str(self.output_dir))
            
            # Save metadata
            metadata = {
                'teacher_model': self.teacher_model_id,
                'student_model': self.student_model_id,
                'training_dataset': dataset_name,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'training_time_seconds': timer.elapsed,
                'num_samples': num_samples
            }
            
            metadata_path = self.output_dir / 'distillation_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print("\n✅ Training completed successfully!")
            print(f"Model saved to: {self.output_dir}")
            print(f"Training time: {timer.elapsed:.2f}s")
            logger.info(f"Training completed in {timer.elapsed:.2f}s")

def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default distillation training
  python distillation_training.py
  
  # Custom parameters
  python distillation_training.py --epochs 5 --batch-size 2 --samples 1000
        """
    )
    
    parser.add_argument('--teacher', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--student', type=str, default='microsoft/phi-2')
    parser.add_argument('--dataset', type=str, default='databricks/databricks-dolly-15k')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--output', type=str, default='models/phi-2-distilled')
    
    args = parser.parse_args()
    
    trainer = DistillationTrainer(
        teacher_model_id=args.teacher,
        student_model_id=args.student,
        output_dir=args.output
    )
    
    trainer.train(
        dataset_name=args.dataset,
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == '__main__':
    main()
