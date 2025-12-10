# Dependencies and Variables Documentation

## Project Dependencies

### Core Libraries

1. **PyTorch** (`torch`)
   - Version: >=2.0.0
   - Purpose: Deep learning framework for model inference and training
   - Key components used:
     - `torch.cuda`: GPU acceleration
     - `torch.no_grad()`: Disable gradient computation for inference
     - `torch.float16`: Half-precision floating point
     - `torch.cuda.empty_cache()`: Clear GPU memory

2. **Transformers** (`transformers`)
   - Version: >=4.30.0
   - Purpose: Pre-trained model loading and inference
   - Key components:
     - `AutoTokenizer`: Load tokenizer by model name
     - `AutoModelForCausalLM`: Load language model
     - `BitsAndBytesConfig`: Configure 4-bit quantization
     - `Trainer`: High-level training API
     - `TrainingArguments`: Configuration for training
     - `DataCollatorForLanguageModeling`: Batch processing for language models

3. **PEFT** (Parameter-Efficient Fine-Tuning) (`peft`)
   - Version: >=0.4.0
   - Purpose: Low-rank adaptation (LoRA) for efficient fine-tuning
   - Key components:
     - `LoraConfig`: Configure LoRA parameters
     - `get_peft_model`: Wrap model with LoRA
     - `AutoPeftModelForCausalLM`: Load LoRA-adapted models

4. **BitsAndBytes** (`bitsandbytes`)
   - Version: >=0.39.0
   - Purpose: 8-bit and 4-bit quantization
   - Key functions:
     - `BitsAndBytesConfig`: 4-bit quantization configuration

5. **Datasets** (`datasets`)
   - Version: >=2.10.0
   - Purpose: Download and process datasets
   - Key functions:
     - `load_dataset()`: Load dataset from Hub
     - `.map()`: Apply transformations
     - `.select()`: Sample subset

6. **Accelerate** (`accelerate`)
   - Version: >=0.20.0
   - Purpose: Multi-GPU training and mixed precision
   - Integrated in Transformers Trainer

### Optional Cloud Libraries

7. **Hugging Face Hub** (`huggingface_hub`)
   - Purpose: Upload models to Hugging Face Model Hub
   - Key: `HfApi.upload_folder()`

8. **PyDrive** (`pydrive`)
   - Purpose: Upload to Google Drive
   - Key: `GoogleAuth`, `GoogleDrive`

9. **Boto3** (`boto3`)
   - Purpose: AWS S3 upload
   - Key: `s3.upload_file()`

---

## Key Variables Explained

### Model Configuration Variables

```python
# Model names and paths
models = {
    'base': 'models/phi-2-base',           # Base Phi-2 model directory
    'quantized': 'models/phi-2-quantized', # 4-bit quantized version
    'lora': 'models/phi-2-lora',           # LoRA fine-tuned version
    'lora_quantized': 'models/phi-2-lora-quantized'  # LoRA + quantized
}

# Training hyperparameters
num_samples = 100          # Number of training samples
num_epochs = 1             # Training epochs
batch_size = 1             # Batch size per device
learning_rate = 5e-5       # Learning rate
```

### LoRA Configuration Variables

```python
# From LoraConfig
r = 8                  # LoRA rank (factorization dimension)
lora_alpha = 16        # LoRA scaling factor
target_modules = [     # Which model layers to apply LoRA
    "q_proj",         # Query projection
    "v_proj"          # Value projection
]
lora_dropout = 0.1     # Dropout for LoRA layers
```

### Quantization Configuration Variables

```python
# From BitsAndBytesConfig
load_in_4bit = True              # Enable 4-bit quantization
bnb_4bit_use_double_quant = True # Use double quantization
bnb_4bit_quant_type = "nf4"      # Quantization type (nf4 = normalized float4)
bnb_4bit_compute_dtype = torch.float16  # Computation precision
```

### Dataset Variables

```python
dataset_name = "databricks/databricks-dolly-15k"  # Instruction-following dataset
max_length = 256               # Maximum token sequence length
padding = "max_length"         # Padding strategy
truncation = True              # Truncate sequences
```

### Training Variables

```python
# TrainingArguments
save_strategy = "no"           # Don't save intermediate checkpoints
fp16 = True                    # Use 16-bit floating point
optim = "paged_adamw_8bit"    # Optimizer (8-bit Adam)
max_grad_norm = 0.3            # Gradient clipping
gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps
warmup_steps = 10              # Learning rate warmup
```

### Device and Memory Variables

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_memory = torch.cuda.get_device_properties(0).total_memory
supports_tf32 = gpu_capability[0] >= 8  # Ampere or newer
```

### Tokenizer Variables

```python
pad_token = tokenizer.eos_token  # Use EOS token for padding
eos_token_id = 50256             # End of sequence token ID
```

---

## Installation Command

```bash
pip install torch transformers peft bitsandbytes datasets accelerate

# Optional: For cloud upload
pip install huggingface_hub pydrive boto3
```

---

## File Structure and Variables

### complete_training_pipeline.py

**Key Classes:**
- `Timer`: Context manager for timing operations
- `CompleteTrainingPipeline`: Main training orchestrator

**Key Methods:**
- `prepare_dataset()`: Load and tokenize data
- `stage1_save_base_model()`: Download and save base model
- `stage2_quantize_base_model()`: Apply 4-bit quantization
- `stage3_lora_finetuning()`: Fine-tune with LoRA
- `stage4_quantize_lora_model()`: Quantize fine-tuned model

**Key Variables:**
- `self.paths`: Dictionary of model directories
- `self.device`: 'cuda' or 'cpu'
- `self.supports_tf32`: GPU capability flag

### model_evaluation.py

**Key Class:**
- `ModelEvaluator`: Evaluate and compare model versions

**Key Methods:**
- `load_*_model()`: Load different model versions
- `evaluate_model()`: Test inference speed and quality
- `print_summary()`: Generate comparison report

**Key Variables:**
- `self.models`: Dictionary mapping model types to paths
- `self.test_prompts`: List of evaluation prompts
- `results`: Dictionary storing evaluation metrics

### upload_to_cloud.py

**Key Class:**
- `CloudUploader`: Handle cloud storage uploads

**Key Methods:**
- `upload_to_huggingface()`: Upload to HF Hub
- `upload_to_gdrive()`: Upload to Google Drive
- `upload_to_s3()`: Upload to AWS S3
- `create_zip_backup()`: Local backup

**Key Variables:**
- `self.model_path`: Path to model directory
- `self.model_size`: Size in GB

---

## Common Issues and Solutions

### Out of Memory (OOM)
- Reduce `batch_size`
- Reduce `max_length`
- Reduce `num_samples`
- Enable `gradient_checkpointing`

### Slow Inference on Quantized Models
- Ensure `device_map="auto"` for GPU loading
- Check GPU utilization with `nvidia-smi`
- Verify quantization was applied correctly

### LoRA Not Improving Results
- Check `target_modules` matches model architecture
- Increase `lora_rank` (r)
- Use larger `num_samples` and `num_epochs`
- Verify training loss is decreasing

---

## Performance Metrics

### Expected Results
- Base model: ~2.3s per inference
- Quantized: ~0.8s per inference (2.9x faster)
- LoRA: ~2.4s per inference (7% better accuracy)
- LoRA Quantized: ~0.8s per inference (2.9x faster + 7% better accuracy)

### Model Sizes
- Base: ~5GB
- Quantized: ~1.2GB (24% of original)
- LoRA: ~5GB (with separate 655K adapter)
- LoRA Quantized: ~1.2GB (24% of original)
