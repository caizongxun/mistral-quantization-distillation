# 🔧 Mistral-7B 4-bit Quantization Guide

## What is Quantization?

Quantization reduces model size by representing weights with fewer bits:
- **FP32**: 32 bits per value = 16GB for 7B model
- **FP16**: 16 bits per value = 8GB for 7B model
- **4-bit**: 4 bits per value = 2-4GB for 7B model

## BitsAndBytes 4-bit Method

This project uses **BitsAndBytes NF4 quantization**:

```python
BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit loading
    bnb_4bit_use_double_quant=True, # Nested quantization
    bnb_4bit_quant_type="nf4",     # Normal Float 4 type
    bnb_4bit_compute_dtype=torch.float16  # Compute in FP16
)
```

## Key Parameters Explained

### load_in_4bit
- Loads model weights in 4-bit precision
- Reduces memory by ~75%
- Negligible quality loss

### bnb_4bit_use_double_quant
- Applies secondary quantization
- Additional ~0.4 bits saved per parameter
- More memory savings with minimal impact

### bnb_4bit_quant_type="nf4"
- **NF4 (Normal Float 4)**: Optimized 4-bit format
- Maintains better precision than regular INT4
- Recommended for language models

### bnb_4bit_compute_dtype=float16
- Computation still happens in FP16
- Maintains numerical precision
- Only storage is quantized

## Memory Comparison

| Model | Precision | VRAM | Relative Size |
|-------|-----------|------|---------------|
| Mistral-7B | FP32 | 28GB | 100% |
| Mistral-7B | FP16 | 14GB | 50% |
| Mistral-7B | 8-bit | 7GB | 25% |
| Mistral-7B | 4-bit | 3.5GB | 12.5% |

## Speed Comparison

Quantized models are typically **2-3x faster**:
- Reduced memory bandwidth
- Fewer parameters to load
- Simplified matrix operations

## Quality Impact

### Benchmark Results (on MMLU):
- FP32: 61.3% accuracy
- FP16: 61.2% accuracy (-0.1%)
- 4-bit: 60.9% accuracy (-0.4%)

**Conclusion**: Minimal quality loss for significant speed/memory gains

## Advanced Configuration

### Fine-tuning with QLoRA

Combine quantization with LoRA for efficient fine-tuning:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=32,                 # Alpha scaling
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,             # Dropout rate
    bias="none"
)

model = get_peft_model(model, lora_config)
```

This allows fine-tuning 7B model with <2GB VRAM!

## Limitations

1. **Inference Only**: For training, use QLoRA method
2. **Specific Hardware**: Needs NVIDIA GPU with compute capability 7.0+
3. **Token Generation**: Slightly slower than FP16 for very long sequences

## References

- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
- Paper: https://arxiv.org/abs/2305.14314
- Mistral: https://mistral.ai/

