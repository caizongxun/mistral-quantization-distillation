# 🎓 Knowledge Distillation Guide

## What is Knowledge Distillation?

Knowledge distillation transfers knowledge from a **teacher model** (large, accurate) to a **student model** (small, fast):

```
Teacher Model (Mistral-7B)
        ↑
     Train
        ↑
Student Model (Phi-2)
```

## Why Distill?

| Aspect | Teacher (Mistral) | Student (Phi-2) |
|--------|-------------------|------------------|
| Size | 7B | 2.7B |
| Parameters | 7B | 2.7B |
| VRAM | 14GB (FP16) | 5GB (FP16) |
| Speed | 12 t/s | 30 t/s |
| Quality | Baseline | 92-96% of teacher |

## Training Process

### Step 1: Prepare Dataset

Use instruction-following dataset (Dolly-15k):

```python
from datasets import load_dataset

dataset = load_dataset("databricks/databricks-dolly-15k")
# Format: {instruction, input, response}
```

### Step 2: Setup Models

```python
# Teacher: Mistral in 4-bit (frozen)
teacher_model = load_quantized_model()
teacher_model.eval()

# Student: Phi-2 with LoRA (trainable)
student_model = load_student_model()
apply_lora(student_model)
```

### Step 3: Define Loss Function

Combine two losses:

```python
# KL Divergence Loss (for distribution matching)
kl_loss = torch.nn.functional.kl_div(
    student_probs,
    teacher_probs,
    reduction='batchmean'
) * (temperature ** 2)

# Language Modeling Loss (for next-token prediction)
lm_loss = model(input_ids, labels=labels).loss

# Total Loss
total_loss = alpha * kl_loss + (1-alpha) * lm_loss
```

### Step 4: Training Configuration

```json
{
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 5e-5,
  "temperature": 3.0,
  "gradient_accumulation_steps": 2
}
```

**Parameters:**
- **temperature**: Higher = softer probabilities (more info transfer)
- **alpha**: Weight balance between KL loss and LM loss
- **gradient_accumulation**: Simulate larger batch on limited VRAM

## Key Concepts

### Temperature Scaling

Soft targets from teacher:

```
soft_targets = softmax(teacher_logits / T)
```

Higher temperature → softer probabilities → more knowledge transfer

### KL Divergence

Measures difference between probability distributions:

```
KL(P||Q) = Σ P(i) * log(P(i) / Q(i))
```

Where P = teacher, Q = student

### LoRA (Low-Rank Adaptation)

Add trainable low-rank matrices to frozen weights:

```
Output = Frozen_Weight @ x + (A @ B) @ x
```

- **A**: (hidden_dim, r)
- **B**: (r, hidden_dim)
- **r**: Rank (typically 8-16)

Reduces trainable parameters by 99%!

## Results

### Typical Improvements

| Metric | FP16 Phi-2 | Distilled Phi-2 |
|--------|-----------|------------------|
| MMLU | 45% | 48-50% |
| Common Sense QA | 72% | 74-76% |
| Instruction Following | 78% | 82-85% |

### Training Time

- **Hardware**: RTX 3090 (24GB VRAM)
- **Dataset**: 10k samples
- **Time**: ~2 hours for 3 epochs
- **VRAM Used**: 18GB

## Advanced Techniques

### Multi-Teacher Distillation

Distill from multiple teachers:

```python
teachers = [mistral, llama, qwen]
for teacher in teachers:
    teacher_probs = teacher(input_ids)
    student_loss += kl_divergence(student, teacher_probs)
```

### Layer-wise Distillation

Match hidden states of intermediate layers:

```python
# In addition to output distillation
hidden_loss = mse_loss(student_hidden, teacher_hidden)
total_loss = output_loss + lambda * hidden_loss
```

## Troubleshooting

### Training Loss Not Decreasing

1. Reduce learning rate (1e-5)
2. Increase warmup steps
3. Check data format

### Poor Student Quality

1. Increase training duration
2. Use larger temperature (4-5)
3. Better dataset or sampling

### OOM Error

1. Reduce batch size (2)
2. Increase gradient accumulation (4)
3. Use lower precision (bfloat16)

## References

- Hinton et al.: https://arxiv.org/abs/1503.02531
- DistilBERT: https://arxiv.org/abs/1910.01108
- Phi-2 Paper: https://arxiv.org/abs/2309.05463

