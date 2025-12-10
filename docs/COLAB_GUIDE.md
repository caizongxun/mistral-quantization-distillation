# 🚀 Google Colab 完整指南

## 前置條件

- Google 帳號
- 訪問 Google Colab：https://colab.research.google.com
- 免費 GPU（T4 或 L4）

## Step 1：建立新 Notebook 並啟用 GPU

1. 打開 https://colab.research.google.com
2. 點擊「新增筆記本」
3. 右上角點「⚙️ 設定」
4. 在「硬體加速器」選 **GPU (T4 或 L4)**
5. 點「儲存」

## Step 2：環境檢查（Colab Cell 1）

複製貼上此代碼：

```python
# 檢查 GPU 狀態
import torch

print("\n" + "="*60)
print("🔧 Environment Check")
print("="*60)

print(f"\n✅ PyTorch Version: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n🖥️  GPU Details:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   VRAM: {vram_gb:.1f} GB")
    print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
else:
    print("⚠️  No GPU detected")

print("\n✅ Ready to start!")
print("="*60)
```

應該看到：

```
PyTorch Version: 2.9.0+cu126
CUDA Available: True
Device: Tesla T4
VRAM: 15.8 GB
```

---

## Step 3：克隆專案 + 安裝依賴（Colab Cell 2）

```python
# 克隆倉庫
!git clone https://github.com/caizongxun/mistral-quantization-distillation.git
%cd mistral-quantization-distillation

# 安裝依賴（會用 Colab 優化版本）
print("⏳ Installing dependencies (this may take 2-3 minutes)...")
!pip install -q -r requirements-colab.txt

print("\n✅ Dependencies installed!")
print("\n📂 Current directory:")
!pwd
!ls -la
```

---

## Step 4：Mistral-7B 4-bit 量化（Colab Cell 3）

**預計時間：10-15 分鐘**

```python
# 步驟 2.1：量化
print("\n" + "="*60)
print("🔥 Step 1: Mistral-7B 4-bit Quantization")
print("="*60)

!python mistral_quantization.py \
    --model mistralai/Mistral-7B-v0.1 \
    --output models/mistral-7b-4bit \
    --device cuda

print("\n✅ Quantization complete!")
```

會看到進度：

```
✅ Tokenizer downloaded
✅ Model loaded with 4-bit quantization
💾 After Quantization | Current: 4.1GB | Peak: 4.2GB | Diff: +4.1GB
✅ Model, tokenizer, and metadata saved successfully
```

---

## Step 5：FP16 vs 4-bit Benchmark（Colab Cell 4）

**預計時間：20-30 分鐘**

```python
print("\n" + "="*60)
print("📊 Step 2: Benchmark FP16 vs 4-bit")
print("="*60)

!python benchmark.py \
    --fp16-model mistralai/Mistral-7B-v0.1 \
    --quantized models/mistral-7b-4bit \
    --output outputs

print("\n✅ Benchmark complete!")
```

檢視結果：

```python
import pandas as pd

df = pd.read_csv("outputs/benchmark_results.csv")
print("\n📈 Benchmark Results:")
print(df.to_string(index=False))

# 計算差異
fp16_speed = df[df['Model'].str.contains('FP16')]['Tokens/s'].values[0]
quant_speed = df[df['Model'].str.contains('4-bit')]['Tokens/s'].values[0]
speedup = quant_speed / fp16_speed

print(f"\n🚀 Speedup: {speedup:.1f}x faster with 4-bit!")
```

---

## Step 6：Phi-2 知識蒸餾訓練（Colab Cell 5）

**預計時間：45-90 分鐘**（取決於配置）

```python
print("\n" + "="*60)
print("🎓 Step 3: Knowledge Distillation Training")
print("="*60)
print("\n⏳ This will take 45-90 minutes...\n")

!python distillation_training.py \
    --teacher mistralai/Mistral-7B-v0.1 \
    --student microsoft/phi-2 \
    --dataset databricks/databricks-dolly-15k \
    --samples 500 \
    --epochs 3 \
    --batch-size 4 \
    --lr 5e-5 \
    --output models/phi-2-distilled

print("\n✅ Distillation training complete!")
```

訓練會顯示：

```
👨‍🏫 Loading Teacher Model (Mistral-7B)...
👩‍💻 Loading Student Model (Phi-2)...
📚 Dataset prepared: 500 samples
🔄 Training student model...
Epoch 1/3: Loss 2.34
Epoch 2/3: Loss 1.89
Epoch 3/3: Loss 1.67
✅ Training completed in 1234.5s
```

---

## Step 7：三模型推理對比（Colab Cell 6）

**預計時間：5-10 分鐘**

```python
print("\n" + "="*60)
print("🧠 Step 4: Inference Comparison")
print("="*60)

# 單次測試
test_prompt = "What is artificial intelligence?"
print(f"\n📝 Test Prompt: {test_prompt}\n")

!python inference_comparison.py \
    --prompt "$test_prompt" \
    --fp16-model mistralai/Mistral-7B-v0.1 \
    --quantized models/mistral-7b-4bit \
    --distilled models/phi-2-distilled

print("\n✅ Inference comparison complete!")
```

會看到：

```
[Mistral FP16] Time: 0.80s | Tokens/s: 12.4 → Response...
[Mistral 4-bit] Time: 0.26s | Tokens/s: 38.6 → Response...
[Distilled Phi-2] Time: 0.35s | Tokens/s: 25.2 → Response...
```

---

## Step 8：啟動互動 Gradio Demo（Colab Cell 7）

**永遠在線（直到關閉 Notebook）**

```python
print("\n" + "="*60)
print("🎨 Step 5: Launch Gradio Demo")
print("="*60)

import subprocess
import time

print("\n⏳ Starting Gradio app...\n")

# 在後台啟動
process = subprocess.Popen(
    ['python', 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(5)

print("\n" + "="*60)
print("✅ Gradio app is running!")
print("="*60)
print("\n🌐 Open the link below in a new tab:")
print("   The interface will auto-appear in output below")
print("\n💡 You can now:")
print("   1. Enter prompts in the text box")
print("   2. See all 3 models respond in parallel")
print("   3. Compare speed/memory/quality")
print("   4. Chat history saved to outputs/chat_history.csv")
print("\n⏹️  To stop: Execute cell below")
```

---

## Step 9：下載結果到本地（Colab Cell 8）

```python
print("\n" + "="*60)
print("💾 Download Results")
print("="*60)

from google.colab import files
import shutil

print("\n📦 Preparing download package...\n")

# 建立下載資料夾
!mkdir -p colab_results
!cp -r models/mistral-7b-4bit colab_results/ 2>/dev/null || echo "Mistral 4-bit ready"
!cp -r models/phi-2-distilled colab_results/ 2>/dev/null || echo "Phi-2 distilled ready"
!cp outputs/benchmark_results.csv colab_results/ 2>/dev/null || echo "Benchmark results ready"
!cp outputs/chat_history.csv colab_results/ 2>/dev/null || echo "Chat history ready"

print("\n✅ Creating archive...\n")
!cd colab_results && du -sh * 2>/dev/null || echo ""

print("\n📥 Download starting...")
print("\n（如果沒有自動下載，點下面的連結）\n")

# 直接下載重要檔案
print("📄 Downloading individual files:\n")

try:
    files.download('outputs/benchmark_results.csv')
    print("✅ benchmark_results.csv")
except:
    print("⏭️  benchmark_results.csv not found yet")

try:
    files.download('outputs/chat_history.csv')
    print("✅ chat_history.csv")
except:
    print("⏭️  chat_history.csv not found yet")

print("\n💡 模型檔案可能很大，如需下載：")
print("   1. 右上角三點選『下載全部』")
print("   2. 或手動在檔案瀏覽器選取")
print("   3. 或用 Google Drive 同步")
```

---

## 完整 One-Cell 版本（快速方案）

如果想一個 cell 執行所有步驟：

```python
print("\n🚀 STARTING COMPLETE PIPELINE\n")

# 1. Clone & Install
!git clone https://github.com/caizongxun/mistral-quantization-distillation.git
%cd mistral-quantization-distillation
!pip install -q -r requirements-colab.txt

# 2. Quantize
print("\n--- STEP 1: Quantization ---")
!python mistral_quantization.py --output models/mistral-7b-4bit --device cuda

# 3. Benchmark
print("\n--- STEP 2: Benchmark ---")
!python benchmark.py --quantized models/mistral-7b-4bit --output outputs

print("\n" + "="*60)
print("✅ PIPELINE COMPLETE!")
print("="*60)

import pandas as pd
df = pd.read_csv("outputs/benchmark_results.csv")
print("\n📊 Results:")
print(df)
```

---

## 常見錯誤與解決

### ❌ `torch==2.1.2 not found`

**解決**：requirements 已更新用 `>=2.4.0`，自動用 Colab 的最新版本

```python
!pip install --upgrade -r requirements-colab.txt
```

### ❌ `CUDA out of memory`

**解決**：

```python
# 清空 GPU 記憶體
import torch
torch.cuda.empty_cache()

# 或降低 batch size
!python distillation_training.py --batch-size 2 --samples 200
```

### ❌ Model download hangs

**解決**：

```python
# 登入 Hugging Face（如需驗證）
from huggingface_hub import login
login(token="hf_xxxxx")  # 從 https://huggingface.co/settings/tokens 取得
```

### ❌ Colab 連線斷開

**預防**：

- 啟用「Keep session alive」（右上角）
- 模型會每次重新下載（用 cache）
- 定時下載結果

---

## 預期時間表

| Step | 任務 | 時間 | VRAM |
|------|------|------|------|
| 1 | Setup | 2 min | - |
| 2 | Quantize | 10 min | 4.1 GB |
| 3 | Benchmark | 25 min | 16 GB |
| 4 | Distill | 60 min | 14 GB |
| 5 | Inference | 5 min | 4 GB |
| 6 | Demo | unlimited | 4 GB |
| **Total** | **完整管線** | **~2 小時** | **16 GB** |

---

## 額外資源

- 🔗 [完整 Notebook](https://colab.research.google.com/github/caizongxun/mistral-quantization-distillation/blob/main/colab_full_pipeline.ipynb)
- 📖 [量化深入講解](docs/QUANTIZATION.md)
- 🎓 [蒸餾原理](docs/DISTILLATION.md)
- 🐛 [故障排查](docs/TROUBLESHOOTING.md)

---

## 下一步

1. ✅ 完成 Colab 上的所有步驟
2. 📥 下載量化模型 + 蒸餾模型
3. 🏠 解壓到本地專案資料夾
4. 🚀 本地執行 `python app.py` 使用

祝你使用愉快！ 🎉
