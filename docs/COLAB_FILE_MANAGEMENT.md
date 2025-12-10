# 📁 Colab 檔案管理 + 下載指南

## 檔案存放位置

當你在 Colab 執行完量化 / 蒸餾 / Benchmark 後，檔案會在 **Colab 虛擬機的暫時儲存空間** 裡。

### 完整目錄結構

```
/content/mistral-quantization-distillation/
├── models/
│   ├── mistral-7b-4bit/              ← 量化模型（主要檔案）
│   │   ├── model-00001-of-00002.safetensors  (2.5GB)
│   │   ├── model-00002-of-00002.safetensors  (1.5GB)
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer.model
│   │   ├── special_tokens_map.json
│   │   ├── quantization_metadata.json
│   │   └── ...
│   ├── phi-2-distilled/               ← 蒸餾模型（如果執行）
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── ...
│   └── checkpoints/                   ← 訓練檢查點
├── outputs/
│   ├── benchmark_results.csv          ← 基準測試結果
│   └── chat_history.csv               ← 聊天記錄（如有執行 Demo）
├── logs/
│   ├── quantization.log
│   ├── benchmark.log
│   ├── distillation.log
│   └── app.log
└── ...
```

---

## 🔍 檢查檔案位置

### 方法 1：在 Colab 檢查（簡單）

```python
# 在 Colab Cell 執行
import os
import shutil

print("\n=== 檔案位置 ===")
print(f"\n當前目錄: {os.getcwd()}")

# 檢查量化模型
quant_path = "models/mistral-7b-4bit"
if os.path.exists(quant_path):
    files = os.listdir(quant_path)
    total_size = sum(os.path.getsize(os.path.join(quant_path, f)) for f in files) / 1e9
    print(f"\n✅ 量化模型位置: {os.path.abspath(quant_path)}")
    print(f"   檔案數: {len(files)}")
    print(f"   總大小: {total_size:.2f}GB")
    print(f"\n   包含檔案:")
    for f in sorted(files)[:10]:
        size = os.path.getsize(os.path.join(quant_path, f)) / 1e9
        print(f"      - {f} ({size:.2f}GB)")

# 檢查蒸餾模型
dist_path = "models/phi-2-distilled"
if os.path.exists(dist_path):
    files = os.listdir(dist_path)
    total_size = sum(os.path.getsize(os.path.join(dist_path, f)) for f in files) / 1e9
    print(f"\n✅ 蒸餾模型位置: {os.path.abspath(dist_path)}")
    print(f"   檔案數: {len(files)}")
    print(f"   總大小: {total_size:.2f}GB")

# 檢查結果
outputs_path = "outputs"
if os.path.exists(outputs_path):
    files = os.listdir(outputs_path)
    print(f"\n✅ 結果位置: {os.path.abspath(outputs_path)}")
    print(f"   包含檔案:")
    for f in files:
        fpath = os.path.join(outputs_path, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1e6
            print(f"      - {f} ({size:.2f}MB)")
```

執行後會看到：

```
當前目錄: /content/mistral-quantization-distillation

✅ 量化模型位置: /content/mistral-quantization-distillation/models/mistral-7b-4bit
   檔案數: 12
   總大小: 4.12GB
   
   包含檔案:
      - config.json (0.00GB)
      - model-00001-of-00002.safetensors (2.48GB)
      - model-00002-of-00002.safetensors (1.50GB)
      - tokenizer.json (1.72GB)
      - ...

✅ 結果位置: /content/mistral-quantization-distillation/outputs
   包含檔案:
      - benchmark_results.csv (0.01MB)
      - chat_history.csv (0.15MB)
```

---

## 📥 下載檔案到本地

### 方法 1：Google Drive 同步（推薦 - 大檔案）

**優點：** 可以同步大量檔案，不怕 Colab 連線斷開

```python
# 在 Colab Cell 執行
from google.colab import drive
import shutil
import os

# 掛載 Google Drive
print("🔐 Mounting Google Drive...")
drive.mount('/content/gdrive', force_remount=True)

print("\n✅ Google Drive mounted!")

# 建立備份資料夾
backup_dir = '/content/gdrive/My Drive/mistral-backup'
os.makedirs(backup_dir, exist_ok=True)

print(f"\n📁 Backup directory: {backup_dir}")

# 複製量化模型
print("\n⏳ Copying quantized model (this may take 5-10 minutes)...")
src = "models/mistral-7b-4bit"
dst = os.path.join(backup_dir, "mistral-7b-4bit")
shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"✅ Copied to: {dst}")

# 複製蒸餾模型（如果有）
if os.path.exists("models/phi-2-distilled"):
    print("\n⏳ Copying distilled model...")
    src = "models/phi-2-distilled"
    dst = os.path.join(backup_dir, "phi-2-distilled")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"✅ Copied to: {dst}")

# 複製結果
print("\n⏳ Copying results...")
src = "outputs"
dst = os.path.join(backup_dir, "outputs")
shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"✅ Copied to: {dst}")

print("\n✅ Backup complete!")
print(f"\n📱 Files are now in Google Drive: {backup_dir}")
print("   You can download them anytime from drive.google.com")
```

執行後，檔案會出現在：
- **Google Drive** → **My Drive** → **mistral-backup** → 分別有你的模型和結果

然後你可以：
1. 打開 drive.google.com
2. 在 mistral-backup 資料夾裡右鍵 → 下載

---

### 方法 2：直接從 Colab 下載（小檔案）

**適用於：** CSV 結果、config 檔等小檔案

```python
# 在 Colab Cell 執行
from google.colab import files

print("\n📥 Downloading files...\n")

# 下載 Benchmark 結果
if os.path.exists("outputs/benchmark_results.csv"):
    files.download("outputs/benchmark_results.csv")
    print("✅ benchmark_results.csv downloaded")

# 下載聊天記錄
if os.path.exists("outputs/chat_history.csv"):
    files.download("outputs/chat_history.csv")
    print("✅ chat_history.csv downloaded")

# 下載量化 metadata
if os.path.exists("models/mistral-7b-4bit/quantization_metadata.json"):
    files.download("models/mistral-7b-4bit/quantization_metadata.json")
    print("✅ quantization_metadata.json downloaded")

print("\n💾 Files downloaded to your computer!")
```

**注意：** 模型檔案很大（4GB），直接下載會很慢。建議用 Google Drive 方法。

---

### 方法 3：打包後下載（中檔案）

**適用於：** 想要打包一些檔案一起下載

```python
# 在 Colab Cell 執行
import zipfile
import os
from pathlib import Path

print("\n📦 Creating archive...\n")

# 只打包 config 和 metadata（不含大的 safetensors）
with zipfile.ZipFile('mistral_config_only.zip', 'w') as zipf:
    # 添加 config
    if os.path.exists("models/mistral-7b-4bit/config.json"):
        zipf.write("models/mistral-7b-4bit/config.json", "config.json")
    
    # 添加 metadata
    if os.path.exists("models/mistral-7b-4bit/quantization_metadata.json"):
        zipf.write("models/mistral-7b-4bit/quantization_metadata.json", "quantization_metadata.json")
    
    # 添加結果
    if os.path.exists("outputs/benchmark_results.csv"):
        zipf.write("outputs/benchmark_results.csv", "benchmark_results.csv")

file_size = os.path.getsize('mistral_config_only.zip') / 1e6
print(f"✅ Archive created: mistral_config_only.zip ({file_size:.2f}MB)")

print("\n📥 Downloading...")
from google.colab import files
files.download('mistral_config_only.zip')
```

---

## ⏰ Colab 檔案會保留多久？

### 時間限制

| 情況 | 保留時間 |
|------|----------|
| Colab 連線斷開 | 12 小時 |
| Notebook 關閉 | 12 小時 |
| 未使用 | 12 小時 |
| 在 Google Drive | **永久** |

**結論：** 為了不遺失檔案，**一定要複製到 Google Drive** 或 **立即下載** ❌

---

## 🎯 推薦流程

### Step 1：完成訓練（在 Colab）

```python
!python mistral_quantization.py --output models/mistral-7b-4bit
!python distillation_training.py --samples 500 --output models/phi-2-distilled
```

### Step 2：備份到 Google Drive（在 Colab）

```python
from google.colab import drive
import shutil

drive.mount('/content/gdrive')
backup_dir = '/content/gdrive/My Drive/mistral-models'

# 複製所有東西
shutil.copytree('models/mistral-7b-4bit', 
                f'{backup_dir}/mistral-7b-4bit', 
                dirs_exist_ok=True)
shutil.copytree('models/phi-2-distilled', 
                f'{backup_dir}/phi-2-distilled', 
                dirs_exist_ok=True)
shutil.copytree('outputs', f'{backup_dir}/outputs', dirs_exist_ok=True)

print("✅ Backup complete!")
```

### Step 3：下載結果到電腦

1. 打開 Google Drive
2. 進入 My Drive → mistral-models
3. 選擇要下載的檔案夾 → 右鍵 → 下載

### Step 4：本地使用

```bash
# 解壓下載的檔案
unzip mistral-7b-4bit.zip -d ./models/

# 執行推理
python inference_comparison.py
python app.py
```

---

## 💾 儲存空間限制

### Colab 免費版

- **總空間：** ~100GB
- **可用空間：** 約 70-80GB（系統占用）
- **模型需求：**
  - Mistral 4-bit: 4.1GB
  - Phi-2 蒸餾: 3-5GB
  - 結果檔案: <1GB
  - **總計：** 約 8-10GB（完全沒問題）

### 檢查剩餘空間

```python
!df -h | grep -E 'Filesystem|root'
```

---

## 🚨 檔案遺失救助

### 如果 Colab 連線斷開

**情況1：檔案還在 Google Drive**
- ✅ 打開 drive.google.com 直接下載

**情況2：檔案只在 Colab 暫存**
- ✅ 重新執行 Colab 會重新下載模型（10-15 分鐘）
- ✅ 或從 Hugging Face 直接載入

```python
from transformers import AutoModelForCausalLM

# 直接從 HF 載入量化版本
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    device_map="auto",
    load_in_4bit=True
)
```

---

## 📋 檔案清單

### 你應該保存的檔案

| 檔案 | 大小 | 重要性 | 說明 |
|-----|------|--------|------|
| `mistral-7b-4bit/` | 4.1GB | ⭐⭐⭐ | 量化模型 - 必須保存 |
| `phi-2-distilled/` | 3-5GB | ⭐⭐⭐ | 蒸餾模型 - 必須保存 |
| `outputs/benchmark_results.csv` | <1MB | ⭐⭐ | 性能對比結果 |
| `outputs/chat_history.csv` | <5MB | ⭐ | 聊天記錄 - 可選 |
| `models/checkpoints/` | 變動 | ⭐⭐ | 訓練檢查點 - 用於恢復 |

---

## ✅ 檢查清單

- [ ] 檔案已在 Colab `/content/mistral-quantization-distillation/` 下
- [ ] 已備份到 Google Drive
- [ ] 已下載主要檔案到本地
- [ ] benchmark_results.csv 已查看
- [ ] 準備好本地使用

---

**下次見！** 🚀
