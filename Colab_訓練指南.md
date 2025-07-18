# Google Colab Breeze-ASR-25 訓練指南

## 🚀 快速開始

### 1. 環境準備

在 Colab notebook 中執行以下指令：

```python
# 1. 檢查 GPU 環境
!nvidia-smi

# 2. 安裝必要套件
!pip install datasets transformers accelerate jiwer librosa soundfile

# 3. 掛載 Google Drive（如果資料在 Drive 中）
from google.colab import drive
drive.mount('/content/drive')

# 4. 切換到專案目錄
import os
os.chdir('/content/drive/MyDrive/audio_model')  # 調整為你的路徑
```

### 2. 上傳訓練腳本

將 `finetune_Breeze_optimal.py` 上傳到 Colab 或 Google Drive。

### 3. 準備資料檔案

確保以下檔案在同一目錄下：
- `metadata_train.csv` - 訓練資料
- `metadata_test.csv` - 測試資料
- 音訊檔案路徑正確

### 4. 執行訓練

```python
# 直接執行訓練腳本
!python finetune_Breeze_optimal.py
```

## 🎯 GPU 特定最佳化

### 自動 GPU 偵測配置

腳本會自動偵測你的 GPU 類型並設定最佳參數：

| GPU 類型 | 記憶體 | 批次大小 | 有效批次 | 預估時間 | 特殊設定 |
|---------|--------|----------|----------|----------|----------|
| K80     | 12GB   | 1×8      | 8        | 3-4小時  | FP32 only |
| T4      | 15GB   | 2×8      | 16       | 2-3小時  | FP16 |
| P100    | 16GB   | 3×6      | 18       | 2小時    | FP16 |
| V100    | 16GB   | 4×4      | 16       | 1.5小時  | FP16 |
| L4      | 22GB   | 6×3      | 18       | 1小時    | FP16 |
| A100    | 40GB   | 8×2      | 16       | 30分鐘   | FP16 |

### 手動參數調整

如果需要手動調整參數，編輯腳本中的 `detect_gpu_type()` 函數：

```python
# 例如：調整 T4 配置
elif "t4" in gpu_name:
    return "t4", {
        "batch_size": 1,  # 降低批次大小
        "gradient_accumulation_steps": 16,  # 增加累積步數
        "max_steps": 2000,  # 減少訓練步數
        # ... 其他參數
    }
```

## 📊 訓練監控

### 即時監控 GPU 使用

```python
# 在另一個 cell 中執行
import time
import torch

while True:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 記憶體: {allocated:.1f}GB 已分配, {cached:.1f}GB 已快取")
    time.sleep(30)
```

### 查看訓練日誌

訓練過程中會顯示：
- 🔍 GPU 類型偵測結果
- 📊 資料集大小和清理結果
- 📈 訓練參數配置
- 🏋️ 即時訓練指標

## 💾 模型保存與下載

### 自動下載

訓練完成後，執行腳本提供的下載指令：

```python
# 壓縮模型檔案
!zip -r breeze-asr-25-colab-optimized.zip breeze-asr-25-colab-optimized

# 下載到本地
from google.colab import files
files.download('breeze-asr-25-colab-optimized.zip')
```

### 儲存到 Google Drive

```python
# 複製到 Drive
!cp -r breeze-asr-25-colab-optimized /content/drive/MyDrive/
```

## ⚠️ 常見問題與解決方案

### 1. CUDA OOM 錯誤

```
RuntimeError: CUDA out of memory
```

**解決方案：**
```python
# 方法 1：重啟運行時
# Runtime -> Restart Runtime

# 方法 2：手動清理記憶體
import torch
torch.cuda.empty_cache()

# 方法 3：降低批次大小
# 編輯腳本中的 batch_size 參數
```

### 2. 會話中斷

**預防措施：**
```python
# 1. 啟用背景運行
from google.colab import output
output.enable_custom_widget_manager()

# 2. 設定更頻繁的保存
# 在腳本中調整 save_steps 為更小值
```

### 3. 資料載入失敗

**檢查清單：**
- ✅ CSV 檔案路徑正確
- ✅ 音訊檔案存在且可存取
- ✅ 欄位名稱正確（"file", "中文意譯"）
- ✅ 檔案編碼為 UTF-8

### 4. 記憶體不足警告

**最佳化策略：**
```python
# 1. 減少資料集大小
train_df = train_df.sample(n=10000)  # 只使用 10K 樣本

# 2. 縮短音訊長度
# 在 prepare_dataset_colab 中調整 duration=15

# 3. 降低序列長度
# 調整 max_length=128
```

## 🔧 進階設定

### 1. 啟用 Gradient Checkpointing

對於記憶體極度受限的情況：

```python
# 在 training_args 中設定
gradient_checkpointing=True,
```

### 2. 使用混合精度訓練

```python
# 確保 FP16 啟用
fp16=True,
fp16_opt_level="O1",  # 可選
```

### 3. 自定義學習率調度

```python
# 在 training_args 中
lr_scheduler_type="polynomial",  # 或 "linear"
warmup_ratio=0.1,  # 替代 warmup_steps
```

## 📈 效能最佳化建議

### Colab Pro 使用者

- 💡 選擇 **Premium GPU** 以獲得 V100 或 A100
- ⏰ 在**離峰時間**訓練（美國時間深夜）
- 💾 使用 **高RAM 運行時** 處理大資料集

### 免費用戶優化

- 🎯 使用 **小資料集**（< 20K 樣本）
- ⚡ 設定較少的 **訓練步數**（< 3000 步）
- 🔄 準備在會話中斷時**恢復訓練**

### 資料前處理優化

```python
# 1. 預先篩選音訊長度
df = df[df['duration'] < 30]  # 只保留 30 秒以下

# 2. 移除異常資料
df = df[df['中文意譯'].str.len().between(5, 100)]

# 3. 批次載入
# 使用 with_transform 而非預先處理
```

## 🎯 訓練策略建議

### 快速驗證（< 1 小時）

```python
# 設定小規模測試
max_steps = 500
train_samples = 5000
eval_steps = 100
save_steps = 100
```

### 完整訓練（2-4 小時）

```python
# 使用預設 GPU 偵測設定
# 建議在 V100 或更好的 GPU 上執行
```

### 生產級訓練（> 4 小時）

```python
# 使用 A100 或多 GPU
# 考慮分段訓練和檢查點恢復
```

## 📝 檢查清單

### 開始訓練前

- [ ] GPU 運行時已啟用
- [ ] 必要套件已安裝
- [ ] 資料檔案路徑正確
- [ ] 磁碟空間充足（> 5GB）
- [ ] Google Drive 已掛載（如需要）

### 訓練期間

- [ ] 監控 GPU 記憶體使用
- [ ] 檢查訓練日誌中的錯誤
- [ ] 確認損失值下降
- [ ] 檢查 WER 指標改善

### 訓練完成後

- [ ] 模型檔案完整保存
- [ ] 下載或備份到 Drive
- [ ] 記錄最終指標
- [ ] 清理暫存檔案

---

## 💬 支援與回饋

如果遇到問題：

1. 檢查錯誤訊息中的具體 GPU 類型
2. 參考上述對應的解決方案
3. 嘗試降低參數設定重新訓練
4. 必要時重啟 Colab 運行時

**祝你訓練順利！** 🎉 