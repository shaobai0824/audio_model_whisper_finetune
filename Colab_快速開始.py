# Google Colab Breeze-ASR-25 快速開始腳本
# 在 Colab notebook 中逐個執行這些程式碼塊

# =============================================================================
# 第一步：環境設定和安裝
# =============================================================================

# 1.1 檢查 GPU 環境
print("🔍 檢查 GPU 環境...")
!nvidia-smi

# 1.2 安裝必要套件
print("\n📦 安裝必要套件...")
!pip install datasets transformers accelerate jiwer librosa soundfile

# 1.3 掛載 Google Drive
print("\n💾 掛載 Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# 第二步：專案設定
# =============================================================================

# 2.1 設定專案目錄
import os
from pathlib import Path

# 修改為你的實際路徑
project_dir = "/content/drive/MyDrive/audio_model"
os.chdir(project_dir)

print(f"📁 當前工作目錄: {os.getcwd()}")

# 2.2 檢查必要檔案
required_files = [
    "metadata_train.csv",
    "metadata_test.csv", 
    "finetune_Breeze_optimal.py"
]

missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - 未找到")
        missing_files.append(file)

if missing_files:
    print(f"\n⚠️  請確保以下檔案存在於專案目錄中: {missing_files}")

# =============================================================================
# 第三步：路徑檢查和修正
# =============================================================================

# 3.1 檢查音訊檔案目錄
audio_dir = os.path.join(project_dir, "audio_files")
if os.path.exists(audio_dir):
    file_count = len([f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3', '.flac'))])
    print(f"✅ 音訊目錄存在: {audio_dir}")
    print(f"   音訊檔案數量: {file_count}")
else:
    print(f"❌ 音訊目錄不存在: {audio_dir}")
    print("   請將音訊檔案上傳到 Google Drive 的正確位置")

# 3.2 自動修正 CSV 路徑（如果需要）
import pandas as pd

def check_and_fix_csv_paths(csv_file):
    """檢查和修正 CSV 中的路徑"""
    if not os.path.exists(csv_file):
        print(f"❌ CSV 檔案不存在: {csv_file}")
        return False
    
    df = pd.read_csv(csv_file)
    path_column = "file"  # 假設路徑在 'file' 欄位
    
    if path_column not in df.columns:
        print(f"⚠️  CSV 中未找到 '{path_column}' 欄位")
        print(f"   可用欄位: {list(df.columns)}")
        return False
    
    sample_path = str(df[path_column].iloc[0]).strip()
    print(f"📝 {csv_file} 範例路徑: {sample_path}")
    
    # 檢查是否需要修正
    if sample_path.startswith(('C:', 'D:', 'E:', 'F:')):
        print(f"   🔄 修正 Windows 路徑...")
        df[path_column] = df[path_column].apply(
            lambda x: f"/content/drive/MyDrive/audio_model/audio_files/{Path(x).name}"
            if isinstance(x, str) and x.startswith(('C:', 'D:', 'E:', 'F:')) else x
        )
        
        # 保存修正後的檔案
        backup_file = f"{csv_file}.backup"
        df.to_csv(backup_file, index=False)  # 備份原檔案
        df.to_csv(csv_file, index=False)    # 覆蓋原檔案
        
        print(f"   ✅ 路徑已修正並保存")
        print(f"   💾 原檔案備份至: {backup_file}")
        print(f"   📝 新路徑範例: {df[path_column].iloc[0]}")
        return True
    
    elif sample_path.startswith('/content/drive/'):
        print(f"   ✅ 路徑格式正確")
        return True
    
    else:
        print(f"   ⚠️  路徑格式未知，可能需要手動調整")
        return False

# 檢查和修正訓練資料
print("\n🔧 檢查訓練資料路徑...")
check_and_fix_csv_paths("metadata_train.csv")

print("\n🔧 檢查測試資料路徑...")
check_and_fix_csv_paths("metadata_test.csv")

# =============================================================================
# 第四步：驗證設定
# =============================================================================

# 4.1 測試讀取資料
print("\n🧪 測試資料讀取...")
try:
    train_df = pd.read_csv("metadata_train.csv")
    test_df = pd.read_csv("metadata_test.csv")
    
    print(f"✅ 訓練資料: {len(train_df)} 筆")
    print(f"✅ 測試資料: {len(test_df)} 筆")
    
    # 檢查路徑欄位
    if "file" in train_df.columns:
        sample_audio_path = train_df["file"].iloc[0]
        print(f"📝 範例音訊路徑: {sample_audio_path}")
        
        # 檢查檔案是否實際存在
        if os.path.exists(sample_audio_path):
            print("✅ 範例檔案存在")
        else:
            print("❌ 範例檔案不存在")
            print("   請確認音訊檔案已正確上傳到 Google Drive")
    
    # 檢查文字欄位
    if "中文意譯" in train_df.columns:
        sample_text = train_df["中文意譯"].iloc[0]
        print(f"📝 範例文字: {sample_text}")
        print("✅ 文字欄位正常")
    else:
        print("❌ 未找到 '中文意譯' 欄位")
        print(f"   可用欄位: {list(train_df.columns)}")

except Exception as e:
    print(f"❌ 資料讀取失敗: {e}")

# 4.2 記憶體檢查
print("\n💾 記憶體檢查...")
import psutil

ram_gb = psutil.virtual_memory().total / 1024**3
print(f"   RAM: {ram_gb:.1f} GB")

disk_usage = psutil.disk_usage('/')
free_gb = disk_usage.free / 1024**3
print(f"   可用磁碟空間: {free_gb:.1f} GB")

if free_gb < 5:
    print("   ⚠️  磁碟空間不足，建議清理後再訓練")

# =============================================================================
# 第五步：開始訓練
# =============================================================================

print("\n🚀 準備開始訓練...")
print("如果上述檢查都通過，請執行以下指令開始訓練：")
print("!python finetune_Breeze_optimal.py")

# 可選：直接執行訓練（取消註解下面這行）
# !python finetune_Breeze_optimal.py

# =============================================================================
# 附加工具：訓練監控
# =============================================================================

def monitor_gpu_memory():
    """監控 GPU 記憶體使用"""
    import time
    import torch
    
    print("開始監控 GPU 記憶體（按 Ctrl+C 停止）...")
    
    try:
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"GPU 記憶體: {allocated:.1f}GB 已分配 / {cached:.1f}GB 已快取 / {total:.1f}GB 總計")
                
                if allocated / total > 0.9:
                    print("⚠️  GPU 記憶體使用率超過 90%")
            else:
                print("❌ 未偵測到 GPU")
                break
                
            time.sleep(30)
    except KeyboardInterrupt:
        print("監控已停止")

# 使用方法：在另一個 cell 中執行
# monitor_gpu_memory()

print("\n✅ 快速開始設定完成！")
print("💡 提示：建議在新的 cell 中執行訓練指令，以便更好地監控進度") 