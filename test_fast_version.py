#!/usr/bin/env python3
# ==============================================================================
# 檔案：test_fast_version.py
# 描述：測試快速訓練版本的準備工作和相容性
# ==============================================================================

import os

import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def test_data_availability():
    """測試資料檔案是否存在"""
    csv_path = "output_zh_optimized_v2.csv"

    print("=== 資料可用性測試 ===")

    if not os.path.exists(csv_path):
        print(f"❌ 錯誤：找不到資料檔案 {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 資料檔案載入成功")
        print(f"   完整資料集大小：{len(df)} 樣本")
        print(f"   10% 資料集大小：{int(len(df) * 0.1)} 樣本")

        # 檢查必要的欄位
        required_columns = ["file", "transcription"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ 錯誤：缺少必要欄位 {missing_columns}")
            return False

        print(f"✅ 資料欄位驗證通過")

        # 檢查檔案路徑
        sample_files = df["file"].head(5).tolist()
        existing_files = []
        for file_path in sample_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)

        print(f"✅ 樣本檔案檢查：{len(existing_files)}/{len(sample_files)} 個檔案存在")

        if len(existing_files) == 0:
            print("❌ 警告：沒有找到任何音訊檔案")
            return False

        return True

    except Exception as e:
        print(f"❌ 錯誤：無法讀取資料檔案 - {e}")
        return False


def test_model_loading():
    """測試模型載入"""
    print("\n=== 模型載入測試 ===")

    try:
        model_name = "MediaTek-Research/Breeze-ASR-25"
        print(f"載入模型：{model_name}")

        processor = WhisperProcessor.from_pretrained(
            model_name, language="zh", task="transcribe"
        )
        print("✅ Processor 載入成功")

        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        print("✅ 模型載入成功")

        # 檢查模型配置
        print(f"   模型類型：{model.config.model_type}")
        print(f"   模型大小：{model.config.d_model}")

        return True

    except Exception as e:
        print(f"❌ 錯誤：模型載入失敗 - {e}")
        return False


def test_gpu_availability():
    """測試 GPU 可用性"""
    print("\n=== GPU 可用性測試 ===")

    if torch.cuda.is_available():
        print("✅ CUDA 可用")
        print(f"   GPU 數量：{torch.cuda.device_count()}")
        print(f"   當前 GPU：{torch.cuda.get_device_name()}")

        # 檢查 VRAM
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   總 VRAM：{gpu_memory:.1f} GB")

        if gpu_memory >= 8:
            print("✅ GPU 記憶體充足")
            return True
        else:
            print("⚠️  警告：GPU 記憶體可能不足")
            return False
    else:
        print("❌ CUDA 不可用，將使用 CPU 訓練（非常慢）")
        return False


def test_dependencies():
    """測試依賴套件"""
    print("\n=== 依賴套件測試 ===")

    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "evaluate": "Evaluate",
        "librosa": "Librosa (音訊處理)",
        "soundfile": "SoundFile (音訊檔案)",
        "tensorboard": "TensorBoard (監控)",
    }

    all_available = True

    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - 需要安裝")
            all_available = False

    return all_available


def estimate_training_time():
    """估算訓練時間"""
    print("\n=== 訓練時間估算 ===")

    # 基於經驗值的估算
    steps_per_minute_gpu = 8  # RTX 3060Ti 的大約速度
    steps_per_minute_cpu = 1  # CPU 的大約速度

    total_steps = 1000

    if torch.cuda.is_available():
        estimated_minutes = total_steps / steps_per_minute_gpu
        print(f"📊 GPU 訓練估算：約 {estimated_minutes:.0f} 分鐘")
    else:
        estimated_minutes = total_steps / steps_per_minute_cpu
        print(f"📊 CPU 訓練估算：約 {estimated_minutes:.0f} 分鐘 (不建議)")


def main():
    print("=== 快速訓練版本相容性測試 ===\n")

    tests_passed = 0
    total_tests = 4

    # 執行所有測試
    if test_data_availability():
        tests_passed += 1

    if test_model_loading():
        tests_passed += 1

    if test_gpu_availability():
        tests_passed += 1

    if test_dependencies():
        tests_passed += 1

    # 估算訓練時間
    estimate_training_time()

    # 總結
    print(f"\n=== 測試總結 ===")
    print(f"通過測試：{tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("✅ 所有測試通過！可以開始快速訓練")
        print("\n下一步：")
        print("1. 執行：python finetune_Breeze_fast.py")
        print("2. 在另一個終端執行：python monitor_fast_training.py")
    elif tests_passed >= 2:
        print("⚠️  部分測試通過，可以嘗試訓練但可能遇到問題")
    else:
        print("❌ 測試失敗過多，建議先解決問題再進行訓練")


if __name__ == "__main__":
    main()
