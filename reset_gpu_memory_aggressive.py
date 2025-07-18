#!/usr/bin/env python3
# ==============================================================================
# 檔案：reset_gpu_memory.py
# 描述：重設 GPU 記憶體並檢查系統狀態
# ==============================================================================

import gc
import os
import subprocess
import sys

import torch


def force_memory_cleanup():
    """強制清理所有記憶體"""
    print("🧹 開始強制記憶體清理...")

    # 1. Python 垃圾回收（多次執行）
    for i in range(5):
        collected = gc.collect()
        print(f"   垃圾回收第 {i+1} 次：清理了 {collected} 個物件")

    # 2. CUDA 記憶體清理
    if torch.cuda.is_available():
        print("   清理 CUDA 記憶體快取...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 檢查記憶體狀態
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(
            f"   記憶體狀態：{allocated:.2f}GB 分配 / {reserved:.2f}GB 保留 / {total:.2f}GB 總計"
        )

        if allocated > 0.1:  # 如果還有超過 100MB 被分配
            print("   ⚠️ 仍有記憶體被佔用，建議重啟 Python 程序")
        else:
            print("   ✅ CUDA 記憶體已清理")
    else:
        print("   ⚠️ CUDA 不可用")


def check_gpu_processes():
    """檢查 GPU 進程"""
    print("\n🔍 檢查 GPU 進程...")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                print("   發現 GPU 進程：")
                for line in lines:
                    if line.strip():
                        parts = line.split(", ")
                        if len(parts) >= 3:
                            pid, name, memory = parts[0], parts[1], parts[2]
                            print(f"     PID: {pid}, 程序: {name}, 記憶體: {memory}MB")
            else:
                print("   ✅ 沒有 GPU 進程")
        else:
            print("   ⚠️ 無法執行 nvidia-smi")

    except FileNotFoundError:
        print("   ⚠️ 找不到 nvidia-smi")
    except Exception as e:
        print(f"   ❌ 檢查失敗：{e}")


def test_memory_allocation():
    """測試記憶體分配"""
    print("\n🧪 測試記憶體分配...")

    if not torch.cuda.is_available():
        print("   跳過（CUDA 不可用）")
        return False

    try:
        # 測試分配 100MB
        test_tensor = torch.rand(100 * 1024 * 1024 // 4, device="cuda")
        print("   ✅ 能夠分配 100MB")
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        print(f"   ❌ 無法分配記憶體：{e}")
        return False


def provide_troubleshooting():
    """提供故障排除建議"""
    print("\n🔧 故障排除建議：")
    print("1. 如果記憶體仍被佔用：")
    print("   - 重啟 Jupyter Notebook/Python 環境")
    print("   - 重啟終端機")
    print("   - 重啟電腦（最徹底）")
    print()
    print("2. 運行訓練前：")
    print("   - 關閉瀏覽器和其他程式")
    print("   - 運行這個腳本清理記憶體")
    print("   - 使用記憶體安全版本：finetune_Breeze_memory_safe.py")
    print()
    print("3. 如果問題持續：")
    print("   - 降低批次大小到 1")
    print("   - 減少資料集大小")
    print("   - 考慮使用 CPU 訓練")


def main():
    print("=== GPU 記憶體重設工具 ===\n")

    # 執行清理
    force_memory_cleanup()

    # 檢查進程
    check_gpu_processes()

    # 測試分配
    success = test_memory_allocation()

    # 提供建議
    provide_troubleshooting()

    # 總結
    print("\n=== 總結 ===")
    if success:
        print("✅ GPU 記憶體已清理，可以開始訓練")
        print("建議使用：python finetune_Breeze_memory_safe.py")
    else:
        print("❌ GPU 記憶體問題未解決")
        print("建議重啟 Python 環境後再試")


if __name__ == "__main__":
    main()
