#!/usr/bin/env python3
"""
修正 CSV 檔案中的路徑以適用於 Google Colab 環境
將本地絕對路徑轉換為 Colab 相對路徑或 Google Drive 路徑
"""

import os
import re
from pathlib import Path

import pandas as pd


def detect_path_type(file_path):
    """偵測路徑類型"""
    if file_path.startswith(("C:", "D:", "E:", "F:")):
        return "windows_absolute"
    elif file_path.startswith("/content/drive/"):
        return "colab_drive"
    elif file_path.startswith("/content/"):
        return "colab_content"
    elif file_path.startswith("/"):
        return "linux_absolute"
    else:
        return "relative"


def fix_paths_for_colab(csv_file, output_file=None, audio_base_path=None):
    """
    修正 CSV 檔案中的路徑以適用於 Colab

    Args:
        csv_file: 輸入 CSV 檔案路徑
        output_file: 輸出 CSV 檔案路徑（如果為 None，則覆蓋原檔案）
        audio_base_path: 音訊檔案在 Colab 中的基礎路徑
    """

    # 設定預設的 Colab 路徑
    if audio_base_path is None:
        audio_base_path = "/content/drive/MyDrive/audio_model/audio_files"

    print(f"🔧 修正 CSV 路徑: {csv_file}")
    print(f"📁 目標基礎路徑: {audio_base_path}")

    # 讀取 CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ 成功讀取 CSV，共 {len(df)} 筆資料")
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
        return False

    # 檢查檔案路徑欄位
    path_columns = []
    for col in df.columns:
        if "file" in col.lower() or "path" in col.lower() or "audio" in col.lower():
            path_columns.append(col)

    if not path_columns:
        print("⚠️  未找到包含路徑的欄位，嘗試使用第一個欄位")
        path_columns = [df.columns[0]]

    print(f"🎯 處理路徑欄位: {path_columns}")

    # 修正每個路徑欄位
    for col in path_columns:
        print(f"\n處理欄位: {col}")

        fixed_paths = []
        path_stats = {
            "windows_absolute": 0,
            "colab_drive": 0,
            "colab_content": 0,
            "linux_absolute": 0,
            "relative": 0,
            "fixed": 0,
            "errors": 0,
        }

        for idx, file_path in enumerate(df[col]):
            try:
                if pd.isna(file_path):
                    fixed_paths.append(file_path)
                    continue

                original_path = str(file_path).strip()
                path_type = detect_path_type(original_path)
                path_stats[path_type] += 1

                # 根據路徑類型進行修正
                if path_type == "windows_absolute":
                    # Windows 路徑 -> Colab 路徑
                    filename = Path(original_path).name
                    new_path = f"{audio_base_path}/{filename}"
                    fixed_paths.append(new_path)
                    path_stats["fixed"] += 1

                    if idx < 5:  # 顯示前 5 個範例
                        print(f"   修正: {original_path} -> {new_path}")

                elif path_type == "linux_absolute":
                    # Linux 絕對路徑 -> 檢查是否需要調整
                    if original_path.startswith(
                        "/content/drive/"
                    ) or original_path.startswith("/content/"):
                        # 已經是 Colab 路徑
                        fixed_paths.append(original_path)
                    else:
                        # 其他 Linux 路徑 -> 修正為 Colab 路徑
                        filename = Path(original_path).name
                        new_path = f"{audio_base_path}/{filename}"
                        fixed_paths.append(new_path)
                        path_stats["fixed"] += 1

                        if idx < 5:
                            print(f"   修正: {original_path} -> {new_path}")

                elif path_type == "relative":
                    # 相對路徑 -> 轉為絕對路徑
                    if not original_path.startswith(
                        "./"
                    ) and not original_path.startswith("../"):
                        new_path = f"{audio_base_path}/{original_path}"
                    else:
                        # 處理 ./ 和 ../ 路徑
                        clean_path = original_path.replace("./", "").replace("../", "")
                        new_path = f"{audio_base_path}/{clean_path}"

                    fixed_paths.append(new_path)
                    path_stats["fixed"] += 1

                    if idx < 5:
                        print(f"   修正: {original_path} -> {new_path}")

                else:
                    # 已經是正確的 Colab 路徑
                    fixed_paths.append(original_path)

            except Exception as e:
                print(f"   ❌ 處理路徑失敗 (第 {idx+1} 行): {e}")
                fixed_paths.append(original_path)  # 保留原路徑
                path_stats["errors"] += 1

        # 更新 DataFrame
        df[col] = fixed_paths

        # 顯示統計
        print(f"\n📊 路徑修正統計 ({col}):")
        for path_type, count in path_stats.items():
            if count > 0:
                print(f"   {path_type}: {count}")

    # 保存修正後的 CSV
    output_file = output_file or csv_file
    try:
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"\n✅ 修正完成，已保存至: {output_file}")
        return True
    except Exception as e:
        print(f"\n❌ 保存失敗: {e}")
        return False


def validate_colab_paths(csv_file, check_existence=False):
    """驗證 CSV 中的路徑在 Colab 環境中是否有效"""

    print(f"🔍 驗證 Colab 路徑: {csv_file}")

    df = pd.read_csv(csv_file)

    # 找到路徑欄位
    path_columns = []
    for col in df.columns:
        if "file" in col.lower() or "path" in col.lower() or "audio" in col.lower():
            path_columns.append(col)

    if not path_columns:
        path_columns = [df.columns[0]]

    validation_results = {}

    for col in path_columns:
        print(f"\n驗證欄位: {col}")

        valid_paths = 0
        invalid_paths = 0
        missing_files = 0

        for file_path in df[col][:10]:  # 檢查前 10 個路徑
            if pd.isna(file_path):
                continue

            path_str = str(file_path).strip()

            # 檢查路徑格式
            if (
                path_str.startswith("/content/")
                or path_str.startswith("./")
                or not path_str.startswith(("C:", "D:", "E:"))
            ):
                valid_paths += 1

                # 可選：檢查檔案是否實際存在
                if check_existence and os.path.exists(path_str):
                    print(f"   ✅ 存在: {path_str}")
                elif check_existence:
                    missing_files += 1
                    print(f"   ❌ 不存在: {path_str}")
            else:
                invalid_paths += 1
                print(f"   ⚠️  可能無效: {path_str}")

        validation_results[col] = {
            "valid": valid_paths,
            "invalid": invalid_paths,
            "missing": missing_files,
        }

    return validation_results


def create_sample_colab_setup():
    """創建範例 Colab 設定代碼"""

    setup_code = """
# Google Colab 路徑設定範例

# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 設定專案目錄
import os
project_dir = "/content/drive/MyDrive/audio_model"
os.chdir(project_dir)

# 3. 檢查音訊檔案目錄
audio_dir = os.path.join(project_dir, "audio_files")
if os.path.exists(audio_dir):
    print(f"✅ 音訊目錄存在: {audio_dir}")
    print(f"檔案數量: {len(os.listdir(audio_dir))}")
else:
    print(f"❌ 音訊目錄不存在: {audio_dir}")
    print("請確認音訊檔案已上傳到正確位置")

# 4. 修正 CSV 路徑
!python fix_csv_paths_for_colab.py --csv metadata_train.csv --base-path /content/drive/MyDrive/audio_model/audio_files
!python fix_csv_paths_for_colab.py --csv metadata_test.csv --base-path /content/drive/MyDrive/audio_model/audio_files
"""

    with open("colab_setup_example.py", "w", encoding="utf-8") as f:
        f.write(setup_code)

    print("📝 已創建 colab_setup_example.py")


def main():
    """主函數 - 提供命令行介面"""
    import argparse

    parser = argparse.ArgumentParser(description="修正 CSV 路徑以適用於 Google Colab")
    parser.add_argument("--csv", required=True, help="輸入 CSV 檔案路徑")
    parser.add_argument("--output", help="輸出 CSV 檔案路徑（預設覆蓋原檔案）")
    parser.add_argument(
        "--base-path",
        default="/content/drive/MyDrive/audio_model/audio_files",
        help="音訊檔案在 Colab 中的基礎路徑",
    )
    parser.add_argument("--validate", action="store_true", help="驗證修正後的路徑")
    parser.add_argument(
        "--check-files", action="store_true", help="檢查檔案是否實際存在"
    )

    args = parser.parse_args()

    # 修正路徑
    success = fix_paths_for_colab(args.csv, args.output, args.base_path)

    if success and args.validate:
        # 驗證路徑
        validate_colab_paths(args.output or args.csv, args.check_files)

    # 創建設定範例
    create_sample_colab_setup()


if __name__ == "__main__":
    # 如果沒有命令行參數，提供互動式介面
    import sys

    if len(sys.argv) == 1:
        print("🔧 CSV 路徑修正工具 - Colab 版本")
        print("=" * 50)

        csv_file = input("請輸入 CSV 檔案路徑: ").strip()
        if not csv_file:
            print("❌ 請提供有效的 CSV 檔案路徑")
            exit(1)

        base_path = input(
            "音訊檔案基礎路徑 [/content/drive/MyDrive/audio_model/audio_files]: "
        ).strip()
        if not base_path:
            base_path = "/content/drive/MyDrive/audio_model/audio_files"

        print("\n開始修正...")
        success = fix_paths_for_colab(csv_file, None, base_path)

        if success:
            validate = input("\n是否驗證修正結果? (y/n): ").strip().lower()
            if validate == "y":
                validate_colab_paths(csv_file)

        create_sample_colab_setup()
    else:
        main()
