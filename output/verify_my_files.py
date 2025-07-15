# ==============================================================================
# 檔案：verify_my_files.py
# 描述：讀取原始 CSV，檢查每一行的檔案路徑是否存在。如果不存在，該行將被
#       「刪除」（即不寫入到新的檔案中），最終生成一個 100% 有效的乾淨 CSV。
# ==============================================================================
import os
import sys

import pandas as pd


def verify_and_clean_csv(csv_path: str, path_column: str) -> None:
    """
    驗證 CSV 中的檔案路徑，並生成一個只包含有效路徑的「乾淨版」新 CSV。
    """
    print(f"--- 開始驗證並清理檔案：{csv_path} ---")

    # --- 讀取原始 CSV ---
    try:
        df = pd.read_csv(csv_path)
        print(f"成功讀取檔案，共包含 {len(df)} 筆記錄。")
    except FileNotFoundError:
        print(f"[錯誤] 在目前目錄下找不到指定的 CSV 檔案：{csv_path}")
        return

    if path_column not in df.columns:
        print(f"[錯誤] CSV 檔案中找不到指定的路徑欄位：'{path_column}'")
        return

    # --- 逐一驗證路徑，只保留有效的 ---
    valid_rows = []
    missing_files_count = 0

    print(f"\n正在檢查每一筆記錄的路徑有效性... 請耐心等候。")

    total_rows = len(df)
    for index, row in df.iterrows():
        if (index + 1) % 5000 == 0:
            print(f"已檢查 {index + 1} / {total_rows} 筆記錄...")

        file_path = row[path_column]

        # 只有當路徑有效且檔案存在時，才將該行加入到我們的「通過名單」中
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            valid_rows.append(row)
        else:
            # 如果檔案不存在，則這行記錄會被「拋棄」，等同於被刪除
            missing_files_count += 1

    # --- 產生報告並儲存結果 ---
    print("\n--- 驗證完成 ---")

    if missing_files_count == 0:
        print(
            "[✅ 成功] 恭喜！所有 {total_rows} 筆記錄的路徑都有效，您的資料集非常乾淨，無需建立新檔案。"
        )
    else:
        print(f"[⚠️ 警告] 發現 {missing_files_count} 筆記錄的檔案路徑無效或不存在。")

        if valid_rows:
            cleaned_df = pd.DataFrame(valid_rows)
            base, ext = os.path.splitext(csv_path)
            output_path = f"{base}_cleaned{ext}"

            cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

            print(f"\n[✅ 成功] 一個全新的、乾淨的 CSV 檔案已被建立！")
            print(f"    -> 原始記錄數：{total_rows}")
            print(f"    -> 無效記錄數（已刪除）：{missing_files_count}")
            print(f"    -> 有效記錄數（已保留）：{len(cleaned_df)}")
            print(f"\n    新的檔案已儲存至：{output_path}")
            print(
                "\n    強烈建議您在下一次訓練時，將訓練腳本中的 CSV_PATH 修改為指向這個「_cleaned」結尾的新檔案。"
            )
        else:
            print("\n[錯誤] 沒有任何有效的檔案路徑，無法產生新的 CSV 檔案。")


def find_and_run_verification():
    """自動偵測存在的 CSV 檔案並執行驗證。"""
    # ... (此部分與之前相同) ...


if __name__ == "__main__":
    # 我們將直接指定要處理的檔案，以確保清晰
    INPUT_CSV_PATH = "final_audio_paths_zh.csv"
    PATH_COLUMN_NAME = "file"

    if os.path.exists(INPUT_CSV_PATH):
        verify_and_clean_csv(csv_path=INPUT_CSV_PATH, path_column=PATH_COLUMN_NAME)
    else:
        print(
            f"[致命錯誤] 找不到目標檔案 '{INPUT_CSV_PATH}'。請確認腳本與您的 CSV 檔案在同一個資料夾下。"
        )
