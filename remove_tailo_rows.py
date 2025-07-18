import csv
import re
import shutil


def contains_tailo_vowels(text):
    """
    檢查文字是否包含台羅拼音母音（帶聲調符號）
    """
    if not text:
        return False

    # 台羅拼音的所有帶聲調母音
    tailo_vowels = [
        # 第2聲（上揚調）
        "á",
        "é",
        "í",
        "ó",
        "ú",
        "ḿ",
        "ń",
        # 第3聲（上聲）
        "à",
        "è",
        "ì",
        "ò",
        "ù",
        "m̀",
        "ǹ",
        # 第5聲（陽平調）
        "â",
        "ê",
        "î",
        "ô",
        "û",
        "m̂",
        "n̂",
        # 第7聲（陽去調）
        "ā",
        "ē",
        "ī",
        "ō",
        "ū",
        "m̄",
        "n̄",
        # 第8聲（入聲）
        "a̍",
        "e̍",
        "i̍",
        "o̍",
        "u̍",
        "m̍",
        "n̍",
        # 其他特殊符號
        "ńg",
        "ǹg",
        "n̂g",
        "n̄g",
        "ńg",
    ]

    # 檢查是否包含任何台羅母音
    for vowel in tailo_vowels:
        if vowel in text:
            return True

    return False


def remove_tailo_rows(csv_file, column):
    """
    刪除指定欄位包含台羅拼音母音的整列
    """
    print(f"=== 刪除台羅拼音行 ===")
    print(f"處理檔案: {csv_file}")
    print(f"檢查欄位: {column}")

    # 備份原始檔案
    backup_file = csv_file + ".bak_before_tailo_removal"
    if not os.path.exists(backup_file):
        try:
            shutil.copyfile(csv_file, backup_file)
            print(f"✓ 已備份原始檔案為: {backup_file}")
        except Exception as e:
            print(f"✗ 備份檔案失敗: {e}")
            return False
    else:
        print(f"⚠ 備份檔案已存在: {backup_file}")

    try:
        # 讀取檔案
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            if not fieldnames:
                print("✗ 無法讀取檔案標題行")
                return False

            if column not in fieldnames:
                print(f"✗ 找不到欄位 '{column}'")
                print(f"可用欄位: {list(fieldnames)}")
                return False

            # 處理資料
            kept_rows = []
            removed_rows = []
            total_rows = 0

            for row in reader:
                total_rows += 1
                content = row.get(column, "")

                if contains_tailo_vowels(content):
                    removed_rows.append(row)
                    # 顯示前幾個被刪除的範例
                    if len(removed_rows) <= 5:
                        print(f"刪除範例 {len(removed_rows)}: {content[:60]}...")
                else:
                    kept_rows.append(row)

        # 寫入處理後的檔案
        print(f"\n開始寫入處理結果...")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(kept_rows)

        # 顯示處理結果
        print(f"\n=== 處理完成 ===")
        print(f"✓ 原始總行數: {total_rows}")
        print(f"✓ 保留行數: {len(kept_rows)}")
        print(f"✓ 刪除行數: {len(removed_rows)}")
        print(f"✓ 刪除比例: {len(removed_rows)/total_rows*100:.2f}%")
        print(f"✓ 檔案已更新: {csv_file}")
        print(f"✓ 原始檔備份: {backup_file}")

        return True

    except Exception as e:
        print(f"✗ 處理過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import os

    csv_file = "output_zh_optimized_v2.csv"
    target_column = "中文意譯"

    # 檢查檔案是否存在
    if not os.path.exists(csv_file):
        print(f"✗ 檔案不存在: {csv_file}")
        exit(1)

    file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
    print(f"檔案大小: {file_size_mb:.2f} MB")

    # 執行處理
    success = remove_tailo_rows(csv_file, target_column)

    if success:
        print("\n🎉 台羅拼音行刪除完成!")

        # 顯示處理後檔案大小
        new_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
        print(f"處理後檔案大小: {new_size_mb:.2f} MB")
        print(f"節省空間: {file_size_mb - new_size_mb:.2f} MB")
    else:
        print("\n❌ 處理失敗!")
