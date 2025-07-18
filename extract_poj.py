import csv
import re
import shutil


def extract_poj_content(text):
    """
    有 [POJ]...[/POJ] 則取出並去除 "，否則保留原始內容並去除 "。
    """
    if not text:
        return ""
    match = re.search(r"\[POJ\](.*?)\[/POJ\]", text, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = text
    return result.replace('"', "")


def process_csv_extract_poj(csv_file, column):
    # 備份原始檔
    shutil.copyfile(csv_file, csv_file + ".bak")

    rows = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            return
        fieldnames = reader[0].keys()
        for row in reader:
            value = row.get(column, "")
            row[column] = extract_poj_content(value)
            rows.append(row)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "已完成：[POJ]...[/POJ] 內容提取（無則保留原始內容），並去除所有引號，原始檔已備份為 .bak"
    )


if __name__ == "__main__":
    # 處理 output_zh_optimized_v2.csv 的中文意譯欄位（假設是第3個欄位）
    process_csv_extract_poj("output_zh_optimized_v2.csv", "中文意譯")
