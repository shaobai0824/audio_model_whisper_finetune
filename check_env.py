import sys
import os

print("--- 環境檢測報告 ---")

# 打印出當前執行這段程式碼的 Python 直譯器的完整路徑
print(f"[*] 當前使用的 Python 直譯器 (sys.executable):\n    {sys.executable}\n")

# 打印出當前 Python 的模組搜尋路徑
print("[*] 當前模組的搜尋路徑 (sys.path):")
for path in sys.path:
    print(f"    {path}")

print("\n--- 檢測完畢 ---")