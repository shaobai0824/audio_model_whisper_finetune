import os

# 台羅音調字母
TAILO_TONES = [
    "ā",
    "á",
    "à",
    "â",
    "a̍",
    "ē",
    "é",
    "è",
    "ê",
    "e̍",
    "ī",
    "í",
    "ì",
    "i̍",
    "ō",
    "ó",
    "ò",
    "ô",
    "o̍",
    "ū",
    "ú",
    "ù",
    "u̍",
]
FOLDER = "audio_files"

files = [f for f in os.listdir(FOLDER) if os.path.isfile(os.path.join(FOLDER, f))]
to_delete = []

for f in files:
    name, _ = os.path.splitext(f)
    if any(tone in name for tone in TAILO_TONES):
        to_delete.append(f)

print("將刪除以下檔案：", to_delete)

for f in to_delete:
    try:
        os.remove(os.path.join(FOLDER, f))
    except Exception as e:
        print(f"刪除失敗：{f}，原因：{e}")
