import os
import random
import time

import pandas as pd
import requests

input_path = "output/final_audio_paths_numTune.csv"
output_path = "output/final_audio_paths_numTune_zh.csv"
temp_path = "output/final_audio_paths_numTune_zh_temp.csv"
batch_size = 1000


def tai_lo_to_chinese(tai_lo, retry=3):
    url = "https://itaigi.tw/api/taigi/taibun"
    params = {"taibun": tai_lo}
    for _ in range(retry):
        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            if data and isinstance(data, list) and "phiau_ji" in data[0]:
                return data[0]["phiau_ji"]
            else:
                return "查無翻譯"
        except Exception:
            time.sleep(random.uniform(1, 2))
    return "查無翻譯"


# 讀取已完成的暫存檔
if os.path.exists(temp_path):
    done_df = pd.read_csv(temp_path)
    done_count = len(done_df)
else:
    done_df = pd.DataFrame(columns=["transcription", "file"])
    done_count = 0

df = pd.read_csv(input_path, names=["transcription", "file"])
total = len(df)

for start in range(done_count, total, batch_size):
    end = min(start + batch_size, total)
    batch = df.iloc[start:end].copy()
    translated = []
    for rel_idx, (idx, row) in enumerate(batch.iterrows()):
        chinese = tai_lo_to_chinese(row["transcription"])
        translated.append(chinese)
        print(
            f"第{start + rel_idx + 1}/{total}筆 | 台羅: {row['transcription']} | 中文: {chinese[:20]}"
        )
        time.sleep(random.uniform(1, 2))
    batch["transcription"] = translated
    done_df = pd.concat([done_df, batch], ignore_index=True)
    done_df.to_csv(temp_path, index=False)
    print(f"已完成 {end}/{total} 筆，暫存於 {temp_path}")

done_df.to_csv(output_path, index=False)
print(f"全部完成，結果儲存於 {output_path}")
