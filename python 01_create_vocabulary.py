import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# --- 1. 配置 (Configuration) ---
DATA_DIR = Path("./")
OUTPUT_DIR = Path("./model_outputs")

# 輸入：單一的、包含所有資料的 metadata 檔案名稱
MASTER_METADATA_FILE = "output_zh_optimized_v2.csv"

# 輸出：切分後的訓練與測試 metadata 檔案名稱
TRAIN_METADATA_FILE = "metadata_train.csv"
TEST_METADATA_FILE = "metadata_test.csv"

# 詞彙表檔案名稱
VOCAB_FILE = "vocab.json"

# 切分參數
TEST_SIZE: float = 0.2  # 20% 的資料作為測試集
RANDOM_STATE: int = 42  # 固定隨機種子以確保可複現性

# (可選) 用於分層抽樣的欄位名稱，例如 'speaker_id'
# 如果您的 metadata 中沒有此欄位，請設為 None
STRATIFY_COLUMN: Optional[str] = None


def prepare_data_and_vocab(
    data_dir: Path,
    output_dir: Path,
    master_file: str,
    train_file: str,
    test_file: str,
    vocab_file: str,
    test_size: float,
    random_state: int,
    stratify_col: Optional[str],
) -> None:
    """
    讀取主 metadata 檔案，切分訓練/測試集，並建立詞彙表。
    """
    # --- 步驟 1: 讀取主 metadata 檔案 ---
    master_path = data_dir / master_file
    if not master_path.exists():
        print(f"錯誤：主元數據檔案 {master_path} 不存在！")
        return

    print(f"正在讀取主元數據檔案: {master_path}")
    df = pd.read_csv(master_path)

    # --- 步驟 2: 切分訓練集與測試集 ---
    print(f"正在以 {1-test_size:.0%}/{test_size:.0%} 的比例切分資料...")

    stratify_data = (
        df[stratify_col] if stratify_col and stratify_col in df.columns else None
    )

    if stratify_data is not None:
        print(f"採用依據 '{stratify_col}' 欄位的分層抽樣。")
    else:
        print("採用簡單隨機抽樣。")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_data
    )

    # --- 步驟 3: 儲存切分後的 metadata 檔案 ---
    train_output_path = data_dir / train_file
    test_output_path = data_dir / test_file

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    print(f"訓練集元數據已儲存至: {train_output_path} ({len(train_df)} 筆)")
    print(f"測試集元數據已儲存至: {test_output_path} ({len(test_df)} 筆)")

    # --- 步驟 4: 建立詞彙表 (使用完整的資料集以確保詞彙完整性) ---
    print("正在從完整的資料集中建立詞彙表...")

    # 確保 transcription 欄位為 string 型態
    all_text = " ".join(df["中文意譯"].astype(str).tolist())

    char_counts = Counter(all_text)
    vocab_list = sorted(char_counts.keys())
    vocab_dict = {char: i for i, char in enumerate(vocab_list)}

    # 加入 Wav2Vec2 需要的特殊 tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    vocab_dict["|"] = len(vocab_dict)

    output_dir.mkdir(exist_ok=True)
    vocab_output_path = output_dir / vocab_file
    with open(vocab_output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(
        f"詞彙表建立完成，共 {len(vocab_dict)} 個 tokens。已儲存至: {vocab_output_path}"
    )
    print("\n資料準備完成！")


if __name__ == "__main__":
    prepare_data_and_vocab(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        master_file=MASTER_METADATA_FILE,
        train_file=TRAIN_METADATA_FILE,
        test_file=TEST_METADATA_FILE,
        vocab_file=VOCAB_FILE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify_col=STRATIFY_COLUMN,
    )
