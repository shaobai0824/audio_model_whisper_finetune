import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# --- 1. 全域配置 (Global Configuration) ---
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
DATASET_DIR = Path("./")
TRAIN_CSV = DATASET_DIR / "metadata_train.csv"
TEST_CSV = DATASET_DIR / "metadata_test.csv"


# --- 2. 載入資料集 (Load Dataset) ---
def load_csv_dataset(csv_path: Path, sample_fraction=0.1):
    """載入 CSV 格式的資料集，並取樣指定比例的資料"""
    df = pd.read_csv(csv_path)

    # 隨機取樣十分之一的資料
    sample_size = int(len(df) * sample_fraction)
    df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"原始資料: {len(df)} 筆")
    print(f"取樣資料: {len(df_sampled)} 筆 ({sample_fraction*100:.1f}%)")

    return df_sampled


# --- 3. 資料前處理 (Data Preprocessing) ---
# 全域 processor 變數，用於多進程
_processor = None


def init_processor(proc):
    """初始化 processor 供多進程使用"""
    global _processor
    _processor = proc


def prepare_dataset(batch):
    """準備資料集，包括音訊載入和文字編碼"""
    import librosa
    import numpy as np

    # 載入音訊檔案
    audio_arrays = []
    for audio_path in batch["file"]:
        try:
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            audio_arrays.append(audio_array)
        except Exception as e:
            print(f"錯誤載入音訊檔案 {audio_path}: {e}")
            # 創建一個空的音訊數組作為替代
            audio_arrays.append(np.zeros(16000))

    # 處理音訊 - 使用 Whisper 的處理方式
    # 逐個處理音訊以確保形狀一致
    input_features_list = []
    for audio_array in audio_arrays:
        inputs = _processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        )
        # 將張量轉換為 numpy 再轉回 tensor 以避免梯度追蹤問題
        feature = inputs.input_features.squeeze(0).detach().cpu().numpy()
        input_features_list.append(feature)

    # 編碼文字標籤 - 使用 Whisper tokenizer
    # 確保文字是字串列表
    texts = [str(text) for text in batch["中文意譯"]]
    labels = _processor.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    )

    batch["input_features"] = input_features_list
    batch["labels"] = labels.input_ids.detach().cpu().numpy()

    return batch


# --- 4. 自訂 Data Collator (適用於 Whisper) ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    自訂的 Data Collator，用於 Whisper 語音到序列訓練
    """

    processor: WhisperProcessor
    decoder_start_token_id: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 處理 input_features - 直接使用 torch.stack 而不是 pad
        input_features = [feature["input_features"] for feature in features]

        # 確保所有 input_features 具有相同的形狀
        if isinstance(input_features[0], torch.Tensor):
            # 如果已經是 tensor，先 detach 再 stack 避免梯度問題
            batch_input_features = torch.stack([f.detach() for f in input_features])
        else:
            # 如果是 list，轉換為 tensor 後 stack
            batch_input_features = torch.stack(
                [torch.tensor(f, dtype=torch.float32) for f in input_features]
            )

        # 處理 labels
        label_features = [
            {"input_ids": torch.tensor(feature["labels"], dtype=torch.long)}
            for feature in features
        ]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 將 padding tokens 替換為 -100，這樣它們就不會被計算在損失中
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 如果 bos token 被添加到了開頭，我們需要切掉它
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch = {"input_features": batch_input_features, "labels": labels}

        return batch


# --- 5. 評估指標 (Evaluation Metrics) ---
def compute_metrics(pred):
    """計算 Word Error Rate (WER)"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 將 -100 替換為 pad token id
    label_ids[label_ids == -100] = _processor.tokenizer.pad_token_id

    # 解碼預測和標籤
    pred_str = _processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = _processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 計算 WER
    wer = jiwer.wer(label_str, pred_str)

    return {"wer": wer}


def main():
    """主要執行函數"""
    global _processor  # 讓 processor 在全域範圍內可用

    # Windows 多進程支援
    if __name__ == "__main__":
        multiprocessing.freeze_support()

    # 記憶體優化設定
    import os

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 清理 GPU 記憶體
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(
            f"GPU 記憶體已清理，可用記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print("=== Breeze-ASR-25 記憶體優化測試版本 ===")
    print("使用 10% 資料，極小 batch size 進行記憶體測試")
    print("開始載入資料集...")

    # 檢查檔案是否存在
    if not TRAIN_CSV.exists():
        print(f"錯誤：找不到訓練資料檔案 {TRAIN_CSV}")
        return
    if not TEST_CSV.exists():
        print(f"錯誤：找不到測試資料檔案 {TEST_CSV}")
        return

    # 載入訓練和測試資料（取樣 10%）
    train_df = load_csv_dataset(TRAIN_CSV, sample_fraction=0.1)
    test_df = load_csv_dataset(TEST_CSV, sample_fraction=0.1)

    print(f"訓練資料: {len(train_df)} 筆")
    print(f"測試資料: {len(test_df)} 筆")

    # 轉換為 datasets 格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 載入模型和處理器
    print("載入 Whisper 模型和處理器...")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,  # 使用 fp32 載入模型避免梯度問題
            low_cpu_mem_usage=True,  # 低 CPU 記憶體使用
        )

        # 將模型移到 GPU 並優化記憶體
        model = model.to("cuda")
        model.config.use_cache = False  # 關閉快取以節省記憶體

        print("模型和處理器載入成功！")
        print(
            f"模型載入後 GPU 記憶體: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return

    # 初始化全域 processor 供多進程使用
    init_processor(processor)

    # 應用資料前處理
    print("開始資料前處理...")
    try:
        train_dataset = train_dataset.map(
            prepare_dataset,
            remove_columns=train_dataset.column_names,
            batched=True,
            batch_size=8,  # 較小的批次大小以加速測試
        )
        test_dataset = test_dataset.map(
            prepare_dataset,
            remove_columns=test_dataset.column_names,
            batched=True,
            batch_size=8,  # 較小的批次大小以加速測試
        )
        print("資料前處理完成！")
    except Exception as e:
        print(f"資料前處理失敗: {e}")
        return

    # 創建 data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 訓練設定 - 極度節省記憶體的測試版本
    training_args = TrainingArguments(
        output_dir="./results_test",
        per_device_train_batch_size=1,  # 極小的 batch size 避免 OOM
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # 減少梯度累積以節省記憶體
        eval_strategy="steps",
        num_train_epochs=1,  # 只訓練 1 個 epoch 以快速測試
        fp16=True,
        gradient_checkpointing=False,  # 關閉梯度檢查點避免 Whisper 相容性問題
        save_steps=50,  # 更頻繁的保存以便觀察進度
        eval_steps=50,  # 更頻繁的評估
        logging_steps=5,  # 更頻繁的日誌輸出
        learning_rate=1e-3,  # SGD 通常需要更高的學習率
        weight_decay=0.01,
        warmup_steps=20,  # 較少的預熱步數
        save_total_limit=1,  # 只保留一個檢查點
        push_to_hub=False,
        dataloader_num_workers=0,  # Windows 上設為 0 避免多進程問題
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=False,  # 關閉以節省記憶體
        # 記憶體優化設定
        dataloader_pin_memory=False,  # 關閉 pin memory
        optim="sgd",  # 使用 SGD 替代 Adam 以節省記憶體
        lr_scheduler_type="linear",  # 使用更簡單的調度器
        report_to=None,
        max_steps=50,  # 大幅減少步數以快速測試
        # 額外的記憶體優化
        ddp_find_unused_parameters=False,
    )

    # 創建 Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=processor,
    )

    # 訓練前最後一次記憶體清理
    torch.cuda.empty_cache()
    print(f"訓練前 GPU 記憶體使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 開始訓練
    print("開始 Breeze-ASR-25 記憶體優化測試...")
    print(
        f"預計訓練步數: {min(50, len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))}"
    )
    try:
        trainer.train()
        print("記憶體優化測試完成！")

        # 保存模型
        print("保存測試模型...")
        trainer.save_model("./breeze-asr-25-test")
        processor.save_pretrained("./breeze-asr-25-test")
        print("模型保存完成！")

        # 進行最終評估
        print("進行最終評估...")
        eval_results = trainer.evaluate()
        print(f"最終 WER: {eval_results['eval_wer']:.4f}")

    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
