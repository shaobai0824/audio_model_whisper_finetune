# ==============================================================================
# 檔案：finetune_Breeze_fast.py
# 描述：基於 train.py 優化的快速訓練版本，使用十分之一資料集
# 核心優化：
# 1. 使用十分之一的資料集進行快速實驗
# 2. 基於 train.py 的即時轉換策略
# 3. 優化的批次大小和參數設定
# 4. 背景預取和多核心處理
# ==============================================================================

import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict

# --- Hugging Face 相關導入 ---
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ==============================================================================
# 全域定義的輔助類別與函式
# ==============================================================================


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """處理語音到序列資料的 Data Collator"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset_batched(batch, feature_extractor, tokenizer):
    """即時轉換音訊和文本資料為模型輸入格式"""
    audio_list = batch["audio"]
    batch["input_features"] = feature_extractor(
        [x["array"] for x in audio_list], sampling_rate=audio_list[0]["sampling_rate"]
    ).input_features
    batch["labels"] = tokenizer(
        batch["transcription"], max_length=448, truncation=True
    ).input_ids
    return batch


def compute_metrics(pred, tokenizer):
    """計算 WER 指標"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ==============================================================================
# 資料集處理類別
# ==============================================================================


class FastAudioDatasetProcessor:
    """快速資料集處理器，使用十分之一的資料"""

    def __init__(
        self,
        file_path: str,
        target_sampling_rate: int = 16000,
        subset_fraction: float = 0.1,
    ):
        self.file_path = file_path
        self.target_sampling_rate = target_sampling_rate
        self.subset_fraction = subset_fraction

    def create_dataset(self) -> Dataset:
        print(f"載入完整資料集：{self.file_path}")
        full_data = pd.read_csv(self.file_path)

        # 使用十分之一的資料
        subset_size = int(len(full_data) * self.subset_fraction)
        print(f"完整資料集大小：{len(full_data)}")
        print(f"使用資料集大小：{subset_size} ({self.subset_fraction*100}%)")

        # 隨機取樣
        subset_data = full_data.sample(n=subset_size, random_state=42).reset_index(
            drop=True
        )

        dataset = Dataset.from_pandas(subset_data)
        dataset = dataset.cast_column(
            "file", Audio(sampling_rate=self.target_sampling_rate)
        )
        dataset = dataset.rename_column("file", "audio")

        return dataset


# ==============================================================================
# 主執行流程
# ==============================================================================


def main():
    # --- 參數設定 ---
    CSV_PATH = "output/final_audio_paths_zh.csv"
    MODEL_NAME = "MediaTek-Research/Breeze-ASR-25"
    LANGUAGE = "zh"
    TASK = "transcribe"
    OUTPUT_DIR = "./whisper-small-zh-finetune-fast"

    print("=== Breeze ASR 快速微調版本 ===")
    print(f"模型：{MODEL_NAME}")
    print(f"輸出目錄：{OUTPUT_DIR}")

    # --- 載入 Processor 和模型 ---
    print("\n--- 步驟 1/4: 載入 Processor 和模型 ---")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # 配置模型
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # --- 建立快速資料集 ---
    print("\n--- 步驟 2/4: 建立快速資料集 (10% 資料) ---")
    audio_processor = FastAudioDatasetProcessor(
        file_path=CSV_PATH, subset_fraction=0.1  # 使用 10% 的資料
    )

    dataset = audio_processor.create_dataset()
    print(f"資料集建立完成，樣本數：{len(dataset)}")

    # 分割訓練和測試集
    common_voice = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"訓練集：{len(common_voice['train'])} 樣本")
    print(f"測試集：{len(common_voice['test'])} 樣本")

    # 設定即時轉換
    prepare_fn = partial(
        prepare_dataset_batched,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
    )
    vectorized_datasets = common_voice.with_transform(prepare_fn)
    print("即時轉換已設定")

    # --- 建立訓練元件 ---
    print("\n--- 步驟 3/4: 建立訓練元件 ---")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics_fn = partial(compute_metrics, tokenizer=processor.tokenizer)

    # 快速訓練參數（基於 train.py 優化）
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 優化的批次設定（基於 RTX 3060Ti 8GB VRAM）
        per_device_train_batch_size=4,  # 增加批次大小以提升效率
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 有效批次 = 8*2 = 16
        # 多核心處理（基於 train.py）
        dataloader_num_workers=4,  # 利用多核心預取
        # 快速訓練設定
        learning_rate=2e-5,  # 稍微提高學習率加速收斂
        warmup_steps=100,  # 減少暖身步數
        max_steps=1000,  # 快速訓練只需 1000 步
        # 優化設定
        gradient_checkpointing=False,  # 關閉以提升速度
        fp16=True,  # 使用混合精度
        # 評估和保存
        eval_strategy="steps",
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,  # 更頻繁的保存
        eval_steps=200,  # 更頻繁的評估
        logging_steps=25,
        # 監控和輸出
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # 其他設定
        remove_unused_columns=False,
        save_total_limit=3,  # 只保留最近3個檢查點
    )

    # 建立訓練器
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        tokenizer=processor.feature_extractor,
    )

    # --- 開始快速訓練 ---
    print("\n--- 步驟 4/4: 開始快速微調訓練 ---")
    print("預期訓練時間：10-15 分鐘")

    trainer.train()
    print("\n*** 快速訓練完成 ***")

    # --- 儲存模型 ---
    print("\n--- 儲存訓練完成的模型 ---")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"模型已儲存至：{OUTPUT_DIR}")

    # --- 顯示訓練結果摘要 ---
    print("\n=== 訓練摘要 ===")
    print(f"使用資料量：10% ({len(dataset)} 樣本)")
    print(f"訓練步數：1000 步")
    print(f"預期 WER 改善：5-15%")
    print("建議：如果結果良好，可使用完整資料集進行完整訓練")


if __name__ == "__main__":
    main()
