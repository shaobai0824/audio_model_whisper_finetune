# ==============================================================================
# 檔案：finetune_Breeze_final_fix.py
# 描述：修復 FP16 梯度問題的最終穩定版本
# 核心策略：
# 1. 使用 FP32 精度避免梯度縮放問題
# 2. 保持極小批次大小確保記憶體安全
# 3. 優化的訓練配置確保穩定性
# ==============================================================================

import gc
import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import pandas as pd
import torch

# 設定環境變數
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datasets import Audio, Dataset

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
# 記憶體管理工具
# ==============================================================================


def cleanup_memory():
    """記憶體清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 記憶體清理完成")


def check_memory():
    """檢查記憶體使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(
            f"💾 記憶體：{allocated:.2f}GB 分配 / {reserved:.2f}GB 保留 / {total:.2f}GB 總計"
        )


# ==============================================================================
# 資料處理組件
# ==============================================================================


@dataclass
class SimpleDataCollator:
    """簡化的 Data Collator"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 處理輸入特徵
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 處理標籤
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 移除 BOS token
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset_simple(batch, feature_extractor, tokenizer):
    """簡化的資料預處理"""
    audio_list = batch["audio"]

    # 處理音訊特徵
    input_features = feature_extractor(
        [x["array"] for x in audio_list], sampling_rate=audio_list[0]["sampling_rate"]
    ).input_features

    # 處理標籤
    labels = tokenizer(
        batch["transcription"], max_length=448, truncation=True
    ).input_ids

    return {"input_features": input_features, "labels": labels}


def compute_metrics_simple(pred, tokenizer):
    """簡化的指標計算"""
    try:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        metric = evaluate.load("wer")
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    except Exception as e:
        print(f"⚠️ 指標計算錯誤：{e}")
        return {"wer": 100.0}


# ==============================================================================
# 資料集處理器
# ==============================================================================


class SimpleDatasetProcessor:
    """簡化的資料集處理器"""

    def __init__(
        self,
        file_path: str,
        target_sampling_rate: int = 16000,
        subset_fraction: float = 0.01,
    ):
        self.file_path = file_path
        self.target_sampling_rate = target_sampling_rate
        self.subset_fraction = subset_fraction

    def create_dataset(self) -> Dataset:
        print(f"載入資料檔案：{self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("❌ 找不到檔案，嘗試使用備用路徑...")
            alternative_path = "output/final_audio_paths_zh.csv"
            df = pd.read_csv(alternative_path)
            print(f"✅ 使用備用檔案：{alternative_path}")

        # 使用極小的資料子集
        subset_size = max(20, int(len(df) * self.subset_fraction))  # 至少 20 個樣本
        print(f"完整資料集大小：{len(df)}")
        print(f"使用資料集大小：{subset_size} ({(subset_size/len(df)*100):.1f}%)")

        # 隨機取樣
        subset_data = df.sample(n=subset_size, random_state=42).reset_index(drop=True)

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
    print("=== Breeze ASR 最終修復版本 ===")
    print("🔧 修復 FP16 梯度問題")

    # --- 參數設定 ---
    CSV_PATH = "output_zh_optimized_v2.csv"
    MODEL_NAME = "MediaTek-Research/Breeze-ASR-25"
    LANGUAGE = "zh"
    TASK = "transcribe"
    OUTPUT_DIR = "./whisper-small-zh-finetune-final"

    print(f"模型：{MODEL_NAME}")
    print(f"輸出目錄：{OUTPUT_DIR}")

    # 初始清理
    cleanup_memory()
    check_memory()

    try:
        # --- 載入 Processor 和模型 ---
        print("\n--- 步驟 1/4: 載入 Processor 和模型 ---")
        processor = WhisperProcessor.from_pretrained(
            MODEL_NAME, language=LANGUAGE, task=TASK
        )
        print("✅ Processor 載入成功")

        # 載入模型（使用 FP32 避免梯度問題）
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # 使用 FP32 避免梯度縮放問題
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("✅ 模型載入成功（FP32 模式）")

        # 配置模型
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        check_memory()

        # --- 建立資料集 ---
        print("\n--- 步驟 2/4: 建立資料集 (1% 資料) ---")
        audio_processor = SimpleDatasetProcessor(
            file_path=CSV_PATH, subset_fraction=0.01  # 使用 1% 的資料
        )

        dataset = audio_processor.create_dataset()
        print(f"資料集建立完成，樣本數：{len(dataset)}")

        # 分割訓練和測試集
        common_voice = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"訓練集：{len(common_voice['train'])} 樣本")
        print(f"測試集：{len(common_voice['test'])} 樣本")

        # 設定即時轉換
        prepare_fn = partial(
            prepare_dataset_simple,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
        )
        vectorized_datasets = common_voice.with_transform(prepare_fn)
        print("即時轉換已設定")

        check_memory()

        # --- 建立訓練元件 ---
        print("\n--- 步驟 3/4: 建立訓練元件 ---")
        data_collator = SimpleDataCollator(processor=processor)
        compute_metrics_fn = partial(
            compute_metrics_simple, tokenizer=processor.tokenizer
        )

        # 穩定的訓練參數（關鍵：不使用 FP16）
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            # 小批次配置
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            # 關閉多進程
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            # 學習參數
            learning_rate=1e-5,
            warmup_steps=5,
            max_steps=50,  # 極少步數確保成功
            # 關鍵：不使用 FP16 以避免梯度縮放問題
            fp16=False,  # 設為 False 避免梯度問題
            bf16=False,  # 也不使用 bf16
            # 簡化的評估和保存
            eval_strategy="steps",
            predict_with_generate=True,
            generation_max_length=128,
            save_steps=25,
            eval_steps=25,
            logging_steps=5,
            # 關閉不必要的功能
            report_to=[],
            load_best_model_at_end=False,
            save_total_limit=1,
            # 其他設定
            remove_unused_columns=False,
            optim="adamw_torch",
            gradient_checkpointing=False,
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

        check_memory()

        # --- 開始訓練 ---
        print("\n--- 步驟 4/4: 開始最終訓練 ---")
        print("🚀 預期訓練時間：3-5 分鐘")
        print("💡 使用 FP32 精度確保穩定性")

        cleanup_memory()
        check_memory()

        # 開始訓練
        trainer.train()
        print("\n✅ 最終訓練完成！")

        # --- 儲存模型 ---
        print("\n--- 儲存模型 ---")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"模型已儲存至：{OUTPUT_DIR}")

        # --- 訓練摘要 ---
        print("\n=== 最終訓練摘要 ===")
        print(f"✅ 成功完成訓練！")
        print(f"使用資料量：1% ({len(dataset)} 樣本)")
        print(f"訓練步數：50 步")
        print(f"精度模式：FP32（穩定）")
        print(f"批次設定：batch_size=1, accumulation=4")
        print("狀態：無錯誤完成")

    except Exception as e:
        print(f"\n❌ 訓練失敗：{e}")
        print("\n🔧 故障排除：")
        print("1. 確保沒有其他程式使用 GPU")
        print("2. 重啟 Python 環境")
        print("3. 檢查資料檔案路徑")
        cleanup_memory()
        raise e

    finally:
        cleanup_memory()
        print("🧹 最終清理完成")


if __name__ == "__main__":
    main()
