# ==============================================================================
# 檔案：finetune_Breeze_memory_safe.py
# 描述：記憶體安全的訓練版本，徹底解決 CUDA OOM 問題
# 核心策略：
# 1. 極小批次大小 + 大梯度累積
# 2. 強制記憶體清理和碎片整理
# 3. 優化的資料載入和模型配置
# 4. 錯誤恢復機制
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

# 設定環境變數以優化記憶體管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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


def aggressive_memory_cleanup():
    """激進的記憶體清理"""
    import gc

    # Python 垃圾回收
    for _ in range(3):
        gc.collect()

    # CUDA 記憶體清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("🧹 完成激進記憶體清理")


def check_memory_usage():
    """檢查並報告記憶體使用情況"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(
            f"💾 記憶體使用：{allocated:.2f}GB 分配 / {reserved:.2f}GB 保留 / {max_memory:.2f}GB 總計"
        )

        # 如果使用超過 80% 記憶體，發出警告
        if allocated > max_memory * 0.8:
            print("⚠️ 記憶體使用過高，執行清理...")
            aggressive_memory_cleanup()


# ==============================================================================
# 優化的資料處理
# ==============================================================================


@dataclass
class MemorySafeDataCollator:
    """記憶體安全的 Data Collator"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        try:
            # 處理輸入特徵
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            # 處理標籤
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # 移除 BOS token
            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels

            # 清理臨時變數
            del input_features, label_features, labels_batch

            return batch

        except Exception as e:
            print(f"❌ DataCollator 錯誤：{e}")
            aggressive_memory_cleanup()
            raise e


def prepare_dataset_memory_safe(batch, feature_extractor, tokenizer):
    """記憶體安全的資料預處理"""
    try:
        audio_list = batch["audio"]

        # 分批處理音訊以節省記憶體
        input_features_list = []
        for audio in audio_list:
            features = feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
            )
            input_features_list.append(features.input_features[0])

        # 合併特徵
        batch["input_features"] = input_features_list

        # 處理標籤
        batch["labels"] = tokenizer(
            batch["transcription"], max_length=448, truncation=True
        ).input_ids

        # 清理臨時變數
        del audio_list, input_features_list

        return batch

    except Exception as e:
        print(f"❌ 資料預處理錯誤：{e}")
        aggressive_memory_cleanup()
        raise e


def compute_metrics_safe(pred, tokenizer):
    """記憶體安全的指標計算"""
    try:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # 處理標籤
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # 解碼
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # 計算 WER
        metric = evaluate.load("wer")
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        # 清理
        del pred_ids, label_ids, pred_str, label_str

        return {"wer": wer}

    except Exception as e:
        print(f"❌ 指標計算錯誤：{e}")
        return {"wer": 100.0}


# ==============================================================================
# 記憶體安全的資料集處理器
# ==============================================================================


class MemorySafeDatasetProcessor:
    """記憶體安全的資料集處理器"""

    def __init__(
        self,
        file_path: str,
        target_sampling_rate: int = 16000,
        subset_fraction: float = 0.02,
    ):
        self.file_path = file_path
        self.target_sampling_rate = target_sampling_rate
        self.subset_fraction = subset_fraction  # 預設使用 2% 資料

    def create_dataset(self) -> Dataset:
        print(f"載入資料檔案：{self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("❌ 找不到檔案，嘗試使用備用路徑...")
            alternative_path = "output/final_audio_paths_zh.csv"
            df = pd.read_csv(alternative_path)
            print(f"✅ 使用備用檔案：{alternative_path}")

        # 使用極小的資料子集以避免記憶體問題
        subset_size = max(50, int(len(df) * self.subset_fraction))  # 至少 50 個樣本
        print(f"完整資料集大小：{len(df)}")
        print(f"使用資料集大小：{subset_size} ({(subset_size/len(df)*100):.1f}%)")

        # 隨機取樣
        subset_data = df.sample(n=subset_size, random_state=42).reset_index(drop=True)

        # 清理原始資料
        del df
        gc.collect()

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
    print("=== Breeze ASR 記憶體安全訓練版本 ===")
    print("🛡️  針對 CUDA OOM 問題的徹底解決方案")

    # --- 參數設定 ---
    CSV_PATH = "output_zh_optimized_v2.csv"
    MODEL_NAME = "MediaTek-Research/Breeze-ASR-25"
    LANGUAGE = "zh"
    TASK = "transcribe"
    OUTPUT_DIR = "./whisper-small-zh-finetune-memory-safe"

    print(f"模型：{MODEL_NAME}")
    print(f"輸出目錄：{OUTPUT_DIR}")

    # 初始記憶體清理
    aggressive_memory_cleanup()
    check_memory_usage()

    try:
        # --- 載入 Processor 和模型 ---
        print("\n--- 步驟 1/4: 載入 Processor 和模型 ---")
        processor = WhisperProcessor.from_pretrained(
            MODEL_NAME, language=LANGUAGE, task=TASK
        )
        print("✅ Processor 載入成功")

        check_memory_usage()

        # 載入模型時使用記憶體優化配置
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # 使用半精度節省記憶體
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=False,  # 關閉快取以節省記憶體
        )
        print("✅ 模型載入成功")

        # 配置模型
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = False

        check_memory_usage()

        # --- 建立記憶體安全資料集 ---
        print("\n--- 步驟 2/4: 建立記憶體安全資料集 (2% 資料) ---")
        audio_processor = MemorySafeDatasetProcessor(
            file_path=CSV_PATH, subset_fraction=0.02  # 使用 2% 的資料
        )

        dataset = audio_processor.create_dataset()
        print(f"資料集建立完成，樣本數：{len(dataset)}")

        # 分割訓練和測試集
        common_voice = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"訓練集：{len(common_voice['train'])} 樣本")
        print(f"測試集：{len(common_voice['test'])} 樣本")

        # 設定即時轉換
        prepare_fn = partial(
            prepare_dataset_memory_safe,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
        )
        vectorized_datasets = common_voice.with_transform(prepare_fn)
        print("即時轉換已設定")

        check_memory_usage()

        # --- 建立訓練元件 ---
        print("\n--- 步驟 3/4: 建立記憶體安全訓練元件 ---")
        data_collator = MemorySafeDataCollator(processor=processor)
        compute_metrics_fn = partial(
            compute_metrics_safe, tokenizer=processor.tokenizer
        )

        # 極度保守的訓練參數
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            # 最小記憶體配置
            per_device_train_batch_size=1,  # 最小批次
            per_device_eval_batch_size=1,  # 最小評估批次
            gradient_accumulation_steps=8,  # 較小的累積步數
            # 關閉所有可能的記憶體消耗
            dataloader_num_workers=0,  # 關閉多進程
            dataloader_pin_memory=False,  # 關閉 pin memory
            # 保守的學習參數
            learning_rate=5e-6,  # 非常小的學習率
            warmup_steps=10,  # 最少暖身步數
            max_steps=100,  # 極少的訓練步數，先確保能運行
            # 記憶體優化
            gradient_checkpointing=False,  # 關閉梯度檢查點
            fp16=True,  # 使用半精度
            # 最少的評估和保存
            eval_strategy="steps",
            predict_with_generate=True,
            generation_max_length=128,  # 縮短生成長度
            save_steps=50,  # 頻繁保存
            eval_steps=50,  # 頻繁評估
            logging_steps=5,  # 頻繁記錄
            # 關閉不必要的功能
            report_to=[],  # 關閉所有報告
            load_best_model_at_end=False,  # 關閉最佳模型載入
            save_total_limit=1,  # 只保留1個檢查點
            # 其他優化
            remove_unused_columns=False,
            optim="adamw_torch",
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

        check_memory_usage()

        # --- 開始記憶體安全訓練 ---
        print("\n--- 步驟 4/4: 開始記憶體安全微調訓練 ---")
        print("🚀 預期訓練時間：5-10 分鐘")
        print("💡 使用極小批次和極少步數確保穩定性")

        # 訓練前最後一次記憶體清理
        aggressive_memory_cleanup()
        check_memory_usage()

        # 開始訓練
        trainer.train()
        print("\n✅ 記憶體安全訓練完成")

        # --- 儲存模型 ---
        print("\n--- 儲存訓練完成的模型 ---")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"模型已儲存至：{OUTPUT_DIR}")

        # --- 顯示訓練結果摘要 ---
        print("\n=== 記憶體安全訓練摘要 ===")
        print(f"使用資料量：2% ({len(dataset)} 樣本)")
        print(f"訓練步數：100 步")
        print(f"批次設定：batch_size=1, accumulation=8")
        print("狀態：成功完成，沒有記憶體錯誤")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ 仍然遇到記憶體不足：{e}")
        print("💡 建議：")
        print("1. 重啟 Python 環境或電腦")
        print("2. 關閉所有其他程式")
        print("3. 考慮使用 CPU 訓練")
        aggressive_memory_cleanup()

    except Exception as e:
        print(f"\n❌ 其他錯誤：{e}")
        aggressive_memory_cleanup()
        raise e

    finally:
        # 最終清理
        aggressive_memory_cleanup()
        print("🧹 執行最終記憶體清理")


if __name__ == "__main__":
    main()
