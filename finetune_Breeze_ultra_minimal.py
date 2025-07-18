# ==============================================================================
# 檔案：finetune_Breeze_ultra_minimal.py
# 描述：極度精簡版本 - 專為 8GB GPU 設計
# 核心策略：
# 1. 激進的記憶體管理
# 2. CPU 備選方案
# 3. 最小資料集和模型配置
# 4. 詳細的記憶體監控
# ==============================================================================

import gc
import os
import sys
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import pandas as pd
import psutil
import torch

# 設定環境變數 - 激進的記憶體管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Audio, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ==============================================================================
# 激進的記憶體管理工具
# ==============================================================================


def aggressive_cleanup():
    """激進的記憶體清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        # 強制回收 CUDA 記憶體
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass


def check_system_memory():
    """檢查系統和 GPU 記憶體"""
    # 系統記憶體
    memory = psutil.virtual_memory()
    print(
        f"💾 系統記憶體：{memory.used/1024**3:.2f}GB 使用 / {memory.total/1024**3:.2f}GB 總計"
    )

    # GPU 記憶體
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(
                f"🔥 GPU {i}：{allocated:.2f}GB 分配 / {reserved:.2f}GB 保留 / {total:.2f}GB 總計"
            )

            # 警告記憶體使用過高
            if allocated > total * 0.8:
                print(f"⚠️  警告：GPU {i} 記憶體使用率過高！")
                return False
    return True


def force_memory_reset():
    """強制重設記憶體狀態"""
    print("🔄 強制重設記憶體...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 嘗試重設 CUDA 上下文
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass
    gc.collect()


# ==============================================================================
# 極簡資料處理器
# ==============================================================================


@dataclass
class UltraMinimalDataCollator:
    """極簡 Data Collator"""

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        try:
            # 極簡處理
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )

            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            if (
                (labels[:, 0] == self.processor.tokenizer.bos_token_id)
                .all()
                .cpu()
                .item()
            ):
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch
        except Exception as e:
            print(f"❌ Data Collator 錯誤：{e}")
            raise


def prepare_dataset_minimal(batch, feature_extractor, tokenizer):
    """極簡資料預處理"""
    try:
        audio_list = batch["audio"]

        # 極簡音訊處理
        input_features = feature_extractor(
            [x["array"] for x in audio_list],
            sampling_rate=audio_list[0]["sampling_rate"],
            return_tensors="np",
        ).input_features

        # 極簡標籤處理
        labels = tokenizer(
            batch["transcription"],
            max_length=224,  # 減少最大長度
            truncation=True,
            return_tensors="np",
        ).input_ids

        return {"input_features": input_features, "labels": labels}
    except Exception as e:
        print(f"❌ 資料預處理錯誤：{e}")
        raise


# ==============================================================================
# 極簡資料集處理器
# ==============================================================================


class UltraMinimalDatasetProcessor:
    """極簡資料集處理器 - 僅使用最少樣本"""

    def __init__(self, file_path: str, target_sampling_rate: int = 16000):
        self.file_path = file_path
        self.target_sampling_rate = target_sampling_rate

    def create_dataset(self) -> Dataset:
        print(f"載入資料檔案：{self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            print("❌ 找不到檔案，嘗試使用備用路徑...")
            alternative_path = "output/final_audio_paths_zh.csv"
            df = pd.read_csv(alternative_path)
            print(f"✅ 使用備用檔案：{alternative_path}")

        # 使用極小樣本數 - 僅 10 個樣本
        subset_size = 10
        print(f"完整資料集大小：{len(df)}")
        print(f"使用資料集大小：{subset_size} (極小測試集)")

        # 取前 10 個樣本
        subset_data = df.head(subset_size).reset_index(drop=True)

        dataset = Dataset.from_pandas(subset_data)
        dataset = dataset.cast_column(
            "file", Audio(sampling_rate=self.target_sampling_rate)
        )
        dataset = dataset.rename_column("file", "audio")

        return dataset


# ==============================================================================
# 主執行流程 - 支援 CPU 備選
# ==============================================================================


def main():
    print("=== Breeze ASR 極度精簡版本 ===")
    print("🔧 專為 8GB GPU 記憶體限制設計")

    # --- 參數設定 ---
    CSV_PATH = "output_zh_optimized_v2.csv"
    MODEL_NAME = "MediaTek-Research/Breeze-ASR-25"
    LANGUAGE = "zh"
    TASK = "transcribe"
    OUTPUT_DIR = "./whisper-small-zh-finetune-ultra-minimal"

    print(f"模型：{MODEL_NAME}")
    print(f"輸出目錄：{OUTPUT_DIR}")

    # 檢查初始記憶體狀態
    aggressive_cleanup()
    if not check_system_memory():
        print("❌ 初始記憶體狀態不佳，正在重設...")
        force_memory_reset()

    # 決定使用 CPU 還是 GPU
    use_cpu = False
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 10:  # 如果 GPU 記憶體小於 10GB，考慮使用 CPU
            print(f"⚠️  GPU 記憶體不足 ({gpu_memory:.1f}GB)，考慮使用 CPU 模式")
            use_cpu = True
    else:
        use_cpu = True

    device = "cpu" if use_cpu else "cuda"
    print(f"🔥 使用設備：{device.upper()}")

    try:
        # --- 載入 Processor ---
        print("\n--- 步驟 1/4: 載入 Processor ---")
        processor = WhisperProcessor.from_pretrained(
            MODEL_NAME, language=LANGUAGE, task=TASK
        )
        print("✅ Processor 載入成功")

        # --- 載入模型 ---
        print("\n--- 步驟 2/4: 載入模型 ---")
        if use_cpu:
            print("💡 使用 CPU 模式避免 GPU 記憶體限制")
            model = WhisperForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
        else:
            print("💡 使用極簡 GPU 配置")
            model = WhisperForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,  # 使用 FP16 節省記憶體
                device_map="auto",
                low_cpu_mem_usage=True,
            )

        # 配置模型
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        print("✅ 模型載入成功")
        check_system_memory()

        # --- 建立極小資料集 ---
        print("\n--- 步驟 3/4: 建立極小資料集 (10 樣本) ---")
        audio_processor = UltraMinimalDatasetProcessor(file_path=CSV_PATH)
        dataset = audio_processor.create_dataset()
        print(f"資料集建立完成，樣本數：{len(dataset)}")

        # 分割訓練和測試集
        common_voice = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"訓練集：{len(common_voice['train'])} 樣本")
        print(f"測試集：{len(common_voice['test'])} 樣本")

        # 設定即時轉換
        prepare_fn = partial(
            prepare_dataset_minimal,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
        )
        vectorized_datasets = common_voice.with_transform(prepare_fn)
        print("即時轉換已設定")

        check_system_memory()

        # --- 建立訓練元件 ---
        print("\n--- 步驟 4/4: 建立極簡訓練配置 ---")
        data_collator = UltraMinimalDataCollator(processor=processor)

        # 極簡訓練參數
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            # 極小批次配置
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,  # 減少累積步數
            # 關閉多進程
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            # 學習參數
            learning_rate=1e-5,
            warmup_steps=2,
            max_steps=20,  # 極少步數
            # 精度設定
            fp16=False if use_cpu else True,
            bf16=False,
            # 簡化的評估和保存
            eval_strategy="no",  # 關閉評估以節省記憶體
            save_steps=20,
            logging_steps=5,
            # 關閉不必要的功能
            report_to=[],
            load_best_model_at_end=False,
            save_total_limit=1,
            # 其他設定
            remove_unused_columns=False,
            optim="adamw_torch",
            gradient_checkpointing=True,  # 啟用梯度檢查點節省記憶體
        )

        # 建立訓練器
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=vectorized_datasets["train"],
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
        )

        check_system_memory()

        # --- 開始訓練 ---
        print("\n--- 開始極簡訓練 ---")
        print("🚀 預期訓練時間：1-2 分鐘")
        print(f"💡 使用 {device.upper()} 模式")

        aggressive_cleanup()
        check_system_memory()

        # 開始訓練
        trainer.train()
        print("\n✅ 極簡訓練完成！")

        # --- 儲存模型 ---
        print("\n--- 儲存模型 ---")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print(f"模型已儲存至：{OUTPUT_DIR}")

        # --- 訓練摘要 ---
        print("\n=== 極簡訓練摘要 ===")
        print(f"✅ 成功完成訓練！")
        print(f"使用設備：{device.upper()}")
        print(f"使用資料量：10 樣本 (極小測試)")
        print(f"訓練步數：20 步")
        print(f"狀態：無錯誤完成")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU 記憶體不足：{e}")
        print("\n🔧 建議解決方案：")
        print("1. 重啟 Python 環境清理記憶體")
        print("2. 關閉其他使用 GPU 的程式")
        print("3. 使用 CPU 模式：設定 use_cpu=True")
        force_memory_reset()
        raise e

    except Exception as e:
        print(f"\n❌ 訓練失敗：{e}")
        print(f"錯誤類型：{type(e).__name__}")
        aggressive_cleanup()
        raise e

    finally:
        aggressive_cleanup()
        print("🧹 最終清理完成")


if __name__ == "__main__":
    main()
