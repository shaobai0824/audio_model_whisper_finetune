#!/usr/bin/env python3
"""
Breeze-ASR-25 Google Colab 最佳化微調版本
針對 Colab T4/V100/L4/A100 GPU 環境優化
自動偵測 GPU 類型並調整最佳參數
"""

import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import jiwer
import librosa
import numpy as np
import pandas as pd
import torch
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ==============================================================================
# Colab GPU 環境配置
# ==============================================================================

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
DATASET_DIR = Path("./")
TRAIN_CSV = DATASET_DIR / "metadata_train.csv"
TEST_CSV = DATASET_DIR / "metadata_test.csv"
OUTPUT_DIR = "./breeze-asr-25-colab-optimized"

# 全域 processor 變數
_processor = None


def detect_gpu_type():
    """偵測 Colab GPU 類型並返回最佳配置"""
    if not torch.cuda.is_available():
        return "cpu", {}

    gpu_name = torch.cuda.get_device_name(0).lower()
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"🔍 偵測到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU 記憶體: {memory_gb:.1f} GB")

    # 根據 GPU 類型配置最佳參數
    if "k80" in gpu_name:
        return "k80", {
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "fp16": False,  # K80 不支援 FP16
            "max_steps": 2000,
            "learning_rate": 5e-6,
            "warmup_steps": 200,
            "save_steps": 200,
            "eval_steps": 200,
        }
    elif "t4" in gpu_name:
        return "t4", {
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "fp16": True,
            "max_steps": 3000,
            "learning_rate": 1e-5,
            "warmup_steps": 300,
            "save_steps": 300,
            "eval_steps": 300,
        }
    elif "p100" in gpu_name:
        return "p100", {
            "batch_size": 3,
            "gradient_accumulation_steps": 6,
            "fp16": True,
            "max_steps": 4000,
            "learning_rate": 1e-5,
            "warmup_steps": 400,
            "save_steps": 400,
            "eval_steps": 400,
        }
    elif "v100" in gpu_name:
        return "v100", {
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "max_steps": 5000,
            "learning_rate": 2e-5,
            "warmup_steps": 500,
            "save_steps": 500,
            "eval_steps": 500,
        }
    elif "l4" in gpu_name:
        return "l4", {
            "batch_size": 6,
            "gradient_accumulation_steps": 3,
            "fp16": True,
            "max_steps": 6000,
            "learning_rate": 2e-5,
            "warmup_steps": 600,
            "save_steps": 600,
            "eval_steps": 600,
        }
    elif "a100" in gpu_name:
        return "a100", {
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "fp16": True,
            "max_steps": 8000,
            "learning_rate": 3e-5,
            "warmup_steps": 800,
            "save_steps": 800,
            "eval_steps": 800,
        }
    else:
        # 預設配置
        return "unknown", {
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "fp16": True,
            "max_steps": 3000,
            "learning_rate": 1e-5,
            "warmup_steps": 300,
            "save_steps": 300,
            "eval_steps": 300,
        }


def setup_colab_environment():
    """設定 Colab 環境最佳化"""
    # Colab 記憶體最佳化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:512"
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 清理記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("🔧 Colab 環境設定完成")


def init_processor(proc):
    """初始化全域 processor"""
    global _processor
    _processor = proc


# ==============================================================================
# 資料處理函數 (Colab 最佳化版本)
# ==============================================================================


def prepare_dataset_colab(batch, processor=None):
    """
    Colab 最佳化的即時資料轉換函數
    針對 Colab 的記憶體限制進行優化
    """
    import gc

    import librosa
    import numpy as np

    # 使用傳入的 processor 或全域 processor
    current_processor = processor if processor is not None else _processor

    if current_processor is None:
        print("錯誤：無法取得 processor")
        return batch

    # 處理音訊資料 - 分批處理減少記憶體峰值
    audio_list = []
    batch_files = batch["file"] if isinstance(batch["file"], list) else [batch["file"]]

    for audio_path in batch_files:
        try:
            # 載入音訊並確保 16kHz 採樣率
            audio_array, sampling_rate = librosa.load(
                audio_path, sr=16000, duration=30
            )  # 限制 30 秒

            # 正規化音訊
            if len(audio_array) > 0:
                audio_array = audio_array / (np.abs(audio_array).max() + 1e-8)

            audio_list.append({"array": audio_array, "sampling_rate": sampling_rate})
        except Exception as e:
            print(f"警告：載入音訊失敗 {audio_path}: {e}")
            # 使用較短的靜音替代
            audio_list.append({"array": np.zeros(8000), "sampling_rate": 16000})

    # 處理音訊特徵 - 使用更小的批次
    try:
        # 分批處理音訊特徵以避免記憶體溢出
        feature_list = []
        for audio in audio_list:
            features = current_processor.feature_extractor(
                audio["array"],
                sampling_rate=16000,
                return_tensors="np",
                max_length=3000,  # 限制序列長度
                truncation=True,
            ).input_features[0]
            feature_list.append(features)

        batch["input_features"] = (
            np.stack(feature_list) if len(feature_list) > 1 else np.array(feature_list)
        )

    except Exception as e:
        print(f"音訊特徵提取失敗: {e}")
        batch_size = len(audio_list)
        batch["input_features"] = np.zeros((batch_size, 80, 3000))

    # 處理文字標籤
    try:
        texts = (
            batch["中文意譯"]
            if isinstance(batch["中文意譯"], list)
            else [batch["中文意譯"]]
        )
        texts = [
            str(text).strip() if text and str(text).strip() else "無內容"
            for text in texts
        ]

        labels = current_processor.tokenizer(
            texts,
            max_length=256,  # 降低最大長度以節省記憶體
            truncation=True,
            padding=False,  # 不在這裡填充
            return_tensors="np",
        ).input_ids

        batch["labels"] = labels

    except Exception as e:
        print(f"文字標籤處理失敗: {e}")
        batch["labels"] = np.array([[1]] * len(audio_list))

    # 手動垃圾回收
    gc.collect()

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqColab:
    """
    Colab 最佳化的 Data Collator
    針對記憶體效率優化
    """

    processor: WhisperProcessor
    decoder_start_token_id: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # 處理 input_features - 確保統一格式
        input_features = []
        for feature in features:
            if isinstance(feature["input_features"], np.ndarray):
                if feature["input_features"].ndim == 2:
                    # 如果是 2D，添加批次維度
                    input_feat = torch.tensor(
                        feature["input_features"], dtype=torch.float32
                    ).unsqueeze(0)
                else:
                    input_feat = torch.tensor(
                        feature["input_features"], dtype=torch.float32
                    )
            else:
                input_feat = torch.tensor(
                    feature["input_features"], dtype=torch.float32
                )

            input_features.append({"input_features": input_feat.squeeze(0)})

        # 使用 processor 進行填充
        try:
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt", max_length=3000, truncation=True
            )
        except Exception as e:
            print(f"特徵填充失敗: {e}")
            # 手動創建批次
            max_len = 3000
            batch_input_features = torch.zeros((len(features), 80, max_len))
            for i, feature in enumerate(features):
                feat = torch.tensor(feature["input_features"], dtype=torch.float32)
                if feat.ndim == 3:
                    feat = feat.squeeze(0)
                seq_len = min(feat.shape[-1], max_len)
                batch_input_features[i, :, :seq_len] = feat[:, :seq_len]
            batch = {"input_features": batch_input_features}

        # 處理標籤
        label_features = []
        for feature in features:
            labels = feature["labels"]
            if isinstance(labels, np.ndarray):
                if labels.ndim == 2:
                    labels = labels[0]  # 取第一個序列
                labels = torch.tensor(labels, dtype=torch.long)
            else:
                labels = torch.tensor(labels, dtype=torch.long)
            label_features.append({"input_ids": labels})

        try:
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt", max_length=256, truncation=True
            )
        except Exception as e:
            print(f"標籤填充失敗: {e}")
            # 手動創建標籤批次
            max_label_len = 256
            batch_labels = torch.full(
                (len(features), max_label_len), self.processor.tokenizer.pad_token_id
            )
            attention_mask = torch.zeros((len(features), max_label_len))

            for i, feature in enumerate(features):
                labels = torch.tensor(feature["labels"], dtype=torch.long)
                if labels.ndim > 1:
                    labels = labels.flatten()
                label_len = min(len(labels), max_label_len)
                batch_labels[i, :label_len] = labels[:label_len]
                attention_mask[i, :label_len] = 1

            labels_batch = {"input_ids": batch_labels, "attention_mask": attention_mask}

        # 處理標籤遮罩
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        # 移除 BOS token（如果存在）
        if self.decoder_start_token_id is not None:
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics_colab(pred, processor=None):
    """
    Colab 最佳化的評估指標計算
    """
    current_processor = processor if processor is not None else _processor

    if current_processor is None:
        print("錯誤：無法取得 processor 進行評估")
        return {"wer": 100.0}

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 將 -100 替換為 pad token
    label_ids[label_ids == -100] = current_processor.tokenizer.pad_token_id

    # 解碼預測和標籤
    try:
        pred_str = current_processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = current_processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        # 計算 WER（百分比）
        wer = 100 * jiwer.wer(label_str, pred_str)

        # 計算額外指標
        try:
            cer = 100 * jiwer.cer(label_str, pred_str)
            return {"wer": wer, "cer": cer}
        except:
            return {"wer": wer}

    except Exception as e:
        print(f"評估指標計算失敗: {e}")
        return {"wer": 100.0}


# ==============================================================================
# 主要訓練函數
# ==============================================================================


def main():
    """
    Colab 最佳化的主要訓練流程
    """
    global _processor

    print("🚀 Breeze-ASR-25 Google Colab 最佳化微調開始")
    print("=" * 60)

    # 設定 Colab 環境
    setup_colab_environment()

    # 偵測 GPU 並取得最佳配置
    gpu_type, gpu_config = detect_gpu_type()
    print(f"🎯 使用 {gpu_type.upper()} 最佳化配置")

    # 檢查資料檔案
    if not TRAIN_CSV.exists() or not TEST_CSV.exists():
        print(f"❌ 錯誤：找不到資料檔案")
        print(f"   訓練檔案: {TRAIN_CSV}")
        print(f"   測試檔案: {TEST_CSV}")
        return

    # 載入資料
    print("\n📊 載入資料集...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"   原始訓練資料: {len(train_df)} 筆")
    print(f"   原始測試資料: {len(test_df)} 筆")

    # 自動檢查和修正路徑格式
    print("\n🔧 檢查路徑格式...")
    path_column = "file"  # 假設路徑在 'file' 欄位

    if path_column in train_df.columns:
        sample_path = str(train_df[path_column].iloc[0]).strip()

        # 檢查是否為 Windows 絕對路徑
        if sample_path.startswith(("C:", "D:", "E:", "F:")):
            print(f"   ⚠️  偵測到 Windows 絕對路徑: {sample_path}")
            print("   🔄 自動修正為 Colab 路徑...")

            # 修正訓練資料路徑
            train_df[path_column] = train_df[path_column].apply(
                lambda x: (
                    f"/content/drive/MyDrive/audio_model/audio_files/{Path(x).name}"
                    if isinstance(x, str) and x.startswith(("C:", "D:", "E:", "F:"))
                    else x
                )
            )

            # 修正測試資料路徑
            test_df[path_column] = test_df[path_column].apply(
                lambda x: (
                    f"/content/drive/MyDrive/audio_model/audio_files/{Path(x).name}"
                    if isinstance(x, str) and x.startswith(("C:", "D:", "E:", "F:"))
                    else x
                )
            )

            print(f"   ✅ 路徑已修正，範例: {train_df[path_column].iloc[0]}")

        elif sample_path.startswith("/content/drive/"):
            print(f"   ✅ 已是正確的 Colab 路徑: {sample_path}")

        else:
            print(f"   ⚠️  路徑格式可能需要調整: {sample_path}")
            print("   💡 建議使用 fix_csv_paths_for_colab.py 預先處理")

    # 資料品質檢查和清理
    print("\n🔍 資料品質檢查...")
    required_columns = ["file", "中文意譯"]
    for col in required_columns:
        if col not in train_df.columns:
            print(f"❌ 錯誤：訓練資料缺少欄位 '{col}'")
            return

    # 移除空值和無效資料
    train_df = train_df.dropna(subset=required_columns)
    test_df = test_df.dropna(subset=required_columns)

    # 過濾過長的文字（避免記憶體問題）
    train_df = train_df[train_df["中文意譯"].str.len() < 200]
    test_df = test_df[test_df["中文意譯"].str.len() < 200]

    print(f"   清理後訓練資料: {len(train_df)} 筆")
    print(f"   清理後測試資料: {len(test_df)} 筆")

    # 針對較慢的 GPU 減少資料量
    if gpu_type in ["k80", "t4"]:
        train_sample_size = min(len(train_df), 20000)  # 限制訓練資料量
        test_sample_size = min(len(test_df), 2000)

        train_df = train_df.sample(n=train_sample_size, random_state=42)
        test_df = test_df.sample(n=test_sample_size, random_state=42)

        print(
            f"   GPU 記憶體限制，使用子集：訓練 {len(train_df)} 筆，測試 {len(test_df)} 筆"
        )

    # 載入模型和處理器
    print("\n🤖 載入模型和處理器...")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_ID)

        # 根據 GPU 配置選擇精度
        dtype = torch.float32 if not gpu_config.get("fp16", True) else torch.float16

        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_cache=False,  # 節省記憶體
        )

        # 模型配置優化
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        # 如果不支援 FP16，確保使用 FP32
        if not gpu_config.get("fp16", True):
            model = model.float()

        print("   ✅ 模型和處理器載入成功")
        init_processor(processor)

    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 創建資料集
    print("\n📋 設定資料集和即時轉換...")
    try:
        # 合併並重新分割資料
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_dataset = Dataset.from_pandas(full_df)

        dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)

        def prepare_dataset_with_processor(batch):
            return prepare_dataset_colab(batch, processor)

        vectorized_datasets = dataset_split.with_transform(
            prepare_dataset_with_processor
        )

        train_dataset = vectorized_datasets["train"]
        test_dataset = vectorized_datasets["test"]

        print(f"   ✅ 即時轉換設定完成")
        print(f"   最終訓練資料: {len(train_dataset)} 筆")
        print(f"   最終測試資料: {len(test_dataset)} 筆")

    except Exception as e:
        print(f"❌ 資料集設定失敗: {e}")
        return

    # 創建 Data Collator
    data_collator = DataCollatorSpeechSeq2SeqColab(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 設定訓練參數（使用 GPU 特定配置）
    print(f"\n⚙️  配置 {gpu_type.upper()} 最佳化訓練參數...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # GPU 特定的批次設定
        per_device_train_batch_size=gpu_config["batch_size"],
        per_device_eval_batch_size=max(1, gpu_config["batch_size"] // 2),
        gradient_accumulation_steps=gpu_config["gradient_accumulation_steps"],
        # 學習率和調度
        learning_rate=gpu_config["learning_rate"],
        warmup_steps=gpu_config["warmup_steps"],
        max_steps=gpu_config["max_steps"],
        # 評估和生成設定
        eval_strategy="steps",
        eval_steps=gpu_config["eval_steps"],
        predict_with_generate=True,
        generation_max_length=256,  # 降低生成長度
        # 保存策略
        save_steps=gpu_config["save_steps"],
        save_total_limit=2,  # 只保留 2 個檢查點節省空間
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # 記憶體和效能優化
        fp16=gpu_config["fp16"],
        gradient_checkpointing=False,  # 關閉梯度檢查點
        dataloader_num_workers=0,  # Colab 建議設為 0
        dataloader_pin_memory=False,  # Colab 環境建議關閉
        # 日誌和監控
        logging_steps=50,
        report_to=[],  # 關閉外部報告以節省記憶體
        # 其他設定
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
        # 優化器設定
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        # Colab 特殊設定
        ignore_data_skip=True,  # 忽略資料跳過
        dataloader_drop_last=True,  # 丟棄最後不完整批次
    )

    # 記憶體清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 創建 Trainer
    print("\n🏋️  創建訓練器...")

    def compute_metrics_with_processor(pred):
        return compute_metrics_colab(pred, processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_processor,
        tokenizer=processor.feature_extractor,
    )

    # 顯示訓練資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    effective_batch_size = (
        gpu_config["batch_size"] * gpu_config["gradient_accumulation_steps"]
    )

    print(f"\n📈 訓練資訊:")
    print(f"   模型參數總數: {total_params:,}")
    print(f"   可訓練參數: {trainable_params:,}")
    print(f"   有效批次大小: {effective_batch_size}")
    print(f"   預估訓練時間: {gpu_type.upper()} 約 1-3 小時")

    # 開始訓練
    print("\n🚀 開始訓練...")
    print("=" * 60)

    try:
        # 訓練前記憶體檢查
        if torch.cuda.is_available():
            print(
                f"   訓練前 GPU 記憶體使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        # 執行訓練
        trainer.train()

        print("\n🎉 訓練完成！")

        # 保存模型
        print("\n💾 保存模型...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        # 最終評估
        print("\n📊 最終評估...")
        final_metrics = trainer.evaluate()

        print(f"   最終 WER: {final_metrics.get('eval_wer', 'N/A'):.2f}%")
        if "eval_cer" in final_metrics:
            print(f"   最終 CER: {final_metrics['eval_cer']:.2f}%")

        print(f"\n✅ 模型已保存至: {OUTPUT_DIR}")
        print("🎯 Colab 訓練成功完成！")

        # 提供下載指令
        print(f"\n📦 模型下載指令:")
        print(f"!zip -r {OUTPUT_DIR}.zip {OUTPUT_DIR}")
        print(f"from google.colab import files")
        print(f"files.download('{OUTPUT_DIR}.zip')")

    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()

        # 保存中斷狀態
        try:
            print("\n🔄 嘗試保存當前狀態...")
            trainer.save_model(f"{OUTPUT_DIR}_interrupted")
            processor.save_pretrained(f"{OUTPUT_DIR}_interrupted")
            print("   ✅ 中斷狀態已保存")
        except:
            print("   ❌ 無法保存中斷狀態")

    finally:
        # 清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(
                f"\n🧹 清理後 GPU 記憶體: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )


# ==============================================================================
# Colab 輔助函數
# ==============================================================================


def check_colab_environment():
    """檢查 Colab 環境並提供建議"""
    print("🔍 Colab 環境檢查:")

    # GPU 檢查
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✅ GPU: {gpu_name} ({memory_gb:.1f} GB)")
    else:
        print("   ❌ 未檢測到 GPU，請確認已啟用 GPU 運行時")
        return False

    # RAM 檢查
    import psutil

    ram_gb = psutil.virtual_memory().total / 1024**3
    print(f"   ✅ RAM: {ram_gb:.1f} GB")

    # 磁碟空間檢查
    disk_usage = psutil.disk_usage("/")
    free_gb = disk_usage.free / 1024**3
    print(f"   ✅ 可用磁碟空間: {free_gb:.1f} GB")

    if free_gb < 5:
        print("   ⚠️  警告：磁碟空間不足，建議清理後再訓練")

    return True


def setup_colab_training():
    """設定 Colab 訓練環境的完整流程"""
    print("🔧 設定 Colab 訓練環境...")

    # 掛載 Google Drive（如果需要）
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        print("   ✅ Google Drive 已掛載")
    except:
        print("   ℹ️  未掛載 Google Drive（非 Colab 環境或已掛載）")

    # 安裝必要套件
    required_packages = [
        "datasets",
        "transformers",
        "accelerate",
        "jiwer",
        "librosa",
        "soundfile",
    ]

    print("   📦 檢查必要套件...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"      ✅ {package}")
        except ImportError:
            print(f"      ⚠️  {package} 需要安裝")
            print(f"         請執行: !pip install {package}")


if __name__ == "__main__":
    # Colab 環境檢查
    if not check_colab_environment():
        print("❌ 環境檢查失敗，請修正後重試")
        exit(1)

    # 設定訓練環境
    setup_colab_training()

    # 執行主訓練
    main()
