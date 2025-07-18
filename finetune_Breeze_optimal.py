#!/usr/bin/env python3
"""
Breeze-ASR-25 最佳化微調版本
結合 train.py 的記憶體效率和 finetune_Breeze_whisper.py 的功能性
針對 RTX 3060 Ti 8GB 優化，確保穩定高效的訓練
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
# 全域配置和變數 (確保多進程相容性)
# ==============================================================================

MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
DATASET_DIR = Path("./")
TRAIN_CSV = DATASET_DIR / "metadata_train.csv"
TEST_CSV = DATASET_DIR / "metadata_test.csv"
OUTPUT_DIR = "./breeze-asr-25-optimal"

# 全域 processor 變數
_processor = None


def init_processor(proc):
    """初始化全域 processor"""
    global _processor
    _processor = proc


# ==============================================================================
# 資料處理函數 (全域定義以支援多進程)
# ==============================================================================


def prepare_dataset_optimal(batch, processor=None):
    """
    最佳化的即時資料轉換函數
    結合 train.py 的記憶體效率和 Breeze 的功能性
    """
    import librosa
    import numpy as np

    # 使用傳入的 processor 或全域 processor
    current_processor = processor if processor is not None else _processor

    if current_processor is None:
        print("錯誤：無法取得 processor")
        return batch

    # 處理音訊資料
    audio_list = []
    for audio_path in batch["file"]:
        try:
            # 載入音訊並確保 16kHz 採樣率
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            audio_list.append({"array": audio_array, "sampling_rate": sampling_rate})
        except Exception as e:
            print(f"警告：載入音訊失敗 {audio_path}: {e}")
            # 使用靜音替代
            audio_list.append({"array": np.zeros(16000), "sampling_rate": 16000})

    # 處理音訊特徵 - 使用 Whisper 的標準處理方式
    try:
        batch["input_features"] = current_processor.feature_extractor(
            [x["array"] for x in audio_list],
            sampling_rate=16000,
            return_tensors="np",  # 返回 numpy 格式避免 GPU 記憶體累積
        ).input_features
    except Exception as e:
        print(f"音訊特徵提取失敗: {e}")
        # 創建空的特徵作為備用
        batch["input_features"] = np.zeros((len(audio_list), 80, 3000))

    # 處理文字標籤 - 使用中文意譯欄位
    try:
        texts = [str(text).strip() for text in batch["中文意譯"]]
        # 過濾空文字
        texts = [text if text else "無內容" for text in texts]

        batch["labels"] = current_processor.tokenizer(
            texts,
            max_length=448,
            truncation=True,
            return_tensors="np",  # 返回 numpy 格式
        ).input_ids
    except Exception as e:
        print(f"文字標籤處理失敗: {e}")
        # 創建空標籤作為備用
        batch["labels"] = np.zeros((len(batch["中文意譯"]), 1))

    return batch


@dataclass
class DataCollatorSpeechSeq2SeqOptimal:
    """
    最佳化的 Data Collator
    結合記憶體效率和功能完整性
    """

    processor: WhisperProcessor
    decoder_start_token_id: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 處理 input_features
        input_features = [
            {
                "input_features": torch.tensor(
                    feature["input_features"], dtype=torch.float32
                )
            }
            for feature in features
        ]

        # 使用 processor 進行填充
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 處理標籤
        label_features = [
            {"input_ids": torch.tensor(feature["labels"], dtype=torch.long)}
            for feature in features
        ]

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 處理標籤遮罩
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # 移除 BOS token（如果存在）
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics_optimal(pred, processor=None):
    """
    最佳化的評估指標計算
    使用百分比 WER 以便於理解
    """
    # 使用傳入的 processor 或全域 processor
    current_processor = processor if processor is not None else _processor

    if current_processor is None:
        print("錯誤：無法取得 processor 進行評估")
        return {"wer": 100.0}

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 將 -100 替換為 pad token
    label_ids[label_ids == -100] = current_processor.tokenizer.pad_token_id

    # 解碼預測和標籤
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
        # 計算字符錯誤率
        cer = 100 * jiwer.cer(label_str, pred_str)
        return {"wer": wer, "cer": cer}
    except:
        return {"wer": wer}


# ==============================================================================
# 主要訓練函數
# ==============================================================================


def main():
    """
    最佳化的主要訓練流程
    結合兩個版本的優勢
    """
    global _processor

    # Windows 多進程支援
    if __name__ == "__main__":
        multiprocessing.freeze_support()

    print("🚀 Breeze-ASR-25 最佳化微調開始")
    print("=" * 60)

    # 記憶體優化設定
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # 清理 GPU 記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU 記憶體已清理，可用記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

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

    # 資料品質檢查
    print("\n🔍 資料品質檢查...")

    # 檢查必要欄位
    required_columns = ["file", "中文意譯"]
    for col in required_columns:
        if col not in train_df.columns:
            print(f"❌ 錯誤：訓練資料缺少欄位 '{col}'")
            return

    # 移除空值
    train_df = train_df.dropna(subset=required_columns)
    test_df = test_df.dropna(subset=required_columns)

    print(f"   清理後訓練資料: {len(train_df)} 筆")
    print(f"   清理後測試資料: {len(test_df)} 筆")

    # 載入模型和處理器
    print("\n🤖 載入模型和處理器...")
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,  # 使用 FP32 避免梯度問題
            low_cpu_mem_usage=True,
        )

        # 模型配置優化
        model.config.use_cache = False  # 關閉快取節省記憶體
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        print("   ✅ 模型和處理器載入成功")

        # 初始化全域 processor
        init_processor(processor)

    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 創建資料集 - 使用即時轉換策略
    print("\n📋 設定資料集和即時轉換...")
    try:
        # 合併資料並重新分割以確保分佈均勻
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_dataset = Dataset.from_pandas(full_df)

        # 重新分割資料
        dataset_split = full_dataset.train_test_split(test_size=0.2, seed=42)

        # 設定即時轉換函數，將 processor 作為參數傳遞
        def prepare_dataset_with_processor(batch):
            return prepare_dataset_optimal(batch, processor)

        # 使用 with_transform 進行即時轉換
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
    data_collator = DataCollatorSpeechSeq2SeqOptimal(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 最佳化訓練參數
    print("\n⚙️  配置訓練參數...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        # 記憶體優化的批次設定
        per_device_train_batch_size=4,  # 進一步降低以確保穩定
        per_device_eval_batch_size=4,  # 評估也使用小批次
        gradient_accumulation_steps=4,  # 增加累積步數 (有效批次 = 2*8 = 16)
        # 學習率和調度
        learning_rate=1e-5,  # 保守的學習率
        warmup_steps=500,  # 適度的預熱
        max_steps=5000,  # 限制總步數
        # 評估和生成設定
        eval_strategy="steps",
        eval_steps=500,  # 頻繁評估以監控進度
        predict_with_generate=True,  # 啟用生成模式評估
        generation_max_length=448,  # 限制生成長度
        # 保存策略
        save_steps=500,  # 頻繁保存
        save_total_limit=3,  # 保留最近 3 個檢查點
        load_best_model_at_end=True,  # 載入最佳模型
        metric_for_best_model="wer",  # 使用 WER 作為評估指標
        greater_is_better=False,  # WER 越低越好
        # 記憶體和效能優化
        fp16=True,  # 啟用 FP16 節省記憶體
        gradient_checkpointing=False,  # 關閉梯度檢查點避免相容性問題
        dataloader_num_workers=0,  # 關閉多進程避免 Windows 問題
        dataloader_pin_memory=True,  # 啟用 pin memory
        # 日誌和監控
        logging_steps=25,  # 詳細的日誌記錄
        report_to=["tensorboard"],  # 啟用 TensorBoard
        # 其他設定
        remove_unused_columns=False,  # 保留所有欄位
        label_names=["labels"],  # 指定標籤名稱
        push_to_hub=False,  # 不推送到 Hub
        # 優化器設定
        optim="adamw_torch",  # 使用 PyTorch AdamW
        weight_decay=0.01,  # 適度的權重衰減
        lr_scheduler_type="cosine",  # 使用 cosine 調度器
        # 移除早停設定 (Seq2SeqTrainingArguments 不支援)
    )

    # 創建 Trainer
    print("\n🏋️  創建訓練器...")

    # 創建帶有 processor 的 compute_metrics 函數
    def compute_metrics_with_processor(pred):
        return compute_metrics_optimal(pred, processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_with_processor,
        tokenizer=processor.feature_extractor,
    )

    # 訓練前的記憶體檢查
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   訓練前 GPU 記憶體: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 開始訓練
    print("\n🚀 開始訓練...")
    print("=" * 60)

    try:
        # 執行訓練
        trainer.train()

        print("\n🎉 訓練完成！")

        # 保存最終模型
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
        print("🎯 訓練成功完成！")

    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback

        traceback.print_exc()

        # 嘗試保存當前狀態
        try:
            print("\n🔄 嘗試保存當前狀態...")
            trainer.save_model(f"{OUTPUT_DIR}_interrupted")
            processor.save_pretrained(f"{OUTPUT_DIR}_interrupted")
            print("   ✅ 中斷狀態已保存")
        except:
            print("   ❌ 無法保存中斷狀態")

        return

    # 清理記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"\n🧹 清理後 GPU 記憶體: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )


if __name__ == "__main__":
    main()
