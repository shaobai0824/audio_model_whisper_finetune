from pathlib import Path
from typing import Dict, List, Union

import jiwer
import librosa
import pandas as pd
import torch
from datasets import Audio, load_dataset
from transformers import (
    DataCollatorForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# --- 1. 全域配置 (Global Configuration) ---
MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../model_outputs")
MODEL_OUTPUT_PATH = OUTPUT_DIR / "breeze-asr-25-taiwanese-finetuned"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"

# --- 2. 載入自訂資料集 (Load Custom Dataset) ---
# 使用 Hugging Face Datasets 的 'audiofolder' 功能
# 它會自動尋找 CSV 檔案並載入音訊
dataset = load_dataset(
    "audiofolder", data_dir=DATA_DIR, cache_dir="./.cache"  # 指定快取目錄
)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# --- 3. 建立 Processor (Tokenizer + Feature Extractor) ---
# 3.1. 載入自訂詞彙表建立 Tokenizer
tokenizer = Wav2Vec2CTCTokenizer(
    VOCAB_PATH,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

# 3.2. 載入 Feature Extractor
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)

# 3.3. 組合為 Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# --- 4. 資料預處理 (Data Preprocessing) ---
# 確保所有音訊都以 16kHz 採樣率載入
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch: Dict) -> Dict:
    """對單筆資料進行處理"""
    audio = batch["audio"]

    # 使用 processor 處理音訊和文字
    # 音訊 -> input_values
    # 文字 -> labels
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids

    return batch


print("Preprocessing datasets...")
train_dataset = train_dataset.map(
    prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4
)
test_dataset = test_dataset.map(
    prepare_dataset, remove_columns=test_dataset.column_names, num_proc=4
)

# --- 5. 設定訓練器 (Setup Trainer) ---
# 5.1. 資料整理器 (Data Collator)
# DataCollatorForCTC 會動態地對 input_values 和 labels 進行填充
data_collator = DataCollatorForCTC(processor=processor, padding=True)


# 5.2. 評估指標 (Evaluation Metrics) - Word Error Rate (WER)
def compute_metrics(pred) -> Dict[str, float]:
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.from_numpy(pred_logits), dim=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}


# 5.3. 載入預訓練模型
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),  # **非常重要**: 調整詞彙表大小
)

# 凍結 feature extractor 的權重
model.freeze_feature_encoder()

# 5.4. 訓練參數
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    group_by_length=True,
    per_device_train_batch_size=8,  # 根據您的 VRAM 調整
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,  # 根據您的資料集大小調整
    fp16=True,  # 若 GPU 支援，可加速訓練
    save_steps=500,
    eval_steps=500,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    push_to_hub=False,
)

# 5.5. 建立 Trainer
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)

# --- 6. 執行訓練 (Start Training) ---
print("Starting fine-tuning...")
trainer.train()

# --- 7. 儲存最終模型 ---
print("Saving final model...")
trainer.save_model(MODEL_OUTPUT_PATH)
processor.save_pretrained(MODEL_OUTPUT_PATH)
print("Fine-tuning complete.")
