import csv
import math
import time

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_dir = "Bohanlu/Taigi-Llama-2-Translator-7B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 4-bit 量化以節省記憶體
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
)

PROMPT_TEMPLATE = "[TRANS]\n{source_sentence}\n[/TRANS]\n[{target_language}]\n"


def extract_zh_content(text: str) -> str:
    # 先找 [ZH] ... [/ZH]
    start = text.find("[ZH]")
    end = text.find("[/ZH]")
    if start != -1 and end != -1 and start < end:
        return text[start + 4 : end].strip()
    # 若只有 [/ZH]，取其前所有內容
    elif end != -1:
        return text[:end].strip()
    # 其他情況，回傳原始內容或空字串
    return text.strip()


def translate_batch(
    sentences: list, target_language: str = "ZH", batch_size: int = 8
) -> list:
    """批次翻譯函數"""
    results = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    # 分批處理
    for batch_idx in range(0, len(sentences), batch_size):
        batch_sentences = sentences[batch_idx : batch_idx + batch_size]
        current_batch_num = batch_idx // batch_size + 1

        print(f"\n=== 批次 {current_batch_num}/{total_batches} ===")

        # 建立批次 prompt
        prompts = [
            PROMPT_TEMPLATE.format(source_sentence=s, target_language=target_language)
            for s in batch_sentences
        ]

        # 批次 tokenize
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        # 批次生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # 改回 256，參考原版設定
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解碼結果並即時輸出
        batch_results = []
        for j, output in enumerate(outputs):
            # 找到對應的輸入長度
            input_length = inputs["attention_mask"][j].sum()
            generated_text = tokenizer.decode(
                output[input_length:], skip_special_tokens=True
            )

            # 加入 debug 輸出，檢查模型原始回應
            print(f"DEBUG raw output {batch_idx + j + 1}:", repr(generated_text))

            zh_content = extract_zh_content(generated_text)
            batch_results.append(zh_content)

            # 即時輸出翻譯結果
            original_idx = batch_idx + j
            print(f"{original_idx + 1}: {batch_sentences[j]} → {zh_content}")

        results.extend(batch_results)
        print(f"批次 {current_batch_num} 完成，已處理 {len(results)} 筆")

    return results


input_file = "output/final_audio_paths_zh.csv"
output_file = "output_zh_optimized.csv"

# 讀取資料
print("讀取 CSV 檔案...")
with open(input_file, encoding="utf-8-sig") as fin:
    reader = csv.DictReader(fin)
    rows = list(reader)

    # 處理 fieldnames
    if reader.fieldnames is not None:
        fieldnames = (
            list(reader.fieldnames) + ["中文意譯"]
            if "中文意譯" not in reader.fieldnames
            else list(reader.fieldnames)
        )
    else:
        fieldnames = ["transcription", "file", "中文意譯"]

# 準備翻譯資料
sentences = [row["transcription"] for row in rows]
total_sentences = len(sentences)

print(f"開始批次翻譯，總共 {total_sentences} 筆資料...")
print(f"批次大小: 8，預估時間: 約 {math.ceil(total_sentences/8/300):.1f} 小時")

# 批次翻譯
start_time = time.time()
translated_results = translate_batch(sentences, "ZH", batch_size=8)
end_time = time.time()

print(f"\n翻譯完成！總耗時: {(end_time - start_time)/3600:.2f} 小時")

# 寫入結果
print("寫入 CSV 檔案...")
with open(output_file, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for i, (row, result) in enumerate(zip(rows, translated_results)):
        row["中文意譯"] = result
        writer.writerow(row)

        # 每1000筆顯示一次進度
        if (i + 1) % 1000 == 0:
            print(f"已寫入: {i + 1}/{total_sentences}")

print(f"完成！結果已儲存至 {output_file}")
