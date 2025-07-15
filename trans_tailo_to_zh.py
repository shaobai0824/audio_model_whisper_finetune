import csv
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "Bohanlu/Taigi-Llama-2-Translator-7B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

PROMPT_TEMPLATE = "[TRANS]\n{source_sentence}\n[/TRANS]\n[{target_language}]\n"


def translate(source_sentence: str, target_language: str = "ZH") -> str:
    prompt = PROMPT_TEMPLATE.format(
        source_sentence=source_sentence, target_language=target_language
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=256, do_sample=False, repetition_penalty=1.1
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("DEBUG raw output:", repr(result))  # 保留debug
    return result  # 直接回傳完整內容


def extract_zh_content(text: str) -> str:
    start = text.find("[ZH]")
    end = text.find("[/ZH]")
    if start != -1 and end != -1 and start < end:
        return text[start + 4 : end].strip()
    return ""


# 範例
print(translate("紲落來看新竹市明仔載二十六號的天氣"))

input_file = "output/final_audio_paths_zh.csv"
output_file = "output_zh.csv"

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

with open(output_file, "w", encoding="utf-8", newline="") as fout:
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        tai_sentence = row["transcription"]
        try:
            result = translate(tai_sentence, "ZH")
            zh_content = extract_zh_content(result)
        except Exception as e:
            zh_content = ""
            print(f"第{idx}筆翻譯失敗: {e}")
        row["中文意譯"] = zh_content
        print(f"{idx}/{total}: {tai_sentence} → {zh_content}")
        writer.writerow(row)
