import base64
import csv
import datetime
import os
import random
import re
import time

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# 1. 讀取原始 CSV
df = pd.read_csv("output_zh_optimized_dedup_part1.csv")

BATCH_SIZE = 1000
SAVE_INTERVAL = 100
os.makedirs("audio_files", exist_ok=True)
# 讀取已存在的 tts.csv，取得已完成的中文句子集合
done_set = set()
try:
    with open("tts.csv", "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            done_set.add(row[1])  # 假設第二欄是中文
except FileNotFoundError:
    pass


def safe_filename(text, maxlen=40):
    # 移除不合法字元與不可見字元
    name = re.sub(r'[\s\\/:*?"<>|\n\r\t]', "_", text).strip()
    # 避免檔名過長
    return name[:maxlen]


MAX_RETRY = 2

# 2. Selenium 操作
driver = webdriver.Chrome()
driver.get("http://tts001.iptcloud.net:8804/")

results = []

for idx, row in df.iterrows():
    chinese_text = row["chinese_text"]
    if chinese_text in done_set:
        continue

    for attempt in range(MAX_RETRY):
        try:
            start_time = time.time()
            print(
                f"第{idx}筆開始抓取：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，內容：{chinese_text}"
            )

            driver.refresh()  # 每筆都刷新頁面
            time.sleep(2)  # 等待頁面載入

            input_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#js-input-zh"))
            )
            input_box.clear()
            driver.execute_script("document.querySelector('#audio1').src = '';")
            input_box.send_keys(chinese_text)
            driver.find_element(By.CSS_SELECTOR, "#js-translate_taigi_zh_tw_py").click()
            time.sleep(1)
            driver.find_element(By.CSS_SELECTOR, "#button1").click()

            prev_src = ""
            try:
                prev_src = driver.find_element(
                    By.CSS_SELECTOR, "#audio1"
                ).get_attribute("src")
            except Exception:
                prev_src = ""

            def src_changed(driver):
                src = driver.find_element(By.CSS_SELECTOR, "#audio1").get_attribute(
                    "src"
                )
                return src and src != prev_src

            WebDriverWait(driver, 20).until(src_changed)
            time.sleep(2)  # 再多等2秒，確保音檔產生

            # 下載 blob
            js = """
window.blobBase64 = null;
const audio = document.querySelector('#audio1');
const xhr = new XMLHttpRequest();
xhr.open('GET', audio.src, true);
xhr.responseType = 'blob';
xhr.onload = function() {
    const reader = new FileReader();
    reader.onloadend = function() {
        window.blobBase64 = reader.result.split(',')[1];
    };
    reader.readAsDataURL(xhr.response);
};
xhr.send();
return new Promise(resolve => {
    (function waitForBlob(){
        if(window.blobBase64) return resolve(window.blobBase64);
        setTimeout(waitForBlob, 100);
    })();
});
"""
            audio_base64 = driver.execute_script(js)
            filename = safe_filename(chinese_text) + ".mp3"
            audio_path = os.path.join("audio_files", filename)
            if os.path.exists(audio_path):
                filename = safe_filename(chinese_text) + f"_{idx}.mp3"
                audio_path = os.path.join("audio_files", filename)
            if not audio_base64:
                print(f"警告：第{idx}筆未取得音檔，略過。")
                continue
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            # 檢查檔案大小
            if os.path.getsize(audio_path) < 1024:
                print(f"警告：第{idx}筆音檔過小，疑似失敗，略過。")
                os.remove(audio_path)
                continue
            results.append([audio_path, chinese_text])
            end_time = time.time()
            print(
                f"第{idx}筆結束抓取：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，耗時：{end_time - start_time:.2f} 秒"
            )
            time.sleep(random.uniform(1, 2))
            break
        except Exception as e:
            print(f"第{idx}筆重試{attempt+1}失敗，內容：{chinese_text}，錯誤：{e}")
            if attempt == MAX_RETRY - 1:
                print(f"第{idx}筆最終失敗，略過。")
            time.sleep(3)

# 批次結束後儲存剩餘資料
if results:
    with open("tts.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(results)

# 關閉瀏覽器
driver.quit()
