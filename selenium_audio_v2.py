import base64
import csv
import datetime
import os
import random
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def safe_filename(text, maxlen=40):
    name = re.sub(r'[\s\\/:*?"<>|\n\r\t]', "_", str(text)).strip()
    return name[:maxlen]


df = pd.read_csv("output_zh_optimized_dedup_part2.csv")
os.makedirs("audio_files", exist_ok=True)

done_set = set()
try:
    with open("tts.csv", "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            done_set.add(row[1])
except FileNotFoundError:
    pass

MAX_RETRY = 2
results = []

driver = webdriver.Chrome()
driver.get("http://tts001.iptcloud.net:8804/")

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

            # 不刷新頁面，直接操作
            input_box = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#js-input-zh"))
            )
            input_box.clear()
            driver.execute_script("document.querySelector('#audio1').src = '';")
            input_box.send_keys(chinese_text)

            # 點擊產生台文
            driver.find_element(By.CSS_SELECTOR, "#js-translate_taigi_zh_tw_py").click()
            # 等待台文產生（可根據台文欄位內容變化來判斷，這裡用 sleep 0.5 秒）
            time.sleep(0.5)

            # 點擊產生音檔
            driver.find_element(By.CSS_SELECTOR, "#button1").click()

            # 等待 audio1 的 src 變化
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

            WebDriverWait(driver, 15).until(src_changed)
            # 等待音檔產生（可嘗試縮短到 0.5 秒）
            time.sleep(0.5)

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
            if os.path.getsize(audio_path) < 1024:
                print(f"警告：第{idx}筆音檔過小，疑似失敗，略過。")
                os.remove(audio_path)
                continue
            results.append([audio_path, chinese_text])
            end_time = time.time()
            print(
                f"第{idx}筆結束抓取：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，耗時：{end_time - start_time:.2f} 秒"
            )
            # 隨機 sleep 可縮短到 0.5~1 秒
            time.sleep(random.uniform(0.5, 1))
            break
        except Exception as e:
            print(f"第{idx}筆重試{attempt+1}失敗，內容：{chinese_text}，錯誤：{e}")
            if attempt == MAX_RETRY - 1:
                print(f"第{idx}筆最終失敗，略過。")
            time.sleep(2)

if results:
    with open("tts.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(results)

driver.quit()
