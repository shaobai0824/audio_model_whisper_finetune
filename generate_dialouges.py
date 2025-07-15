import csv
import random


def generate_daily_conversations(num_conversations):
    """
    生成指定數量的不重複日常對話。
    """
    # 定義不同主題的詞彙和句型
    # 您可以根據需求擴充這些列表
    greetings = [
        ["你好嗎？"],
        ["早安！"],
        ["晚安。"],
        ["哈囉！"],
        ["嗨！"],
        ["最近怎麼樣？"],
    ]

    weather_phrases = [
        ["今天天氣真", ["好", "棒", "熱", "涼", "差"]],
        [["會", "可能"], "下雨嗎？"],
        ["外面", ["很", "有點"], ["熱", "冷", "涼"]],
        ["天氣", ["怎麼樣", "如何"]],
        ["今天", ["真是", "是個"], ["晴朗", "多雲", "陰天"]],
        ["希望", "明天", "是個", ["好天氣", "大晴天"]],
    ]

    meal_phrases = [
        ["吃飽了嗎？"],
        ["還沒吃。"],
        ["晚餐想吃什麼？"],
        ["我有點餓了。"],
        [["早餐", "午餐", "晚餐"], ["要不要", "想不想"], "一起吃？"],
        ["這個", ["很好吃", "味道不錯", "真美味"]],
        [["你", "喜歡"], "吃", ["什麼", "哪種"], ["食物", "料理"]],
        ["要不要", "再", ["吃一點", "來一點"]],
        ["吃完了嗎？"],
    ]

    shopping_phrases = [
        ["這個", "多少錢？"],
        ["太貴了。"],
        ["可以", "便宜一點嗎？"],
        [["這件", "這個", "那件"], ["很漂亮", "不錯", "蠻好看的"]],
        ["還有", "其他", ["顏色", "款式"], "嗎？"],
        ["我想", "買", ["這個", "那個"]],
        ["我只是", "看看。"],
    ]

    work_study_phrases = [
        ["上班了嗎？"],
        ["今天工作順利嗎？"],
        ["下班了嗎？"],
        ["作業寫完了嗎？"],
        ["學習辛苦了。"],
        [["你", "今天"], "忙嗎？"],
        ["工作", ["還好嗎？", "順利嗎？"]],
        ["週末", "要不要", "加班？"],
    ]

    leisure_phrases = [
        ["週末", "要去哪？"],
        ["我們", "去看", ["電影", "展覽", "表演"], "吧。"],
        ["有空嗎？"],
        ["你", "喜歡", "聽音樂嗎？"],
        ["要不要", "喝杯", ["咖啡", "茶", "飲料"]],
        [["你有", "最近看過", "最近聽過"], "什麼", ["好看的", "好聽的"]],
        ["下次", ["一起", "再一起"], ["出去玩", "逛街"]],
        ["無聊嗎？"],
    ]

    general_phrases = [
        ["不好意思。"],
        ["謝謝你。"],
        ["沒關係。"],
        ["我很開心。"],
        ["有點累了。"],
        ["你需要", "幫忙嗎？"],
        ["好的。"],
        ["沒問題。"],
        ["對不起。"],
        ["真的嗎？"],
        ["我不知道。"],
        ["怎麼辦？"],
        ["別擔心。"],
        ["下次再說吧。"],
        ["小心點。"],
        ["祝你", ["好運", "成功"]],
        ["下次見。"],
        ["一路順風。"],
        ["你", ["幾點睡", "睡得好嗎"]],
        ["要起床了。"],
        ["還想睡覺。"],
        ["有什麼事嗎？"],
        ["請進。"],
        ["請坐。"],
        ["我現在", ["沒空", "很忙"]],
        ["等一下。"],
        ["我馬上到。"],
        ["抱歉，我遲到了。"],
        ["味道不錯。"],
        ["最近怎麼樣？"],
        ["一切都好。"],
        ["沒什麼特別的。"],
        ["你呢？"],
        ["我很忙。"],
        ["我有點閒。"],
        ["要去哪裡？"],
        ["我在家。"],
        ["你現在在哪？"],
        ["我等等就去。"],
        ["等我一下。"],
        ["可以借我嗎？"],
        ["當然可以。"],
        ["不客氣。"],
        ["別客氣。"],
        ["真有趣。"],
        ["好無聊。"],
        ["好驚訝。"],
        ["我很感動。"],
        ["你真棒。"],
        ["加油！"],
        ["別放棄。"],
        ["相信自己。"],
        ["有什麼推薦嗎？"],
        ["這個不錯。"],
        ["可以試試看。"],
        ["我喜歡這個。"],
        ["你喜歡哪個？"],
        ["我都可以。"],
        ["你決定吧。"],
        ["隨便。"],
        ["你還好嗎？"],
        ["我很好。"],
        ["我有點不舒服。"],
        ["要不要去看醫生？"],
        ["沒關係的。"],
        ["請問。"],
        ["不好意思，打擾了。"],
        ["方便說話嗎？"],
        ["請稍等。"],
        ["明白了。"],
        ["知道了。"],
        ["可以嗎？"],
        ["好的，沒問題。"],
        ["你說什麼？"],
        ["再說一次。"],
        ["我聽不懂。"],
        ["可以再清楚一點嗎？"],
        ["真是太棒了！"],
        ["太棒了！"],
        ["真巧。"],
        ["好久不見。"],
        ["很高興見到你。"],
        ["你住哪？"],
        ["你從哪裡來？"],
        ["你去過台灣嗎？"],
        ["你喜歡台灣嗎？"],
        ["這裡很棒。"],
        ["風景很美。"],
        ["食物很好吃。"],
        ["我愛這裡。"],
        ["我喜歡這個城市。"],
        ["你做什麼工作？"],
        ["你是學生嗎？"],
        ["你學什麼？"],
        ["你喜歡你的工作嗎？"],
        ["今天過得好嗎？"],
        ["發生什麼事了？"],
        ["發生什麼事了嗎？"],
        ["你怎麼了？"],
        ["我很抱歉。"],
        ["別難過。"],
        ["振作起來。"],
        ["會有辦法的。"],
        ["加油！"],
    ]

    all_phrases_list = [
        greetings,
        weather_phrases,
        meal_phrases,
        shopping_phrases,
        work_study_phrases,
        leisure_phrases,
        general_phrases,
    ]

    generated_conversations = set()  # 使用集合來確保不重複

    while len(generated_conversations) < num_conversations:
        # 隨機選擇一個類型的對話
        selected_type = random.choice(all_phrases_list)
        selected_phrase_template = random.choice(selected_type)

        current_conversation_parts = []
        for part in selected_phrase_template:
            if isinstance(part, list):  # 如果是列表，表示有可替換的詞彙
                current_conversation_parts.append(random.choice(part))
            else:
                current_conversation_parts.append(part)

        conversation = "".join(current_conversation_parts)

        # 檢查長度是否超過25個字，且不包含逗號
        if (
            len(conversation) <= 25
            and "," not in conversation
            and "，" not in conversation
        ):
            generated_conversations.add(conversation)

        # 如果生成數量達到目標，就結束循環
        if len(generated_conversations) >= num_conversations:
            break

        # 防止無限循環，設定一個最大嘗試次數
        if (
            len(generated_conversations) > num_conversations * 5
            and num_conversations > 1000
        ):
            print("警告：嘗試次數過多，可能難以生成足夠的不重複對話。")
            break

    return list(generated_conversations)


def export_to_csv(conversations, filename="audio_make_csv/daily_conversations.csv"):
    """
    將對話內容匯出為CSV檔案。
    """
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["對話內容"])  # 寫入標題行
        for conv in conversations:
            csv_writer.writerow([conv])
    print(f"已成功生成並匯出 {len(conversations)} 個對話到 {filename}")


if __name__ == "__main__":
    num_to_generate = 10000  # 您想要的對話數量

    print(f"正在嘗試生成 {num_to_generate} 個不重複的日常對話...")
    conversations = generate_daily_conversations(num_to_generate)

    # 如果生成的對話數量少於目標，提示使用者
    if len(conversations) < num_to_generate:
        print(
            f"注意：只生成了 {len(conversations)} 個不重複的對話，未能達到 {num_to_generate} 的目標。"
        )
        print("您可以嘗試擴充程式碼中的詞彙和句型列表，以增加多樣性。")
    else:
        print(f"成功生成 {len(conversations)} 個不重複的對話。")

    export_to_csv(conversations)
