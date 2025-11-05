import re
import json as pyjson
from Gemini import gemini_generate 

def strip_markdown(text):
    """移除常見的 Markdown 格式，回傳純文字"""
    if not text:
        return ""
    # 移除標題符號 (例如 #, ##)
    text = re.sub(r'^\s*#+\s*', '', text, flags=re.MULTILINE)
    # 移除粗體和斜體 (例如 **, *, __, _)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # 移除列表符號 (例如 *, -, +)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    # 移除行內程式碼反引號
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text

def get_chat_reply(user_msg, history, logger):    
    context = ""
    for msg in history:
        role = "你" if msg["sender"] == "user" else "AI"
        context += f"{role}: {msg['text']}\n"
    prompt = (
        "你是一個友善的聊天助手，請務必使用繁體中文回答。\n\n"
        f"{context}你: {user_msg}\nAI:"
    )
    logger.info(f"[Chat Prompt] Sending to Gemini: {prompt}")
    raw_reply = gemini_generate(prompt, max_output_tokens=1024, temperature=0.7)
    return strip_markdown(raw_reply)

def get_emotion_suggestion(emotion_stats, line_series, transcript, logger):
    prompt = (
        "你是一位心理諮商師，請用繁體中文回覆，並嚴格遵守以下格式與規範：\n"
        "任務：根據使用者的情緒統計、時間序列，以及語音轉文字摘要，"
        "先提供 1 句同理且務實的建議，再列出 3–4 點具體、可執行的調節方法。\n\n"
        f"【情緒分布】{pyjson.dumps(emotion_stats, ensure_ascii=False)}\n"
        f"【時間序列】{line_series}\n"
        f"【語音轉文字（摘要）】{transcript}\n\n"
        "請遵守以下規範：\n"
        "1) 以情緒分布中佔比最高的情緒作為主要關注點，若情緒差異不大，則以語音內容判斷主要情緒\n"
        "2) 必須同時參考情緒分布、時間序列趨勢與語音內容，不能只依賴其中之一\n"
        "3) 口吻自然、友善、非醫療診斷\n"
        "4) 方法務實清楚，務必為可立即採取的行動，例如「深呼吸 3 次」、「散步 5 分鐘」、「寫下 3 件感恩的事」。避免使用空泛詞語如「放輕鬆」。\n"
        "5) 若時間序列顯示情緒惡化，先安撫再給穩定情緒的方法；若顯示好轉，先肯定再提供維持的方法。\n"
        "6) 輸出格式必須固定如下（不得省略標題與編號）：\n"
        "【同理建議】\n(一句溫暖務實的話)\n"
        "【情緒調節方法】\n1. ...\n2. ...\n3. ...\n4. ...\n"
        "7) 請勿使用任何 Markdown 語法 (例如 #, *, -)，直接輸出純文字。\n"
    )
    logger.info(f"[Suggestion Prompt] Sending to Gemini...")
    raw_suggestion = gemini_generate(prompt, max_output_tokens=4096, temperature=0.6) or "目前暫無建議，請稍後再試。"
    return strip_markdown(raw_suggestion)

def get_chat_summary(chat_log, logger):
    prompt = (
        "你是一位專業的心理諮商師助理，請使用繁體中文回覆。\n"
        "你的任務是總結以下使用者與AI助理的聊天紀錄，找出關鍵的情緒主題、潛在問題，並為諮商師提供具體的應對建議。\n\n"
        f"【聊天紀錄】\n{chat_log}\n\n"
        "請根據以上內容，嚴格遵守以下格式輸出你的分析報告：\n"
        "【聊天總結】\n(這裡簡要總結對話的核心內容，約 2-3句話)\n\n"
        "【主要情緒與議題】\n(這裡列點說明觀察到的主要情緒，例如：焦慮、低落感、人際關係困擾等)\n\n"
        "【給諮商師的建議】\n(這裡列點提供具體、可操作的建議，幫助諮商師在下次會談時可以切入的重點或可以使用的技巧)\n"
        "最後，請注意：你的回覆中，請勿包含任何 Markdown 格式語法 (例如 #, *, -)，所有內容都必須是純文字。"
    )
    logger.info(f"[Summary Prompt] Sending to Gemini...")
    raw_summary = gemini_generate(prompt, max_output_tokens=4096, temperature=0.5)
    return strip_markdown(raw_summary)