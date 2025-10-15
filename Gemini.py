import os
import google.generativeai as genai
# --- 1. 初始化設定 (當這個檔案被 import 時，這段程式碼就會自動執行) ---

# 從環境變數讀取您的 Gemini API 金鑰
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    # 如果找不到金鑰，直接拋出錯誤，避免後續執行失敗
    raise ValueError("環境變數 'GEMINI_API_KEY' 未設定，請先設定您的 API 金鑰。")

# 使用您的 API 金鑰設定 SDK
genai.configure(api_key=api_key)

# 建立 Gemini 模型物件
gemini_model = genai.GenerativeModel('gemini-2.5-flash')


# --- 2. 可被外部呼叫的 generate 函式 ---

def gemini_generate(prompt: str, temperature: float = 0.7, max_output_tokens: int = 2048, timeout: int = 120) -> str:
    """
    呼叫 Google Gemini API 來生成文字。
    """
    try:
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": timeout}
        )
        return response.text.strip()
    except Exception as e:
        # 在伺服器環境中，印出錯誤是很好的除錯習慣
        print(f"[Gemini] 呼叫 Gemini API 失敗：{e}")
        return ""