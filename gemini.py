import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# 設定 API 金鑰
genai.configure(api_key=os.environ["AI_API_KEY"])

def generate_response(bot_prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2000,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(
            contents=[bot_prompt],
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print(response)
        return response.text.strip() if response.candidates else "無法生成回應。"
    except Exception as e:
        return f"Error: {str(e)}"